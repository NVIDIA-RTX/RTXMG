/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

// ClusterLodStreaming — the demand-paging runtime for cluster-LoD geometry, and
// the default residency backend.  ClusterLodPreloaded is the same interface
// with none of this; read that one first if this file is where you started.
//
// The unit of streaming is a GROUP (GeometryGroup{geometryID, groupID}), not a
// cluster: a group is what the baker emits, what the LoD DAG links, and what
// gets one contiguous blob in the geometry pool.  At scene load only the
// coarsest group of every geometry is uploaded (InitGeometries); its CLAS and a
// one-per-geometry low-detail BLAS are built once (InitClas) and stay resident
// for the life of the scene, so every instance always has something to draw.
// Everything finer is paged in and out.
//
// The CPU/GPU split is the idea worth taking away: the GPU decides what is
// wanted, the host decides what is affordable.  Traversal appends load requests
// and resets the age of every group it used; stream_age_groups appends unload
// requests for the ones that aged out.  Both land in one host-mapped ring that
// CaptureFrameRequests copies toward a readback.  One frame later
// HandleCompletedRequest pops it and accepts as much as four independent
// budgets allow — upload-ring bytes, geometry-pool suballocation, resident
// group/cluster slots, and CLAS bytes.  A refusal is counted, never fatal: the
// group is simply requested again next frame.
//
// Three task rings of kStreamingMaxActiveTasks slots (requests -> storage ->
// updates) carry the work, each slot released by an nvrhi EventQuery against
// the submission that consumed it.  That is where the one-frame request latency
// comes from, and why nothing on this path ever blocks on the GPU.
//
// What the host accepted becomes a patch list that the GPU applies the next
// frame, through the hooks IClusterLodStreamingHooks orders:
//
//   StageResidencyUpdate   Host only.  Drains the rings, applies the budgets,
//                          and memcpys the accepted blobs into the mapped
//                          upload heap (FillStreamingGroupData, which also
//                          strips positions/normals and rewrites each cluster's
//                          vertex slice).
//   ApplyResidencyUpdate   stream_allocator_free_groups -> stream_update_scene
//                          (patches every geometry's GroupAddress table, so
//                          traversal can see the new residency) ->
//                          stream_fill_clas_geometry_indices -> Implicit CLAS
//                          build into scratch -> CLAS-allocator prepass.
//   FinalizeResidency      Runs after traversal.  stream_age_groups ->
//                          stream_allocator_alloc_groups picks pool
//                          destinations -> cluster-level Move from scratch into
//                          the persistent CLAS pool.
//   CaptureFrameRequests   Copies the request ring toward next frame's readback.
//
// Two interchangeable CLAS allocators sit behind
// StreamingConfig::usePersistentClasAllocator: a persistent free-list/bin
// allocator (default, five shaders) and an always-defrag compaction allocator
// (two).  Much of the three functions above is the if/else between them.
//
// Worth knowing before porting this: the streaming dispatches below carry
// almost no explicit UAV barriers.  Their ordering comes from nvrhi's automatic
// barrier in setComputeState, derived from the binding set.  pass.cpp and
// blas_pass.cpp barrier explicitly instead.
//
// State that survives across frames: the resident group/cluster ID pools and
// active list (m_resident), the geometry-pool suballocator (m_storage),
// per-geometry loaded-group counts and cached-BLAS bookkeeping
// (m_persistentGeometries), and a GPU-only StreamingResidentPersistent block
// the host deliberately never re-uploads.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <unordered_set>

#include <donut/core/log.h>
#include <nvrhi/nvrhiHLSL.h>      // for nvrhi::rt::cluster::IndirectArgs (BLAS-build descriptor)
#include <nvrhi/utils.h>          // for BufferUavBarrier

#include "rtxmg/cluster_lod/group_codec.h"  // DecompressGroup
#include "rtxmg/cluster_lod/streaming.h"
#include "rtxmg/profiler/statistics.h"  // per-phase Cluster Lod GPU timers (AccelBuilder tab)
#include "rtxmg/utils/bindless_layout.h"
#include "rtxmg/utils/cluster_operation.h"

namespace rtxmg {

bool ClusterLodStreaming::Init(const std::vector<GeometryView>&        geometries,
                               const std::vector<ClusterLodInstance>&  instances,
                               const BakerConfig&                      bakerConfig,
                               uint32_t                                maxClusterTriangles,
                               uint32_t                                maxClusterVertices,
                               bool                                    hasAlphaMask,
                               uint32_t                                clusterLodMaterialBaseID,
                               const StreamingConfig&                  config,
                               donut::engine::DescriptorTableManager*  descriptorTable,
                               donut::engine::ShaderFactory*           shaderFactory,
                               nvrhi::IDevice*                         device,
                               nvrhi::ICommandList*                    commandList)
{
    m_clusterLodMaterialBaseID = clusterLodMaterialBaseID;
    assert(!m_device && "no init before deinit");
    assert(device && shaderFactory);

    m_device              = device;
    m_shaderFactory       = shaderFactory;
    m_config              = config;
    // Stripping positions requires the hit shader to fetch them from the AS
    // instead; without the intrinsic the pool simply keeps them.
    if (m_config.stripResidentPositions
        && !device->queryFeatureSupport(nvrhi::Feature::RayTracingPositionFetch))
    {
        donut::log::warning("Cluster-LOD: AS position fetch unsupported on this device — "
                            "keeping positions resident (geometry pool ~12 B/vertex larger)");
        m_config.stripResidentPositions = false;
    }
    m_bakerConfig         = bakerConfig;
    m_clasPositionTruncateBits =
        ResolveClasPositionTruncateBits(m_config.clasPositionTruncateBits, bakerConfig);
    // Cache-loaded configs bypass the BakerConfig default the static_assert sees.
    assert(bakerConfig.clusterGroupSize <= shaderio::kTraversalBlasMergingMaxGroupClusters);
    m_maxClusterTriangles = maxClusterTriangles;
    m_maxClusterVertices  = maxClusterVertices;
    m_maxClustersPerGroup = bakerConfig.clusterGroupSize;
    m_geometries          = geometries;  // shallow copy of spans — scene owns the underlying storage
    // m_hasAlphaMaskScene is the base-class mirror, so the shared CLAS-build
    // helpers see the flag from either path.
    m_hasAlphaMask        = hasAlphaMask;
    m_hasAlphaMaskScene   = hasAlphaMask;
    m_enableMaterials     = config.enableMaterials;
    m_descriptorTable     = descriptorTable->GetDescriptorTable();
    // The manager itself (not just its table) — the cached-BLAS pool needs it to
    // create per-block descriptors.
    m_descriptorTableManager = descriptorTable;

    m_shaders   = {};

    m_requiresClas            = false;
    m_frameIndex              = 1;  // intentionally start at 1
    m_operationsSize          = 0;
    m_persistentGeometrySize  = 0;
    m_blasSize                = 0;
    m_clasOperationsSize      = 0;
    m_clasLowDetailSize       = 0;
    m_clasSingleMaxSize       = 0;
    m_clasScratchNewClasSize  = 0;
    m_stats                   = {};

    // some adjustments are required to make the config compatible
    // need at least all lo-res groups of all geometries
    const uint32_t geometryCount = uint32_t(geometries.size());
    m_config.maxGroups = std::max(m_config.maxGroups, geometryCount);
    if (m_config.maxClusters == 0)
    {
        m_config.maxClusters = m_config.maxGroups * bakerConfig.clusterGroupSize;
    }
    m_config.maxClusters = std::max(m_config.maxClusters, geometryCount * bakerConfig.clusterGroupSize);

    m_stats.maxLoadCount     = m_config.maxPerFrameLoadRequests;
    m_stats.maxUnloadCount   = m_config.maxPerFrameUnloadRequests;
    m_stats.maxGroups        = m_config.maxGroups;
    m_stats.maxClusters      = m_config.maxClusters;
    m_stats.maxTransferBytes = uint64_t(m_config.maxTransferMegaBytes) * 1024 * 1024;

    // One-time load-cost breakdown, logged at the end of init (large scenes
    // spend tens of seconds here).
    const auto tInit0 = std::chrono::steady_clock::now();
    auto msSince = [](std::chrono::steady_clock::time_point t0)
    {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    };

    if (!InitShadersAndPipelines(shaderFactory, device))
    {
        return false;
    }
    const double msPipelines = msSince(tInit0);
    const auto tManagers0 = std::chrono::steady_clock::now();

    const uint32_t groupCountAlignment =
        std::max(shaderio::kStreamAgeFilterGroupsThreads, shaderio::kStreamUpdateSceneThreads);
    const uint32_t clusterCountAlignment = 64;

    // setup streaming management
    m_requestsTaskQueue.Init(device);
    m_updatesTaskQueue.Init(device);
    m_storageTaskQueue.Init(device);

    m_requests.Init(device, m_config, groupCountAlignment, clusterCountAlignment);
    m_resident.Init(device, m_config, groupCountAlignment, clusterCountAlignment);
    m_updates.Init(device, m_config, geometryCount, groupCountAlignment, clusterCountAlignment);
    m_storage.Init(device, descriptorTable, m_config);

    // storage uses block allocator, max may be less than what we asked for
    m_stats.maxDataBytes = m_storage.GetMaxDataSize();

    m_operationsSize += m_requests.GetOperationsSize();
    m_operationsSize += m_resident.GetOperationsSize();
    m_operationsSize += m_updates.GetOperationsSize();
    m_operationsSize += m_storage.GetOperationsSize();

    {
        nvrhi::BufferDesc d;
        d.byteSize           = sizeof(shaderio::SceneStreaming);
        d.structStride       = sizeof(shaderio::SceneStreaming);  // bound as StructuredBuffer<SceneStreaming> at u7
        d.canHaveUAVs        = true;
        d.canHaveRawViews    = true;
        d.isDrawIndirectArgs = true;
        // Serves as the srcInfosCount source (update.moveClasCounter) for the
        // CLAS move ops, which VVL requires BUILD_INPUT_READ_ONLY usage for.
        d.isAccelStructBuildInput = true;
        d.initialState       = nvrhi::ResourceStates::Common;
        d.keepInitialState   = true;
        d.debugName          = "ClusterLodStreamingShaderBuffer";
        m_shaderBuffer.Create(d, device);
        m_operationsSize += m_shaderBuffer.GetVramBytes();
    }

    // Placeholder for streaming slots with no backing buffer in the current
    // configuration (see UpdateBindings).  Its generic desc carries canHaveUAVs
    // + canHaveRawViews + structStride=4, satisfying both the RawBuffer_UAV and
    // StructuredBuffer_UAV<uint> binding shapes.
    m_streamingDummyBuffer.Create(1, "ClusterLodStreamingDummy", device);

    const double msManagers = msSince(tManagers0);
    const auto tGeoms0 = std::chrono::steady_clock::now();

    // seed lo res geometry
    InitGeometries(geometries, instances, descriptorTable, device, commandList);

    donut::log::info("ClusterLodStreaming init: pipelines %.0f ms, sub-managers %.0f ms, "
                     "InitGeometries %.0f ms (%u geometries), total %.0f ms",
                     msPipelines, msManagers, msSince(tGeoms0), geometryCount, msSince(tInit0));

    return true;
}

void ClusterLodStreaming::UpdateBindings(nvrhi::ICommandList* /*commandList*/)
{
    if (!m_shaderBuffer || !m_streamingBindingLayout)
        return;  // initClas hasn't run yet — no resident pool / allocator yet

    const shaderio::StreamingUpdate& update = m_shaderData.update;
    const uint64_t patchesOffset = m_updates.GetPatchesByteOffsetForTask(update.taskIndex);
    // Sized by patchGroupsCount, floored at one element: on frames with no
    // pending update the count is 0 and a 0-sized view is invalid.
    const uint64_t patchesSize   = std::max<uint64_t>(
        sizeof(shaderio::StreamingPatch) * uint64_t(update.patchGroupsCount),
        sizeof(shaderio::StreamingPatch));  // never 0-sized

    // BLAS caching: t3 = this task's StreamingGeometryPatch slot, which
    // stream_update_scene reads to write each geometry's cachedBlas*.  Null when
    // caching is off → the dummy stands in and patchCachedBlasCount is 0.
    nvrhi::IBuffer* geometryPatchesBuf = m_updates.GetGeometryPatchesBuffer();
    const uint64_t geometryPatchesOffset = geometryPatchesBuf
        ? m_updates.GetGeometryPatchesByteOffsetForTask(update.taskIndex) : 0ull;
    const uint64_t geometryPatchesSize   = std::max<uint64_t>(
        sizeof(shaderio::StreamingGeometryPatch) * uint64_t(update.patchCachedBlasCount),
        sizeof(shaderio::StreamingGeometryPatch));  // never 0-sized

    // u8 u_ActiveGroups is bound WHOLE: skipping the persistent low-detail
    // prefix by descriptor offset is impossible on Vulkan (storage-buffer
    // offsets must be 16B-aligned, the prefix is an arbitrary group count), so
    // the shaders add streamingRW[0].resident.persistentGroupsCount instead.

    // agefilter UAV: bind the FULL slot ring of the request buffer.  The
    // shader computes its per-task write offset as
    //   taskIndex * taskSlotStride + unloadGroupsOffsetElems + unloadOffset
    // so one binding works for any in-flight taskIndex.
    const uint64_t slotRingBytes = m_requests.GetRequestSlotSize()
                                 * uint64_t(kStreamingMaxActiveTasks);

    // Slots go null in the compaction path (no allocator management buffer / no
    // groupClasSizes) and when CLAS is off entirely (nothing initClas creates
    // exists).  Neither configuration dispatches the PSOs that read them, so
    // substituting the dummy just keeps the unified BindingSet valid.
    nvrhi::IBuffer* dummy = m_streamingDummyBuffer.GetBuffer().Get();
    auto orDummy = [dummy](nvrhi::IBuffer* real) {
        return real ? real : dummy;
    };

    nvrhi::BindingSetDesc bsDesc;
    bsDesc.bindings = {
        // SRVs
        nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_updates.GetPatchesBuffer(),
                                                    nvrhi::Format::UNKNOWN,
                                                    nvrhi::BufferRange{ patchesOffset, patchesSize }),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(1, orDummy(m_updates.GetNewClasSizesBuffer())),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(2, orDummy(m_updates.GetNewClasAddressesBuffer())),
        geometryPatchesBuf
            ? nvrhi::BindingSetItem::StructuredBuffer_SRV(3, geometryPatchesBuf,
                                                          nvrhi::Format::UNKNOWN,
                                                          nvrhi::BufferRange{ geometryPatchesOffset, geometryPatchesSize })
            : nvrhi::BindingSetItem::StructuredBuffer_SRV(3, dummy),
        // UAVs
        nvrhi::BindingSetItem::StructuredBuffer_UAV(0,  m_resident.GetGroupsBuffer().GetBuffer()),
        nvrhi::BindingSetItem::RawBuffer_UAV(1,         orDummy(m_clasAllocator.GetManagementBuffer())),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(2,  orDummy(m_resident.GetClasAddressesBuffer().GetBuffer().Get())),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(3,  orDummy(m_resident.GetClasSizesBuffer())),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(4,  orDummy(m_resident.GetGroupClasSizesBuffer())),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(5,  orDummy(m_updates.GetMoveClasSrcAddressesBuffer())),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(6,  orDummy(m_updates.GetMoveClasDstAddressesBuffer())),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(7,  m_shaderBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(8,  m_resident.GetActiveGroupsBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(9,  m_resident.GetGroupIDsBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(10, m_resident.GetClustersBuffer().GetBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(11, orDummy(m_clasIndirectArgsBuffer)),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(12, m_requests.GetRequestBuffer(),
                                                    nvrhi::Format::UNKNOWN,
                                                    nvrhi::BufferRange{ 0, slotRingBytes }),
        // Mixed-cluster geometry-indices task queue: stream_update_scene stamps
        // each slot with (srvIndex, byteOffset) of the source group blob;
        // stream_fill_clas_geometry_indices reads that, walks the per-triangle
        // material bytes, and overwrites the slot with the CLAS entries.
        nvrhi::BindingSetItem::StructuredBuffer_UAV(13, orDummy(m_updates.GetNewClasGeometryIndicesBuffer())),
        // Compaction allocator: argIdx → clusterResidentID map (written by
        // stream_update_scene, read by stream_compact_append_new).
        nvrhi::BindingSetItem::StructuredBuffer_UAV(14, orDummy(m_updates.GetNewClasResidentIDsBuffer())),
        // Geometries bound as a UAV (not SRV) so stream_update_scene can write
        // cachedBlas*; agefilter + allocator unload read it through the same
        // RWStructuredBuffer, avoiding an SRV/UAV state conflict on one buffer.
        nvrhi::BindingSetItem::StructuredBuffer_UAV(15, m_shaderGeometriesBuffer.GetBuffer()),
    };
    m_streamingBindingSet = m_device->createBindingSet(bsDesc, m_streamingBindingLayout);

    // Compaction allocator: a raw (RWByteAddressBuffer) view of the
    // SceneStreaming aggregate for the 64-bit moveClasSize InterlockedAdd64,
    // in its own register space so it never interacts with the bindless heap.
    if (m_compactionRawLayout)
    {
        nvrhi::BindingSetDesc rawDesc;
        rawDesc.bindings = { nvrhi::BindingSetItem::RawBuffer_UAV(0, m_shaderBuffer) };
        m_compactionRawSet = m_device->createBindingSet(rawDesc, m_compactionRawLayout);
    }

    // GPU-persistent allocator scalars (space2 u0), bound only to stream_dispatch_setup.
    if (m_residentPersistentLayout)
    {
        nvrhi::BindingSetDesc persistentDesc;
        persistentDesc.bindings = { nvrhi::BindingSetItem::StructuredBuffer_UAV(0,
                                        orDummy(m_resident.GetResidentPersistentBuffer())) };
        m_residentPersistentSet = m_device->createBindingSet(persistentDesc, m_residentPersistentLayout);
    }

    // stream_dispatch_setup.hlsl has the same bindings plus a push constant at b0.
    bsDesc.bindings.push_back(nvrhi::BindingSetItem::PushConstants(0, sizeof(uint32_t)));
    m_setupBindingSet = m_device->createBindingSet(bsDesc, m_setupBindingLayout);
}

const std::vector<nvrhi::IBuffer*>* ClusterLodStreaming::GetCachedBlasPoolBlocks()
{
    // Rebuild each frame — the lazy pool grows blocks on demand, so the block
    // set isn't stable.  Empty (but non-null) before the pool exists.
    m_cachedBlasPoolBlocksView.clear();
    if (m_cachedBlasAllocator.isInitialized())
    {
        const uint32_t blockCount = m_cachedBlasAllocator.getBlockCount();
        m_cachedBlasPoolBlocksView.reserve(blockCount);
        for (uint32_t i = 0; i < blockCount; i++)
        {
            if (nvrhi::IBuffer* block = m_cachedBlasAllocator.getBlockBuffer(i))
                m_cachedBlasPoolBlocksView.push_back(block);
        }
    }
    return &m_cachedBlasPoolBlocksView;
}

void ClusterLodStreaming::ResetCachedBlas(nvrhi::ICommandList* commandList)
{
    if (!m_requiresClas || !m_config.allowBlasCaching)
    {
        return;
    }

    for (size_t geometryIndex = 0; geometryIndex < m_persistentGeometries.size(); geometryIndex++)
    {
        PersistentGeometry& persistentGeometry = m_persistentGeometries[geometryIndex];

        persistentGeometry.cachedBlasUpdateFrame = 0;
        persistentGeometry.cachedBlasLevel       = shaderio::kTraversalInvalidLodLevel;
        if (persistentGeometry.cachedBlasAllocation)
        {
            m_cachedBlasAllocator.subFree(persistentGeometry.cachedBlasAllocation);
        }

        // The freed allocations leave a dangling cachedBlasAddress behind.
        m_shaderGeometries[geometryIndex].cachedBlasLodLevel = shaderio::kTraversalInvalidLodLevel;
        m_shaderGeometries[geometryIndex].cachedBlasAddress  = 0;
        m_shaderGeometries[geometryIndex].cachedBlasTriangles = 0;
        m_shaderGeometries[geometryIndex].cachedBlasClusters  = 0;
    }

    commandList->writeBuffer(m_shaderGeometriesBuffer.GetBuffer(),
                             m_shaderGeometries.data(),
                             m_shaderGeometries.size() * sizeof(shaderio::Geometry));
}

void ClusterLodStreaming::ResetGeometryStreamingState(nvrhi::ICommandList* commandList)
{
    // Restore per-geometry group-address sentinels.  The persistent
    // low-detail group is always resident.

    for (size_t geometryIndex = 0; geometryIndex < m_persistentGeometries.size(); geometryIndex++)
    {
        PersistentGeometry& persistentGeometry = m_persistentGeometries[geometryIndex];

        const GeometryMetadataSlice<shaderio::GroupAddress>& slice =
            persistentGeometry.streamingGroupAddresses;
        std::vector<shaderio::GroupAddress> groupAddresses(
            slice.numElements, shaderio::GroupAddress{ shaderio::kStreamingInvalidSrvIndex, 0u });

        const StreamingResident::Group* residentGroup =
            m_resident.FindGroup(GeometryGroup{uint32_t(geometryIndex),
                                               persistentGeometry.lastLodGroupOffset});
        assert(residentGroup);
        groupAddresses[persistentGeometry.lastLodGroupOffset] = residentGroup->groupAddress;
        commandList->writeBuffer(slice.buffer, groupAddresses.data(), slice.ByteSize(),
                                 slice.ByteOffset());

        // also reset the number of groups loaded per lod-level, except last which is also always loaded.
        const uint32_t maxLodLevel = persistentGeometry.lodLevelsCount - 1;
        for (uint32_t i = 0; i < maxLodLevel; i++)
        {
            persistentGeometry.lodLoadedGroupsCount[i] = 0;
        }
        persistentGeometry.lodLoadedGroupsCount[maxLodLevel] = 1;
    }
}

// ---------------------------------------------------------------------------
// Channel stripping (StreamingConfig::stripResidentPositions / ...Normals).
//
// Positions and normals can independently be dropped from a group's resident
// pool copy; either one takes the upload through the same compacted rewrite,
// which recomputes every cluster's vertex slice and rewrites cluster.vertices.
// Stripped positions still ride the upload ring as a no-copy range the CLAS
// build reads by GPU VA (StreamingPatch::positionsGpuAddress); the cluster's
// ClusterAttribute::StrippedVertexPos steers both the hit shader (which
// fetches from the AS) and stream_update_scene's CLAS-build arg fill.
//
// Layout invariant shared with cluster_lod_payload.hlsli (ClusterLodNormalByteBase /
// ClusterLodTex0ByteBase): per rewritten cluster the vertex slice is
// [positions 12*vc, if kept] [packed normals 4*vc, if kept] [pad to 8B] [uv0]
// [pad to 8B] [uv1], with cluster.vertices pointing at the slice start — note
// the pad falls AFTER the normals, not between the two leading arrays.  The 8B
// alignment is on header-relative offsets, and group blobs are 16B-aligned in
// the pool, so the arithmetic here matches the shader's absolute arithmetic.
// ---------------------------------------------------------------------------

// Size of the stripped device blob + the diverted position bytes.  Delegates to
// the shared baked_geometry.h helpers so the Inspector's residency accounting
// uses the same formula as the pool allocation.  Walks cluster headers only
// (counts and attribute bits are layout-invariant even for compressed sources),
// so it can run at allocation time before any vertex bytes are touched.
static void ComputeStrippedGroupSizes(const GroupInfo&         info,
                                      const shaderio::Cluster* clusters,
                                      bool                     stripPositions,
                                      bool                     stripNormals,
                                      uint64_t&                strippedBytes,
                                      uint64_t&                positionBytes)
{
    strippedBytes = ResidentGroupDeviceBytes(info, clusters, stripPositions, stripNormals);
    // Positions are diverted to the upload ring only when they leave the pool;
    // a normals-only strip keeps them in the blob for the CLAS build to read.
    positionBytes = stripPositions ? ComputeGroupAttributeBytes(info, clusters).positions : 0u;
}

// Local helper that memcpy's the baked group blob into the destination, then
// patches the allocator-issued resident IDs into the leading Group header.
// When positionsDst is non-null, positions are stripped from the resident
// copy and written there instead (see the position-stripping block comment).
// decompressScratch is caller-owned so the helper stays stateless; it is only
// touched when a compressed group also has to be strip-rewritten.
static void FillStreamingGroupData(
    const GroupInfo&      srcGroupInfo,
    const GroupView&      srcGroupView,
    std::vector<uint8_t>& decompressScratch,
    uint32_t              clusterResidentID,
    uint32_t              groupResidentID,
    void*                 dst,
    size_t                dstSize,
    void*                 positionsDst     = nullptr,
    size_t                positionsDstSize = 0,
    bool                  stripPositions   = false,
    bool                  stripNormals     = false)
{
    // The rewrite path recomputes every cluster's vertex slice, so it is
    // authoritative regardless of which channel actually leaves.
    const bool rewriteVertices = stripPositions || stripNormals;
    assert((positionsDst != nullptr) == stripPositions);

    // Compressed groups (baker stored arithmetic-packed vertex data) expand into
    // the uncompressed device layout; uncompressed groups are a straight copy.
    const uint8_t* srcUncompressed = srcGroupView.raw;
    if (srcGroupInfo.uncompressedSizeBytes != 0u)
    {
        if (!rewriteVertices)
        {
            DecompressGroup(srcGroupInfo, srcGroupView, dst, dstSize);
        }
        else
        {
            // Strip path: expand into scratch first, strip-copy below.
            decompressScratch.resize(srcGroupInfo.GetDeviceSize());
            DecompressGroup(srcGroupInfo, srcGroupView,
                            decompressScratch.data(), decompressScratch.size());
            srcUncompressed = decompressScratch.data();
        }
    }
    else if (!rewriteVertices)
    {
        assert(srcGroupView.rawSize <= dstSize);
        std::memcpy(dst, srcGroupView.raw, srcGroupView.rawSize);
    }

    if (rewriteVertices)
    {
        // ---- strip-copy: compacted blob to dst, positions to positionsDst ----
        const size_t vertexRegionOff = srcGroupInfo.ComputeUncompressedSectionSize();
        assert(vertexRegionOff <= dstSize);
        std::memcpy(dst, srcUncompressed, vertexRegionOff);  // hdrs..indices verbatim

        uint8_t*       dstBytes = static_cast<uint8_t*>(dst);
        uint8_t*       posBytes = static_cast<uint8_t*>(positionsDst);
        const size_t   hdrBase  = rtxmg::align_up(sizeof(shaderio::Group), size_t(16));
        size_t         dstOff   = vertexRegionOff;
        size_t         posOff   = 0;

        for (uint32_t c = 0; c < srcGroupInfo.clusterCount; c++)
        {
            // dst is a persistently-mapped upload heap (write-combined): every
            // read from it is uncached, so the header is taken from the cached
            // source copy and written back once at the end of the iteration.
            const size_t      hdrOff = hdrBase + sizeof(shaderio::Cluster) * c;
            shaderio::Cluster hdr    = *reinterpret_cast<const shaderio::Cluster*>(srcUncompressed + hdrOff);
            const size_t      vc     = size_t(hdr.vertexCountMinusOne) + 1;

            // Source offsets in the uncompressed layout (positions first) —
            // must mirror the unstripped shader derivation exactly.
            const size_t srcSlice   = hdrOff + hdr.vertices;
            const size_t srcPos     = srcSlice;
            const size_t srcNormals = srcSlice + vc * 12;
            const bool   hasNormal  = (hdr.attributeBits & shaderio::ClusterAttribute::VertexNormal) != 0;
            const bool   hasTex0    = (hdr.attributeBits & shaderio::ClusterAttribute::VertexTex0) != 0;
            const bool   hasTex1    = (hdr.attributeBits & shaderio::ClusterAttribute::VertexTex1) != 0;
            // UV block sizes: raw float2 array or po2-grid quantized (16B
            // base+step header + one uint/vertex) per the baked
            // ClusterEncoding flags.
            const size_t uv0Bytes   = (hdr.encodingBits & shaderio::ClusterEncoding::QuantizedTex0) ? (16 + vc * 4) : vc * 8;
            const size_t uv1Bytes   = (hdr.encodingBits & shaderio::ClusterEncoding::QuantizedTex1) ? (16 + vc * 4) : vc * 8;
            const size_t srcUv0     = rtxmg::align_up(srcNormals + (hasNormal ? vc * 4 : 0), size_t(8));
            const size_t srcUv1     = rtxmg::align_up(srcUv0 + (hasTex0 ? uv0Bytes : 0), size_t(8));

            // Positions out to the CLAS-build staging (clusters back-to-back,
            // matching stream_update_scene's running-prefix VA derivation).
            if (stripPositions)
            {
                assert(posOff + vc * 12 <= positionsDstSize);
                std::memcpy(posBytes + posOff, srcUncompressed + srcPos, vc * 12);
                posOff += vc * 12;
            }

            // Compacted destination slice — mirrors the shader's derivation for
            // whichever channels survive.  Kept positions lead the slice with no
            // interior padding: ClusterLodTex0ByteBase adds vc*12, then the normals,
            // and only then aligns to 8.
            dstOff = rtxmg::align_up(dstOff, size_t(8));
            const size_t dstSlice = dstOff;
            if (!stripPositions)
            {
                std::memcpy(dstBytes + dstOff, srcUncompressed + srcPos, vc * 12);
                dstOff += vc * 12;
            }
            if (hasNormal && !stripNormals)
            {
                std::memcpy(dstBytes + dstOff, srcUncompressed + srcNormals, vc * 4);
                dstOff += vc * 4;
            }
            if (hasTex0)
            {
                dstOff = rtxmg::align_up(dstOff, size_t(8));
                std::memcpy(dstBytes + dstOff, srcUncompressed + srcUv0, uv0Bytes);
                dstOff += uv0Bytes;
            }
            if (hasTex1)
            {
                dstOff = rtxmg::align_up(dstOff, size_t(8));
                std::memcpy(dstBytes + dstOff, srcUncompressed + srcUv1, uv1Bytes);
                dstOff += uv1Bytes;
            }

            hdr.vertices = uint32_t(dstSlice - hdrOff);
            if (stripPositions)
            {
                // Tells both the hit shader and stream_update_scene's CLAS-build
                // arg fill that positions live in the staging range instead.
                hdr.attributeBits |= shaderio::ClusterAttribute::StrippedVertexPos;
            }
            if (stripNormals)
            {
                // Clearing the bits engages the hit shader's facet-shading
                // fallback and keeps its attribute-offset derivation consistent.
                hdr.attributeBits &= ~(shaderio::ClusterAttribute::VertexNormal
                                       | shaderio::ClusterAttribute::VertexTangent);
            }
            *reinterpret_cast<shaderio::Cluster*>(dstBytes + hdrOff) = hdr;
        }
        assert(dstOff <= dstSize);
        assert(!stripPositions || posOff == positionsDstSize);
    }

    // Patch the allocator-issued resident IDs into the leading Group header.
    // Direct write rather than via GroupStorage's spans, which are built from
    // the unstripped metadata and so don't describe a stripped blob.
    shaderio::Group* dstGroup    = reinterpret_cast<shaderio::Group*>(dst);
    dstGroup->clusterResidentID  = clusterResidentID;
    dstGroup->groupResidentID    = groupResidentID;
}

void ClusterLodStreaming::InitGeometries(
    const std::vector<GeometryView>&        geometries,
    const std::vector<ClusterLodInstance>&  instances,
    donut::engine::DescriptorTableManager*  descriptorTable,
    nvrhi::IDevice*                         device,
    nvrhi::ICommandList*                    commandList)
{
    // This function uploads all persistent per-geometry data.
    // - hierarchy nodes for lod traversal
    // - lowest detail geometry group & clusters
    // - the per-geometry group-address table traversal resolves residency through
    // It also fills the geometry descriptor stored in m_shaderGeometries.

    const size_t numGeom = geometries.size();
    m_shaderGeometries.resize(numGeom);
    m_persistentGeometries.resize(numGeom);

    // Precompute per-geometry instance counts (matches preloaded.cpp's
    // single-pass approach).
    std::vector<uint32_t> instanceCount(numGeom, 0u);
    for (const ClusterLodInstance& inst : instances)
    {
        if (inst.geometryID < numGeom)
        {
            ++instanceCount[inst.geometryID];
        }
    }

    // Static scene totals for the profiler's Cluster LOD statistics.  The
    // Clusters rows count LOD0 only; the all-LOD total is a separate field.
    m_stats.geometryCount        = uint32_t(numGeom);
    m_stats.instanceCount        = uint32_t(instances.size());
    m_stats.modelTriangles       = 0;
    m_stats.modelClusters        = 0;
    m_stats.modelClustersAllLods = 0;
    m_stats.modelGroups          = 0;
    m_stats.sceneTriangles       = 0;
    m_stats.sceneClusters        = 0;
    m_stats.sceneClustersAllLods = 0;
    m_stats.sceneGroups          = 0;
    for (size_t gi = 0; gi < numGeom; ++gi)
    {
        const GeometryView& g = geometries[gi];
        m_stats.modelTriangles       += g.hiTriangleCount;
        m_stats.modelClusters        += g.hiClustersCount;
        m_stats.modelClustersAllLods += g.totalClustersCount;
        m_stats.modelGroups          += g.groupInfos.size();
        m_stats.sceneTriangles       += uint64_t(g.hiTriangleCount)    * instanceCount[gi];
        m_stats.sceneClusters        += uint64_t(g.hiClustersCount)    * instanceCount[gi];
        m_stats.sceneClustersAllLods += uint64_t(g.totalClustersCount) * instanceCount[gi];
        m_stats.sceneGroups          += uint64_t(g.groupInfos.size())  * instanceCount[gi];
    }

    // Load-cost split of the per-geometry loop (metadata upload vs low-detail
    // blob), logged with the init breakdown.
    double msMeta = 0.0, msLowDetail = 0.0;
    auto msBetween = [](std::chrono::steady_clock::time_point a,
                        std::chrono::steady_clock::time_point b)
    { return std::chrono::duration<double, std::milli>(b - a).count(); };

    // Lay out every geometry's low-detail group blob in one shared buffer.
    std::vector<uint64_t> lowDetailOffsets(numGeom);
    uint64_t              lowDetailBytes = 0;
    for (size_t geometryIndex = 0; geometryIndex < numGeom; geometryIndex++)
    {
        const GeometryView& sceneGeometry = geometries[geometryIndex];
        assert(sceneGeometry.lodLevelsCount != 0);
        const shaderio::LodLevel& lastLodLevel = sceneGeometry.lodLevels[sceneGeometry.lodLevelsCount - 1];
        lowDetailOffsets[geometryIndex] = lowDetailBytes;
        lowDetailBytes += rtxmg::align_up(
            uint64_t(sceneGeometry.groupInfos[lastLodLevel.groupOffset].GetDeviceSize()), uint64_t(16));
    }
    if (lowDetailBytes > UINT32_MAX)
        donut::log::fatal("ClusterLodStreaming: low-detail groups need %llu bytes; GroupAddress "
                          "offsets are 32-bit.", (unsigned long long)lowDetailBytes);
    m_lowDetailGroupsData = {};
    m_lowDetailGroupsData.Create(size_t(lowDetailBytes), "ClusterLodStreaming::lowDetailGroupsData", device);
    // Hit shaders reconstruct triangle payload from the scene-global resident
    // cluster-address table. The persistent low-detail blobs are outside the
    // streaming storage allocator, so they need their own bindless raw SRV.
    m_lowDetailGroupsDataSRVHandle = descriptorTable->CreateDescriptorHandle(
        nvrhi::BindingSetItem::RawBuffer_SRV(0, m_lowDetailGroupsData.GetBuffer()));
    const uint32_t lowDetailGroupsDataSRV = uint32_t(m_lowDetailGroupsDataSRVHandle.GetIndexInHeap());
    const nvrhi::GpuVirtualAddress lowDetailGroupsDataVA = m_lowDetailGroupsData.GetGpuVirtualAddress();

    uint32_t instancesOffset = 0;
    for (size_t geometryIndex = 0; geometryIndex < numGeom; geometryIndex++)
    {
        const GeometryView& sceneGeometry      = geometries[geometryIndex];
        shaderio::Geometry& shaderGeometry     = m_shaderGeometries[geometryIndex];
        PersistentGeometry& persistentGeometry = m_persistentGeometries[geometryIndex];

        const auto tMeta0 = std::chrono::steady_clock::now();

        // Shared per-geom setup — adopts the scene's pre-built LoD-tree buffers and
        // bindless SRVs, and fills the base fields of shaderGeometry.  The
        // streaming-only extras are filled below.
        UploadGeometryMetadata(geometryIndex, sceneGeometry,
                               persistentGeometry, shaderGeometry,
                               instancesOffset, instanceCount[geometryIndex]);
        instancesOffset += instanceCount[geometryIndex];

        const auto tMeta1 = std::chrono::steady_clock::now();
        msMeta += msBetween(tMeta0, tMeta1);

        // Cached scalars used by reset paths / StageResidencyUpdate.  The importer drops
        // geometries with no LOD hierarchy, so numLodLevels is never 0 here — the
        // `- 1` below would underflow if one slipped through.
        const uint32_t numLodLevels = uint32_t(sceneGeometry.lodLevelsCount);
        assert(numLodLevels != 0 && numLodLevels <= sceneGeometry.lodLevels.size());
        persistentGeometry.lodLevelsCount = numLodLevels;
        for (uint32_t i = 0; i < numLodLevels; i++)
        {
            persistentGeometry.lodGroupsCount[i] = sceneGeometry.lodLevels[i].groupCount;
        }

        // ---- Low-detail group blob ------------------------------------------
        const shaderio::LodLevel& lastLodLevel = sceneGeometry.lodLevels[numLodLevels - 1];
        persistentGeometry.lastLodGroupOffset = lastLodLevel.groupOffset;

        const GroupInfo& groupInfo = sceneGeometry.groupInfos[lastLodLevel.groupOffset];
        GroupView        groupView(sceneGeometry.groupData, groupInfo);
        // A multi-cluster root group is fine — everything below is sized by
        // clusterCount.  A multi-GROUP root is not (warned about in
        // ClusterLodResourcesBase::UploadGeometryMetadata).
        assert(groupInfo.clusterCount >= 1);

        const uint32_t lastClustersCount = uint32_t(groupInfo.clusterCount);
        const uint64_t lastGroupSize     = groupInfo.GetDeviceSize();
        const uint64_t lowDetailOffset   = lowDetailOffsets[geometryIndex];

        const shaderio::GroupAddress lowDetailGroupAddress = {
            lowDetailGroupsDataSRV,
            uint32_t(lowDetailOffset),
        };

        GeometryGroup geometryGroup{uint32_t(geometryIndex), lastLodLevel.groupOffset};
        StreamingResident::Group* rgroup =
            m_resident.AddGroup(geometryGroup, lastClustersCount, groupInfo.triangleCount);
        if (!rgroup)
            donut::log::fatal("ClusterLodStreaming: no residency slot for geometry %u's low-detail "
                       "group (%u clusters). Raise --maxresidentgroups.",
                       uint32_t(geometryIndex), lastClustersCount);
        rgroup->deviceAddress = lowDetailGroupsDataVA + lowDetailOffset;
        rgroup->groupAddress  = lowDetailGroupAddress;
        rgroup->lodLevel      = groupInfo.lodLevel;

        persistentGeometry.lodLoadedGroupsCount[groupInfo.lodLevel] = 1;

        // Build the per-group runtime data in a host-side staging buffer, then
        // writeBuffer into the device buffer.
        std::vector<uint8_t> loGroupData(static_cast<size_t>(lastGroupSize));
        FillStreamingGroupData(groupInfo, groupView, m_decompressScratch,
                               rgroup->clusterResidentID,
                               rgroup->groupResidentID,
                               loGroupData.data(), loGroupData.size());
        commandList->writeBuffer(m_lowDetailGroupsData.GetBuffer(),
                                 loGroupData.data(), loGroupData.size(), lowDetailOffset);


        const shaderio::BBox& lowDetailBBox = groupView.clusterBboxes[0];
        const float lowDetailBBoxExtent =
            std::max(donut::math::length(lowDetailBBox.hi - lowDetailBBox.lo), 1e-6f);

        shaderGeometry.lowDetailClusterID = rgroup->clusterResidentID;
        shaderGeometry.lowDetailTriangles = uint16_t(groupInfo.triangleCount);
        shaderGeometry.lowDetailClusters  = lastClustersCount;  // render-stats
        shaderGeometry.bbox.longestEdge = lowDetailBBox.longestEdge / lowDetailBBoxExtent;

        msLowDetail += msBetween(tMeta1, std::chrono::steady_clock::now());
    }

    // Geometries share metadata buffers (one set per scene-load batch), so count
    // each buffer once at its real size.
    {
        std::unordered_set<nvrhi::IBuffer*> metadataBuffers;
        for (const PersistentGeometry& pg : m_persistentGeometries)
        {
            metadataBuffers.insert(pg.lodLevels.buffer);
            metadataBuffers.insert(pg.lodNodes.buffer);
            metadataBuffers.insert(pg.lodNodeBboxes.buffer);
            metadataBuffers.insert(pg.streamingGroupAddresses.buffer);
        }
        m_persistentGeometrySize = m_lowDetailGroupsData.GetVramBytes();
        for (nvrhi::IBuffer* buffer : metadataBuffers)
            m_persistentGeometrySize += rtxmg::BufferVramBytes(buffer);
    }

    donut::log::info("ClusterLodStreaming InitGeometries: metadata upload %.0f ms, "
                     "low-detail groups %.0f ms (%zu geometries)",
                     msMeta, msLowDetail, numGeom);

    // ---- Render instances buffer (matches preloaded.cpp's pattern) -----------
    {
        const size_t numInst = instances.size();
        m_renderInstances.resize(numInst);
        for (size_t i = 0; i < numInst; ++i)
        {
            const ClusterLodInstance& src = instances[i];
            shaderio::RenderInstance& dst = m_renderInstances[i];

            const affine3 xf = homogeneousToAffine(src.transform);
            affineToColumnMajor(xf, dst.worldMatrix.m_data);
            // Affine inverse for the BLAS-sharing object-space camera transform.
            affineToColumnMajor(inverse(xf), dst.worldMatrixI.m_data);
            dst.geometryID   = src.geometryID;
            // RenderInstance material state.  Same logic as
            // ClusterLodPreloaded::Init — see comments there.
            const GeometryView& geo = m_geometries[src.geometryID];
            // --nomat: scene only pushed a single default-gray material at
            // m_clusterLodMaterialBaseID — ignore the per-instance offset.
            dst.materialID                = m_enableMaterials
                                                ? (m_clusterLodMaterialBaseID + src.materialID)
                                                : m_clusterLodMaterialBaseID;
            if (m_enableMaterials)
            {
                dst.multiMaterial             = (geo.localMaterialIDs.size() > 1) ? 1u : 0u;
                dst.lowDetailClusterStateBits = geo.lowDetailClusterStateBits;
                bool anyAlpha = false;
                bool allAlpha = !geo.localMaterialStateBits.empty();
                for (uint8_t mb : geo.localMaterialStateBits)
                {
                    const bool a = (mb & shaderio::ClusterState::AlphaMasked) != 0u;
                    anyAlpha = anyAlpha || a;
                    allAlpha = allAlpha && a;
                }
                dst.opaqueStatus = !anyAlpha ? uint32_t(shaderio::OpaqueStatus::Opaque)
                                              : (allAlpha ? uint32_t(shaderio::OpaqueStatus::AlphaMasked)
                                                          : uint32_t(shaderio::OpaqueStatus::Mixed));
            }
            else
            {
                // --nomat: opaque, single-material, no per-cluster alpha hints.
                dst.multiMaterial             = 0u;
                dst.lowDetailClusterStateBits = 0u;
                dst.opaqueStatus              = uint32_t(shaderio::OpaqueStatus::Opaque);
            }
            dst._stateReserved = 0;
            dst._pad2          = 0;
        }
        if (numInst > 0)
        {
            m_renderInstancesBuffer.Create(uint32_t(numInst), "ClusterLodRenderInstances", device);
            commandList->writeBuffer(m_renderInstancesBuffer.GetBuffer(),
                                     m_renderInstances.data(),
                                     numInst * sizeof(shaderio::RenderInstance));
        }
    }

    // The group-address table is seeded by ResetGeometryStreamingState, reached
    // via the renderer's UpdateClasRequired(true) right after init(); seeding it
    // here too would be a second full-table write into the same command list
    // (a WRITE_AFTER_WRITE hazard under VVL sync validation).

    // ---- Upload shader-geometry table ---------------------------------------
    m_shaderGeometriesBuffer.Create(uint32_t(numGeom), "ClusterLodShaderGeometries", device);
    m_operationsSize += m_shaderGeometriesBuffer.GetVramBytes();
    commandList->writeBuffer(m_shaderGeometriesBuffer.GetBuffer(),
                             m_shaderGeometries.data(),
                             numGeom * sizeof(shaderio::Geometry));

    // Freezes everything added so far as the persistent prefix and uploads its
    // StreamingGroup + ClusterAddress entries.
    m_resident.UploadInitialState(commandList, m_shaderData.resident);
}

void ClusterLodStreaming::StageResidencyUpdate(nvrhi::ICommandList* commandList,
                                               const FrameSettings& settings)
{
    // This function sets up all relevant streaming tasks for the frame and
    // configures the host-side sub-manager snapshots that all streaming
    // kernels operate on.  Drain order matters:
    //   - handleCompletedUpdate   : recycle memory from old unloads.
    //   - handleCompletedStorage  : pick up the matching update task.
    //   - HandleCompletedRequest  : process new loads/unloads.
    //   - push a new request slot for this frame.
    //
    // Called by the renderer.

    // For each task queue we must ensure that we have one new task index
    // available to acquire for any potential new work in this frame.
    const bool ensureAcquisition = true;

    // pop all completed old updates to recycle as much memory as we can
    while (m_updatesTaskQueue.CanPop(ensureAcquisition))
    {
        // handleCompletedUpdate — the update has hit GPU timeline, free its
        // backing storage now that nothing on the GPU references it.
        const uint32_t popUpdateIndex = m_updatesTaskQueue.Pop();

        const StreamingUpdates::TaskInfo& update = m_updates.GetCompletedTask(popUpdateIndex);
        for (uint32_t g = 0; g < update.unloadCount; g++)
        {
            // BufferSubAllocation handle carries the size + allocator
            // metadata; free() recovers everything from the handle.
            m_storage.Free(update.unloadHandles[g]);
        }

        m_updatesTaskQueue.ReleaseTaskIndex(popUpdateIndex);
    }

    // Retire a completed storage transfer, freeing its slice of the ring.
    if (m_storageTaskQueue.CanPop(ensureAcquisition))
    {
        m_storageTaskQueue.ReleaseTaskIndex(m_storageTaskQueue.Pop());
    }

    // pop and process one completed request
    uint32_t pushUpdateIndex = kInvalidTaskIndex;
    if (m_requestsTaskQueue.CanPop(ensureAcquisition))
    {
        uint32_t popRequestIndex = m_requestsTaskQueue.Pop();

        // skip ahead to the most recent completed request (avoids stale
        // requests piling up).
        while (m_requestsTaskQueue.CanPop(false))
        {
            m_requestsTaskQueue.ReleaseTaskIndex(popRequestIndex);
            popRequestIndex = m_requestsTaskQueue.Pop();
        }

        pushUpdateIndex = HandleCompletedRequest(commandList, settings, popRequestIndex);
    }

    // test if there is an update to be done this frame
    if (pushUpdateIndex != kInvalidTaskIndex)
    {
        // Given we know all data was uploaded, we can run the updates to the
        // scene in this frame, which ultimately fulfills a past request on the
        // device. Both resident and update operations are a synchronized pair,
        // hence a single task index is sufficient.
        m_resident.ApplyTask(m_shaderData.resident, pushUpdateIndex, m_frameIndex);
        m_updates.ApplyTask(m_shaderData.update,    pushUpdateIndex, m_frameIndex);
        LogDebugUpdateApply(pushUpdateIndex);

        m_pendingSubmittedTasks.push_back({
            PendingSubmittedTask::TaskQueue::Updates,
            pushUpdateIndex,
        });
    }
    else
    {
        // No patch work this frame — zero the per-task update fields so the
        // streaming shaders see a clean "no work" header.
        m_shaderData.update.patchGroupsCount         = 0;
        m_shaderData.update.patchUnloadGroupsCount   = 0;
        m_shaderData.update.patchCachedBlasCount     = 0;
        m_shaderData.update.patchCachedClustersCount = 0;
        m_shaderData.update.loadActiveGroupsOffset   = 0;
        m_shaderData.update.loadActiveClustersOffset = 0;
        m_shaderData.update.newClasCount             = 0;
        m_shaderData.update.moveClasCounter          = 0;
        m_shaderData.update.taskIndex                = kInvalidTaskIndex;
        m_shaderData.update.frameIndex               = m_frameIndex;
    }

    // Compaction allocator: the move cursor starts at 0 so
    // stream_compact_defrag_old packs old CLAS from the pool base and its
    // moveClasCounter atomic accumulates from zero.  On a no-unloads frame
    // stream_dispatch_setup's CompactionOldNoUnloads branch re-seeds moveClasSize from
    // the GPU-persistent cursor instead.  (The persistent allocator keeps
    // moveClasCounter = newClusterCount from StreamingUpdates::ApplyTask.)
    if (m_requiresClas && !m_config.usePersistentClasAllocator)
    {
        m_shaderData.update.moveClasSize    = 0;
        m_shaderData.update.moveClasCounter = 0;
    }

    // Push the new request slot for this frame's GPU work — populates the
    // aggregate m_shaderData.request header that the request-collection
    // shaders write counters into.
    {
        const uint32_t pushRequestIndex = m_requestsTaskQueue.AcquireTaskIndex();
        assert(pushRequestIndex != kInvalidTaskIndex);  // guaranteed by design
        m_requests.ApplyTask(m_shaderData.request, pushRequestIndex, m_frameIndex);
    }

    if (m_requiresClas && m_config.usePersistentClasAllocator)
    {
        // clears the per-class freeSizeRanges to zero + re-uploads the
        // allocator header (carries the per-frame freeGapsCounter reset).
        m_clasAllocator.ClearFreeSizeRanges(commandList);
    }

    // Cross-sub-manager scalars set on the aggregate.
    m_shaderData.frameIndex               = m_frameIndex;
    m_shaderData.ageThreshold             = settings.ageThreshold;
    m_shaderData.useBlasCaching           = settings.useBlasCaching ? 1u : 0u;
    m_shaderData.clasPositionTruncateBits = m_clasTriangleInput.minPositionTruncateBitCount;
    m_shaderData.materialsEnabled         = m_enableMaterials ? 1u : 0u;

    // One wholesale write of this frame's configuration.  It implicitly carries
    // the freeGapsCounter / dispatchFreeGapsInsert reset: the host snapshot
    // holds 0 from init, so the upload re-zeros the device's stale counter.
    commandList->writeBuffer(m_shaderBuffer, &m_shaderData, sizeof(m_shaderData));

    // Rebuild the unified BindingSet now that update.taskIndex is finalized —
    // the patches sub-range offset depends on it.
    UpdateBindings(commandList);
    LogDebugResidentActive();
}

void ClusterLodStreaming::CheckRequestErrors(const StreamingRequests::TaskInfo& request,
                                            uint64_t                           requestFrame,
                                            uint32_t                           popRequestIndex) const
{
    // Overflowing maxLoads is normal while a large scene streams in, so this is
    // --debug-clusterlod only and rate-limited to once a second.  A sustained
    // non-decreasing dropped count at steady state means the working set exceeds
    // budget / aging isn't reclaiming.
    if (DebugClusterLodLoggingEnabled()
        && request.shaderData->loadCounter > request.shaderData->maxLoads
        && (m_frameIndex % 60 == 0))
    {
        donut::log::info("ClusterLodStreaming request-overflow frame=%llu rawLoad=%u maxLoad=%u dropped=%u",
                            (unsigned long long)requestFrame,
                            request.shaderData->loadCounter,
                            request.shaderData->maxLoads,
                            request.shaderData->loadCounter - request.shaderData->maxLoads);
    }

    {
        const char* errorCause = nullptr;
        if (request.shaderData->errorUpdate != 0)
            errorCause = "update";
        else if (request.shaderData->errorAgeFilter != 0)
            errorCause = "age filter";
        else if (request.shaderData->errorClasOversized != 0)
            errorCause = "clas oversized";
        else if (request.shaderData->errorClasNotFound != 0)
            errorCause = "clas not found";
        else if (request.shaderData->errorClasAlloc != 0)
            errorCause = "clas alloc";
        else if (request.shaderData->errorClasDealloc != 0)
            errorCause = "clas dealloc";
        else if (request.shaderData->errorClasList != 0)
            errorCause = "clas list";
        else if (request.shaderData->errorClasUsedVsAlloc != 0)
            errorCause = "clas used vs. alloc";

        if (errorCause)
        {
            donut::log::fatal("streaming: fatal error - %s frame=%llu rawFrame=%llu task=%u shaderTask=%u rawLoad=%u rawUnload=%u maxLoad=%u maxUnload=%u err(update=%d age=%d oversized=%d notFound=%d list/maxBinCount=%d alloc/maxBinOffset=%d dealloc/freeGaps=%d usedVsAlloc/allocSize=%d) maxSizedLeft=%u usedBytes=%llu wastedBytes=%llu",
                              errorCause,
                              (unsigned long long)requestFrame,
                              (unsigned long long)requestFrame,
                              popRequestIndex,
                              request.shaderData->taskIndex,
                              request.shaderData->loadCounter,
                              request.shaderData->unloadCounter,
                              request.shaderData->maxLoads,
                              request.shaderData->maxUnloads,
                              request.shaderData->errorUpdate,
                              request.shaderData->errorAgeFilter,
                              request.shaderData->errorClasOversized,
                              request.shaderData->errorClasNotFound,
                              request.shaderData->errorClasList,
                              request.shaderData->errorClasAlloc,
                              request.shaderData->errorClasDealloc,
                              request.shaderData->errorClasUsedVsAlloc,
                              request.shaderData->clasAllocatedMaxSizedLeft,
                              (unsigned long long)request.shaderData->clasAllocatedUsedSize,
                              (unsigned long long)request.shaderData->clasAllocatedWastedSize);
        }
    }
}

void ClusterLodStreaming::UpdateClasBudgetStats(const StreamingRequests::TaskInfo& request,
                                                uint64_t                           requestFrame)
{
    if (m_requiresClas)
    {
        if (m_config.usePersistentClasAllocator)
        {
            m_stats.usedClasBytes   = request.shaderData->clasAllocatedUsedSize;
            m_stats.wastedClasBytes = request.shaderData->clasAllocatedWastedSize;
            m_stats.maxSizedLeft    = request.shaderData->clasAllocatedMaxSizedLeft;
        }
        else
        {
            const uint64_t clasPoolBytes = uint64_t(m_config.maxClasMegaBytes) * 1024 * 1024;
            const uint64_t usedClas      = request.shaderData->clasCompactionUsedSize;
            m_stats.usedClasBytes   = usedClas;
            m_stats.wastedClasBytes = 0;
            // Clamp: once the pool is full, the unsigned (pool - used) must not
            // wrap to a huge bogus budget (which would green-light over-admission).
            m_stats.maxSizedLeft = usedClas >= clasPoolBytes ? 0u
                : uint32_t((clasPoolBytes - usedClas)
                           / (m_clasSingleMaxSize * m_bakerConfig.clusterGroupSize));
            if (DebugClusterLodLoggingEnabled())
            {
                // clasCompactionUsedSize is the moveClasSize byte cursor after
                // last frame's defrag — the live resident-set CLAS footprint.
                donut::log::info("ClusterLodStreaming clas-compact frame=%llu usedClasBytes=%llu poolBytes=%llu pct=%.2f%% maxSizedLeft=%u",
                                 (unsigned long long)requestFrame,
                                 (unsigned long long)m_stats.usedClasBytes,
                                 (unsigned long long)clasPoolBytes,
                                 clasPoolBytes ? (100.0 * double(m_stats.usedClasBytes) / double(clasPoolBytes)) : 0.0,
                                 m_stats.maxSizedLeft);
            }
        }
    }
}

void ClusterLodStreaming::StageUnloads(const StreamingRequests::TaskInfo& request,
                                       StreamingUpdates::TaskInfo&        updateTask,
                                       uint32_t                           unloadCount,
                                       bool                               useBlasCaching)
{
    // Unloads first so we can recycle resident objects.
    for (uint32_t g = 0; g < unloadCount; g++)
    {
        GeometryGroup geometryGroup = request.unloadGeometryGroups[g];

        assert(geometryGroup.geometryID < m_geometries.size());
        assert(geometryGroup.groupID < m_geometries[geometryGroup.geometryID].groupInfos.size());

        const StreamingResident::Group* group = m_resident.FindGroup(geometryGroup);
        if (!group)
        {
            // already removed by a prior request (GPU-timeline lag).
            continue;
        }

        const uint32_t unloadIndex                       = updateTask.unloadCount++;
        shaderio::StreamingPatch& patch                  = updateTask.unloadPatches[unloadIndex];
        patch                                            = {};
        patch.geometryID                                 = geometryGroup.geometryID;
        patch.groupID                                    = geometryGroup.groupID;
        // The patch carries the blob address and resident IDs so
        // stream_allocator_free_groups never has to read the per-geometry
        // groupAddresses[] table or the group blob.  Both are written through
        // bindless descriptors that nvrhi cannot barrier, so a same-batch
        // stream_update_scene would race the read and orphan the CLAS
        // allocation.
        patch.groupAddress                               = group->groupAddress;
        patch.clasBuildOffset                            = group->clusterResidentID;
        patch.unloadGroupResidentID                      = group->groupResidentID;

        // Defer the actual storage free until the update task completes — only
        // then does the GPU's scene graph stop referencing the data.
        updateTask.unloadHandles[unloadIndex] = group->storageHandle;

        assert(m_persistentGeometries[geometryGroup.geometryID].lodLoadedGroupsCount[group->lodLevel] > 0);
        m_persistentGeometries[geometryGroup.geometryID].lodLoadedGroupsCount[group->lodLevel]--;

        m_resident.RemoveGroup(group->groupResidentID);

        // append to geometry patch list if necessary (BLAS cache invalidation)
        if (useBlasCaching && m_persistentGeometries[geometryGroup.geometryID].cachedBlasUpdateFrame != m_frameIndex)
        {
            m_persistentGeometries[geometryGroup.geometryID].cachedBlasUpdateFrame = m_frameIndex;
            const uint32_t geometryPatchIndex                = updateTask.geometryCachedCount++;
            shaderio::StreamingGeometryPatch& geometryPatch  = updateTask.geometryPatches[geometryPatchIndex];
            geometryPatch                                    = {};
            geometryPatch.geometryID                         = geometryGroup.geometryID;
        }
    }
}

ClusterLodStreaming::LoadStageResult
ClusterLodStreaming::StageLoads(const StreamingRequests::TaskInfo& request,
                               StreamingStorage::TaskInfo&        storageTask,
                               StreamingUpdates::TaskInfo&        updateTask,
                               uint32_t                           loadCount,
                               uint64_t                           requestFrame,
                               bool                               useBlasCaching)
{
    // CLAS budget tracking: two systems (compaction-based and persistent-allocator).
    uint64_t clasMovedUsedSize         = request.shaderData->clasCompactionUsedSize;
    const uint64_t clasMovedReservedSize = uint64_t(m_config.maxClasMegaBytes) * 1024 * 1024;
    uint32_t clasAllocatedMaxSizedLeft = request.shaderData->clasAllocatedMaxSizedLeft;

    // Account for in-flight CLAS operations from later frames that have not
    // yet hit GPU timeline — they indirectly reduce the budget left.
    const StreamingUpdates::NewInfo futureNew = m_updates.GetFutureNew(requestFrame);
    clasMovedUsedSize += m_clasSingleMaxSize * futureNew.clusters;
    clasAllocatedMaxSizedLeft -= std::min(clasAllocatedMaxSizedLeft, futureNew.groups);

    uint32_t clasBuildOffset = 0;
    uint64_t clasBuildSize   = 0;

    updateTask.loadActiveGroupsOffset   = m_resident.GetLoadActiveGroupsOffset();
    updateTask.loadActiveClustersOffset = m_resident.GetLoadActiveClustersOffset();

    uint64_t transferBytes = 0;

    namespace cluster = nvrhi::rt::cluster;

    m_stats.couldNotAllocateClas  = 0;
    m_stats.couldNotTransfer      = 0;
    m_stats.couldNotAllocateGroup = 0;
    m_stats.couldNotStore         = 0;
    m_stats.uncompletedLoadCount  = 0;

    uint32_t skippedResidentCount = 0;

    for (uint32_t g = 0; g < loadCount; g++)
    {
        GeometryGroup geometryGroup = request.loadGeometryGroups[g];

        assert(geometryGroup.geometryID < m_geometries.size());
        assert(geometryGroup.groupID < m_geometries[geometryGroup.geometryID].groupInfos.size());

        if (m_resident.FindGroup(geometryGroup))
        {
            // duplicate request — patch may still be in flight; skip silently.
            skippedResidentCount++;
            continue;
        }

        const GeometryView& sceneGeometry = m_geometries[geometryGroup.geometryID];
        const GroupInfo&    groupInfo     = sceneGeometry.groupInfos[geometryGroup.groupID];
        const uint32_t      clusterCount  = uint32_t(groupInfo.clusterCount);
        const GroupView     groupView(sceneGeometry.groupData, groupInfo);

        // Channel stripping shrinks the pool allocation to the compacted blob;
        // stripped positions still ride the upload ring for the CLAS build, so
        // the transfer budget has to cover both.
        const bool stripPositions = m_config.stripResidentPositions;
        const bool stripNormals   = m_config.stripResidentNormals;
        uint64_t groupDeviceSize   = groupInfo.GetDeviceSize();
        uint64_t groupPositionSize = 0;
        if (stripPositions || stripNormals)
            ComputeStrippedGroupSizes(groupInfo, groupView.clusters.data(),
                                      stripPositions, stripNormals,
                                      groupDeviceSize, groupPositionSize);

        uint64_t            groupClasSize = 0;
        bool                canAllocateClas = true;

        if (m_requiresClas)
        {
            groupClasSize = m_clasSingleMaxSize * clusterCount;
            assert((clasBuildSize + groupClasSize) <= m_clasScratchNewClasSize);

            if (m_config.usePersistentClasAllocator)
            {
                canAllocateClas = clasAllocatedMaxSizedLeft > 0;
            }
            else
            {
                canAllocateClas = (clasMovedUsedSize + (clasBuildSize + groupClasSize)) <= clasMovedReservedSize;
            }
        }

        uint64_t                    deviceAddress = 0;
        rtxmg::BufferSubAllocation  storageHandle;

        const bool canTransfer      = m_storage.CanTransfer(storageTask, groupDeviceSize + groupPositionSize);
        const bool canStore         = m_storage.Allocate(storageHandle, geometryGroup, groupDeviceSize, deviceAddress);
        const bool canAllocateGroup = m_resident.CanAllocateGroup(clusterCount);

        if (!canTransfer || !canStore || !canAllocateGroup || !canAllocateClas)
        {
            m_stats.couldNotAllocateClas  += (!canAllocateClas);
            m_stats.couldNotTransfer      += (!canTransfer);
            m_stats.couldNotAllocateGroup += (!canAllocateGroup);
            m_stats.couldNotStore         += (!canStore);

            if (canStore)
            {
                m_storage.Free(storageHandle);
            }

            if (clusterCount < 8)
            {
                m_stats.uncompletedLoadCount += loadCount - g;
                break;  // heuristic: small groups failing => fully break.
            }
            m_stats.uncompletedLoadCount++;
            continue;
        }

        StreamingResident::Group* residentGroup = m_resident.AddGroup(geometryGroup, clusterCount, groupInfo.triangleCount);
        if (!residentGroup)
        {
            // CanAllocateGroup above says this cannot happen; treat it as the
            // same refusal rather than dereferencing null if it ever does.
            m_storage.Free(storageHandle);
            m_stats.couldNotAllocateGroup++;
            m_stats.uncompletedLoadCount++;
            continue;
        }
        residentGroup->storageHandle            = storageHandle;
        residentGroup->deviceAddress            = deviceAddress;
        residentGroup->lodLevel                 = groupInfo.lodLevel;

        // AppendTransfer resolves the destination buffer + offset via
        // BufferSubAllocator::subRange under the hood.
        void* groupData = m_storage.AppendTransfer(storageTask, storageHandle, size_t(groupDeviceSize));

        assert(deviceAddress % 16 == 0);

        // Diverted positions: a no-copy ring range the CLAS build consumes by
        // GPU VA (published via patch.positionsGpuAddress below).
        void*    positionsPtr = nullptr;
        uint64_t positionsVA  = 0;
        if (groupPositionSize)
            positionsPtr = m_storage.AppendHostRead(storageTask, size_t(groupPositionSize), positionsVA);

        // groupDeviceSize already accounts for the stripped/uncompressed size.
        FillStreamingGroupData(groupInfo, groupView, m_decompressScratch,
                               residentGroup->clusterResidentID,
                               residentGroup->groupResidentID,
                               groupData, size_t(groupDeviceSize),
                               positionsPtr, size_t(groupPositionSize),
                               stripPositions, stripNormals);

        m_persistentGeometries[geometryGroup.geometryID].lodLoadedGroupsCount[groupInfo.lodLevel]++;

        if (useBlasCaching && m_persistentGeometries[geometryGroup.geometryID].cachedBlasUpdateFrame != m_frameIndex)
        {
            m_persistentGeometries[geometryGroup.geometryID].cachedBlasUpdateFrame = m_frameIndex;
            const uint32_t geometryPatchIndex                = updateTask.geometryCachedCount++;
            shaderio::StreamingGeometryPatch& geometryPatch  = updateTask.geometryPatches[geometryPatchIndex];
            geometryPatch                                    = {};
            geometryPatch.geometryID                         = geometryGroup.geometryID;
        }

        // setup load patch: GroupAddress + absolute GVA for CLAS-build payload reads
        const rtxmg::BufferRange storageRange = m_storage.GetAllocator().subRange(storageHandle);
        shaderio::StreamingPatch& patch = updateTask.loadPatches[updateTask.loadCount++];
        patch                          = {};
        patch.geometryID               = geometryGroup.geometryID;
        patch.groupID                  = geometryGroup.groupID;
        patch.groupAddress             = shaderio::GroupAddress{ storageRange.bindlessSlot,
                                                                  uint32_t(storageRange.offset) };
        residentGroup->groupAddress    = patch.groupAddress;
        patch.clasBuildOffset          = clasBuildOffset;
        // Absolute GVA of the group blob.  stream_update_scene derives every
        // CLAS-build vertex/index GVA from it as `deviceAddress +
        // sizeof(Group) + c*sizeof(Cluster) + cl.{vertices,indices}` — except
        // stripped positions, which come off positionsGpuAddress plus a running
        // vertex-count prefix.
        patch.groupGpuAddress          = deviceAddress;
        patch.positionsGpuAddress      = positionsVA;

        clasBuildOffset += clusterCount;
        clasBuildSize   += groupClasSize;
        // Saturating: the > 0 guard above only runs on the persistent
        // allocator, so the other paths would wrap this to 0xFFFFFFFF.
        if (clasAllocatedMaxSizedLeft > 0)
            clasAllocatedMaxSizedLeft--;

        transferBytes += groupInfo.sizeBytes;
    }

    updateTask.newClusterCount = clasBuildOffset;

    LoadStageResult result;
    result.skippedResidentCount = skippedResidentCount;
    result.clasBudgetForLoads   = clasAllocatedMaxSizedLeft;
    result.futureGroups         = futureNew.groups;
    result.transferBytes        = transferBytes;
    return result;
}

uint32_t ClusterLodStreaming::HandleCompletedRequest(nvrhi::ICommandList* commandList,
                                                     const FrameSettings& settings,
                                                     uint32_t             popRequestIndex)
{
    // This function handles the requests from the device to upload new geometry
    // groups, or unload some that haven't been used in a while.  The readback
    // of the data is guaranteed to have completed at this point.  Uploading
    // tries to handle as much as we have memory for, and every completed upload
    // needs a matching update task.
    //
    // Only called by StageResidencyUpdate.

    const StreamingRequests::TaskInfo& request = m_requests.GetCompletedTask(popRequestIndex);

    // during recording of requests the counters may exceed the limits
    // however the data is always ensured to be within.
    const uint32_t loadCount   = std::min(request.shaderData->maxLoads,   request.shaderData->loadCounter);
    const uint32_t unloadCount = std::min(request.shaderData->maxUnloads, request.shaderData->unloadCounter);

    const uint64_t requestFrame = request.shaderData->frameIndex;
    LogDebugRequestReadback(requestFrame, popRequestIndex, loadCount, unloadCount, request);

    CheckRequestErrors(request, requestFrame, popRequestIndex);
    UpdateClasBudgetStats(request, requestFrame);

    if (loadCount == 0 && unloadCount == 0)
    {
        // no work to do
        m_requestsTaskQueue.ReleaseTaskIndex(popRequestIndex);
        return kInvalidTaskIndex;
    }

    const uint32_t pushStorageIndex = m_storageTaskQueue.AcquireTaskIndex();
    const uint32_t pushUpdateIndex  = m_updatesTaskQueue.AcquireTaskIndex();

    // early out if we are not able to acquire both tasks to serve the request
    if (pushStorageIndex == kInvalidTaskIndex || pushUpdateIndex == kInvalidTaskIndex)
    {
    // give back acquisitions we don't make use of
        if (pushStorageIndex != kInvalidTaskIndex)
        {
            m_storageTaskQueue.ReleaseTaskIndex(pushStorageIndex);
        }
        if (pushUpdateIndex  != kInvalidTaskIndex) 
        {
            m_updatesTaskQueue.ReleaseTaskIndex(pushUpdateIndex);
        }
        m_requestsTaskQueue.ReleaseTaskIndex(popRequestIndex);
        return kInvalidTaskIndex;
    }

    StreamingStorage::TaskInfo& storageTask = m_storage.GetNewTask(pushStorageIndex);
    StreamingUpdates::TaskInfo& updateTask  = m_updates.GetNewTask(pushUpdateIndex);

    const bool useBlasCaching = m_requiresClas && m_config.allowBlasCaching && settings.useBlasCaching;

    StageUnloads(request, updateTask, unloadCount, useBlasCaching);
    const LoadStageResult loads =
        StageLoads(request, storageTask, updateTask, loadCount, requestFrame, useBlasCaching);
    uint64_t transferBytes = loads.transferBytes;

    if (DebugClusterLodLoggingEnabled())
    {
        donut::log::info("ClusterLodStreaming request-stage frame=%llu task=%u requestedLoad=%u stagedLoad=%u skippedResident=%u unloadPatches=%u newClusters=%u transferBytes=%llu clasBudgetReadback=%u futureGroups=%u clasBudgetForLoads=%u failClas=%u failTransfer=%u failGroup=%u failStore=%u uncompleted=%u",
                         (unsigned long long)requestFrame,
                         pushUpdateIndex,
                         loadCount,
                         updateTask.loadCount,
                         loads.skippedResidentCount,
                         updateTask.unloadCount,
                         updateTask.newClusterCount,
                         (unsigned long long)transferBytes,
                         request.shaderData->clasAllocatedMaxSizedLeft,
                         loads.futureGroups,
                         loads.clasBudgetForLoads,
                         m_stats.couldNotAllocateClas,
                         m_stats.couldNotTransfer,
                         m_stats.couldNotAllocateGroup,
                         m_stats.couldNotStore,
                         m_stats.uncompletedLoadCount);
    }

    // Resident slot pool exhausted: the residency arrays are full, so nothing
    // more streams in even with the geometry/CLAS byte pools under budget.  This
    // is the "scene doesn't fully come in" wedge, so it is warned unconditionally
    // (rate-limited to once a second).
    if (m_stats.couldNotAllocateGroup > 0 && (m_frameIndex % 60 == 0))
    {
        // Residency counts live on StreamingResident, not m_stats, which only
        // learns them when GetStats composes the two.
        const StreamingResident::ResidencyStats resident = m_resident.GetStats();
        donut::log::warning("ClusterLodStreaming: max resident groups EXHAUSTED "
                            "(resident %u/%u groups, %u/%u clusters); %u group load(s) rejected "
                            "this frame - geometry cannot fully stream in. Raise --maxresidentgroups.",
                            resident.residentGroups, m_stats.maxGroups,
                            resident.residentClusters, m_stats.maxClusters,
                            m_stats.couldNotAllocateGroup);
    }

    if (updateTask.loadCount == 0 && updateTask.unloadCount == 0)
    {
        // we ended up doing no work
        m_requestsTaskQueue.ReleaseTaskIndex(popRequestIndex);
        m_updatesTaskQueue.ReleaseTaskIndex(pushUpdateIndex);
        m_storageTaskQueue.ReleaseTaskIndex(pushStorageIndex);
        return kInvalidTaskIndex;
    }

    if (useBlasCaching)
    {
        HandleBlasCaching(updateTask, settings);
    }

    uint32_t transferCount = 0;
    transferBytes += m_updates.UploadTaskPatches(commandList, pushUpdateIndex);
    transferBytes += m_resident.UploadActiveGroupsDelta(commandList, pushUpdateIndex);
    transferCount += m_storage.UploadPendingTransfers(commandList);
    transferCount += 2;
    if (DebugClusterLodLoggingEnabled())
    {
        donut::log::info("ClusterLodStreaming upload-stage frame=%llu task=%u patchLoad=%u patchUnload=%u newClusters=%u transferBytes=%llu transferCount=%u",
                         (unsigned long long)requestFrame,
                         pushUpdateIndex,
                         updateTask.loadCount,
                         updateTask.unloadCount,
                         updateTask.newClusterCount,
                         (unsigned long long)transferBytes,
                         transferCount);
    }

    // Cumulative totals (monotonic) — the UI's per-second rate graph takes
    // per-frame deltas of these.
    m_stats.totalTransferBytes += transferBytes;
    if (updateTask.loadCount)
    {
        m_stats.transferBytes = transferBytes;
        m_stats.transferCount = transferCount;
        m_stats.loadCount     = updateTask.loadCount;
        m_stats.totalLoads   += updateTask.loadCount;
    }
    if (updateTask.unloadCount)
    {
        m_stats.unloadCount   = updateTask.unloadCount;
        m_stats.totalUnloads += updateTask.unloadCount;
    }

    m_requestsTaskQueue.ReleaseTaskIndex(popRequestIndex);

    m_pendingSubmittedTasks.push_back({
        PendingSubmittedTask::TaskQueue::Storage,
        pushStorageIndex,
    });

    return pushUpdateIndex;
}

void ClusterLodStreaming::HandleBlasCaching(StreamingUpdates::TaskInfo& updateTask,
                                            const FrameSettings&        settings)
{
    // geometryPatches[0..geometryCachedCount) arrives seeded with one entry per
    // geometry whose resident LoD set changed this frame.  Decide per geometry
    // whether to (re)build / invalidate / keep its cached BLAS, then compact the
    // surviving patches to the front of the array and publish per-frame totals.
    uint32_t writeIndex          = 0;
    uint32_t cachedBuildsTotal   = 0;
    uint32_t cachedClustersTotal = 0;

    for (uint32_t g = 0; g < updateTask.geometryCachedCount; g++)
    {
        shaderio::StreamingGeometryPatch sgpatch            = updateTask.geometryPatches[g];
        PersistentGeometry&              persistentGeometry = m_persistentGeometries[sgpatch.geometryID];
        const GeometryView&              geometryView       = m_geometries[sgpatch.geometryID];

        // Find the finest fully-resident tail LoD level whose cluster count fits
        // a cached BLAS.  Skip the final level — it always exists as the
        // pre-built low-detail BLAS, so caching it would be redundant.
        uint32_t cachedClustersCount  = 0;
        uint32_t cachedTrianglesCount = 0;  // render-stats: triangles of the cached level
        const uint32_t blasCacheMinLevel =
            persistentGeometry.lodLevelsCount
            - std::min(settings.blasCacheMinLevel, persistentGeometry.lodLevelsCount);

        for (uint32_t i = blasCacheMinLevel; i + 1u < persistentGeometry.lodLevelsCount; i++)
        {
            if (persistentGeometry.lodGroupsCount[i] != persistentGeometry.lodLoadedGroupsCount[i])
                continue;  // level not fully resident

            const shaderio::LodLevel& lvl          = geometryView.lodLevels[i];
            const uint32_t            levelClusters = lvl.clusterCount;
            if (levelClusters <= shaderio::kStreamingCachedBlasMaxClusters)
            {
                sgpatch.cachedBlasLodLevel = uint16_t(i);
                cachedClustersCount        = levelClusters;
                for (uint32_t gi = 0; gi < lvl.groupCount
                                       && (lvl.groupOffset + gi) < geometryView.groupInfos.size(); ++gi)
                    cachedTrianglesCount += geometryView.groupInfos[lvl.groupOffset + gi].triangleCount;
                break;
            }
        }

        // Three scenarios:
        //  - lower-detail resident than before: must rebuild or invalidate
        //  - higher-detail resident than before: may rebuild, else keep existing
        //  - nothing to do
        const bool isInvalidateOnly =
            !cachedClustersCount && persistentGeometry.cachedBlasLevel != shaderio::kTraversalInvalidLodLevel;
        const bool isLowerDetail =
            cachedClustersCount && sgpatch.cachedBlasLodLevel > persistentGeometry.cachedBlasLevel;
        const bool isHigherDetail =
            cachedClustersCount && sgpatch.cachedBlasLodLevel < persistentGeometry.cachedBlasLevel;

        if (!(isLowerDetail || isHigherDetail || isInvalidateOnly))
            continue;

        if (isLowerDetail || isInvalidateOnly)
        {
            // De-allocate first: we will use less (or no) space next, guaranteed
            // to fit.  (isHigherDetail defers the free until the larger alloc
            // succeeds so it can keep the old BLAS on failure.)
            if (persistentGeometry.cachedBlasAllocation)
            {
                m_cachedBlasAllocator.subFree(persistentGeometry.cachedBlasAllocation);
            }
        }

        bool canBuild = !isInvalidateOnly
                      && (cachedClustersTotal + cachedClustersCount <= settings.blasCacheMaxClusters)
                      && (cachedBuildsTotal + 1u <= settings.blasCacheMaxBuilds);

        rtxmg::BufferSubAllocation subAllocation;
        canBuild = canBuild && AllocateCachedBlas(persistentGeometry, cachedClustersCount, settings, subAllocation);

        if (canBuild)
        {
            // Free the old allocation now that the new one succeeded (no-op for
            // the lower-detail / invalidate path that already freed above).
            if (persistentGeometry.cachedBlasAllocation)
            {
                m_cachedBlasAllocator.subFree(persistentGeometry.cachedBlasAllocation);
            }

            persistentGeometry.cachedBlasLevel       = sgpatch.cachedBlasLodLevel;
            persistentGeometry.cachedBlasAllocation  = subAllocation;
            persistentGeometry.cachedBlasUpdateFrame = m_frameIndex;

            sgpatch.cachedBlasAddress       = m_cachedBlasAllocator.subRange(subAllocation).address;
            sgpatch.cachedBlasClustersCount = uint16_t(cachedClustersCount);
            sgpatch.cachedBlasTriangles     = cachedTrianglesCount;  // render-stats

            updateTask.geometryPatches[writeIndex++] = sgpatch;

            cachedClustersTotal += cachedClustersCount;
            cachedBuildsTotal++;
        }
        else if (isHigherDetail)
        {
            // Could not get space for higher detail — leave the existing cached
            // BLAS untouched and emit no patch.
        }
        else
        {
            // Invalidate.
            persistentGeometry.cachedBlasLevel       = shaderio::kTraversalInvalidLodLevel;
            persistentGeometry.cachedBlasUpdateFrame = m_frameIndex;

            sgpatch.cachedBlasLodLevel      = uint16_t(shaderio::kTraversalInvalidLodLevel);
            sgpatch.cachedBlasAddress       = 0;
            sgpatch.cachedBlasClustersCount = 0;  // render-stats: no cached geometry
            sgpatch.cachedBlasTriangles     = 0;

            updateTask.geometryPatches[writeIndex++] = sgpatch;
        }
    }

    updateTask.geometryCachedCount         = writeIndex;
    updateTask.geometryCachedClustersCount = cachedClustersTotal;

    // The surviving patches go up with UploadTaskPatches; stream_update_scene is
    // what writes each geometry's cachedBlas* into the shader-visible Geometry,
    // so m_shaderGeometries needs no host-side mirror of them.

    if (writeIndex && DebugClusterLodLoggingEnabled())
    {
        donut::log::info("ClusterLodStreaming BLAS-caching frame=%u cachedBuilds=%u cachedClusters=%u patches=%u",
                         m_frameIndex, cachedBuildsTotal, cachedClustersTotal, writeIndex);
    }
}

bool ClusterLodStreaming::AllocateCachedBlas(const PersistentGeometry&   /*geometry*/,
                                             uint32_t                    lodClustersCount,
                                             const FrameSettings&        /*settings*/,
                                             rtxmg::BufferSubAllocation& subAllocation)
{
    // Query the worst-case explicit-destinations BLAS size for one BLAS holding
    // lodClustersCount CLAS, then sub-allocate it from the cached-BLAS pool.

    // Lazily create the AS-storage pool on first use so a caching-off run never
    // allocates it.  A failed reservation is fatal — caching can't run without.
    if (!m_cachedBlasAllocator.isInitialized())
    {
        const uint64_t maxBytes = uint64_t(m_config.maxBlasCachingMegaBytes) * 1024ull * 1024ull;

        rtxmg::BufferSubAllocator::InitInfo info;
        info.device               = m_device;
        info.descriptorTable      = m_descriptorTableManager;
        info.debugName            = "rtxmg::ClusterLodStreaming::cachedBlas";
        info.isAccelStructStorage = true;
        info.minAlignment         = m_cachedBlasAlignment;
        info.blockSize            = std::min<uint64_t>(uint64_t(16) * 1024 * 1024, maxBytes);
        info.maxAllocatedSize     = maxBytes;
        info.keepLastBlock        = false;

        if (!m_cachedBlasAllocator.init(info))
        {
            donut::log::fatal("ClusterLodStreaming: cached-BLAS allocator init failed (%llu MiB)",
                              (unsigned long long)m_config.maxBlasCachingMegaBytes);
        }
    }

    namespace cluster = nvrhi::rt::cluster;
    cluster::OperationParams params = {};
    params.maxArgCount              = 1;
    params.type                     = cluster::OperationType::BlasBuild;
    params.mode                     = cluster::OperationMode::ExplicitDestinations;
    // Must match the per-instance BLAS build, so the size query agrees with the
    // actual cached build.
    params.flags                      = cluster::OperationFlags::None;
    params.blas.maxClasPerBlasCount   = lodClustersCount;
    params.blas.maxTotalClasCount     = lodClustersCount;

    const cluster::OperationSizeInfo sizeInfo = m_device->getClusterOperationSizeInfo(params);

    return m_cachedBlasAllocator.subAllocate(subAllocation, sizeInfo.resultMaxSizeInBytes, m_cachedBlasAlignment);
}

namespace {

// Items are usually threads, but build_freegaps dispatches one sector per wave
// and passes (sectorCount, sectorsPerGroup).
inline uint32_t GetThreadGroupCount(uint32_t itemCount, uint32_t itemsPerGroup)
{
    return (itemCount + itemsPerGroup - 1u) / itemsPerGroup;
}

}  // anonymous namespace

void ClusterLodStreaming::UpdateSceneResidency(nvrhi::ICommandList*             commandList,
                                               const shaderio::StreamingUpdate& update)
{
    // ----- Deallocate CLAS memory for unloaded groups ------------------------
    if (m_requiresClas && m_config.usePersistentClasAllocator
        && update.patchUnloadGroupsCount)
    {
        assert(m_unloadGroupsPso && m_streamingBindingSet);
        nvrhi::ComputeState state;
        state.pipeline = m_unloadGroupsPso;
        state.bindings = { m_streamingBindingSet, m_descriptorTable };
        commandList->setComputeState(state);
        commandList->dispatch(GetThreadGroupCount(update.patchUnloadGroupsCount,
                                                  shaderio::kStreamAllocatorUnloadGroupsThreads),
                              1u, 1u);

        // The next allocator consumer binds the same UAV-bearing binding set;
        // NVRHI's binding-state path inserts the required UAV barrier.
    }

    // ----- Update scene (scene-global) ---------------------------------------
    // One dispatch fills the resident/active/groupID tables, patches each
    // geometry's GroupAddress table, and writes per-cluster resident clusters
    // for load+unload patches.  Per-frame scalars come from the u7 aggregate.
    if (update.patchGroupsCount || update.patchCachedBlasCount)
    {
        const int alphaPermIdx = m_hasAlphaMask ? 1 : 0;
        assert(m_updateScenePso[alphaPermIdx] && m_streamingBindingSet);
        if (DebugClusterLodLoggingEnabled())
        {
            const uint32_t updateLoadCount =
                update.patchGroupsCount - update.patchUnloadGroupsCount;
            donut::log::info("ClusterLodStreaming stream_update_scene frame=%u task=%u patchLoad=%u patchUnload=%u patchGroups=%u newClas=%u",
                             m_frameIndex,
                             update.taskIndex,
                             updateLoadCount,
                             update.patchUnloadGroupsCount,
                             update.patchGroupsCount,
                             update.newClasCount);
        }

        nvrhi::ComputeState state;
        state.pipeline = m_updateScenePso[alphaPermIdx];
        // The bindless heap pass-through serves the shader's
        // geom.streamingGroupAddressesUAV + patch.groupAddress.srvIndex lookups.
        state.bindings = { m_streamingBindingSet, m_descriptorTable };
        commandList->setComputeState(state);
        commandList->dispatch(GetThreadGroupCount(std::max(update.patchGroupsCount, update.patchCachedBlasCount),
                                                  shaderio::kStreamUpdateSceneThreads),
                              1u, 1u);

        // The dispatch above wrote the LOAD tail of the active list; this copy
        // folds in the host's swap-with-last patches from the UNLOADs.
        m_resident.CommitActiveGroupsDelta(commandList, update.taskIndex);

    }
}

void ClusterLodStreaming::FillMixedClusterGeometryIndices(nvrhi::ICommandList*             commandList,
                                                          const shaderio::StreamingUpdate& update)
{
    // ----- Alpha-mask CLAS-geometry-indices update ---------------------------
    // Every mixed-state cluster appended by stream_update_scene above left a
    // task slot in m_newClasGeometryIndicesBuffer.  stream_dispatch_setup turns the task
    // counter into an indirect dispatch grid, then
    // stream_fill_clas_geometry_indices fills each cluster's slice with one
    // uint32 per triangle (CLAS geometry index + opacity/cull flags), which the
    // CLAS-build hardware reads via arg.geometryIndexAndFlagsBuffer.
    if (m_hasAlphaMask && update.patchGroupsCount)
    {
        assert(m_updateClasGeometryIndicesPso);
        // The task slots and counter are cross-dispatch UAV traffic, so both
        // need a barrier before the consumer reads them.
        assert(m_setupPso && m_setupBindingSet);
        nvrhi::utils::BufferUavBarrier(commandList, m_updates.GetNewClasGeometryIndicesBuffer());
        nvrhi::utils::BufferUavBarrier(commandList, m_shaderBuffer);
        commandList->commitBarriers();

        // Converts the mixed-cluster task counter into the dispatch grid at
        // update.dispatchClasGeometryIndices{X,Y,Z}.  The grid is zero when no
        // mixed cluster was queued, making the indirect dispatch a no-op.
        {
            nvrhi::ComputeState state;
            state.pipeline = m_setupPso;
            state.bindings = { m_setupBindingSet, m_residentPersistentSet };
            commandList->setComputeState(state);
            const uint32_t setup = uint32_t(shaderio::StreamSetup::UpdateGeometryIndices);
            commandList->setPushConstants(&setup, sizeof(setup));
            commandList->dispatch(1u, 1u, 1u);
        }

        // m_shaderBuffer cannot be indirectParams directly: it is simultaneously
        // bound as the u7 UAV, and D3D12 has no state that is both
        // UnorderedAccess and IndirectArgument.  Copying lands the grid in a
        // buffer that only ever holds IndirectArgument.
        assert(m_clasGeometryIndicesDispatchBuffer);
        commandList->copyBuffer(
            m_clasGeometryIndicesDispatchBuffer, 0u,
            m_shaderBuffer, offsetof(shaderio::SceneStreaming, update)
                + offsetof(shaderio::StreamingUpdate, dispatchClasGeometryIndicesX),
            3u * sizeof(uint32_t));

        // Each thread group handles kGeometryIndicesTasksPerGroup tasks, one
        // wave per task; the shader early-exits past the task counter.
        {
            nvrhi::ComputeState state;
            state.pipeline = m_updateClasGeometryIndicesPso;
            state.bindings = { m_streamingBindingSet, m_descriptorTable };
            state.indirectParams = m_clasGeometryIndicesDispatchBuffer;
            commandList->setComputeState(state);
            commandList->dispatchIndirect(0u);
        }

        // Transition m_newClasGeometryIndicesBuffer from UAV to a shader-
        // readable state for the CLAS build. The build derefs the buffer
        // via raw VA (arg.geometryIndexAndFlagsBuffer), so nvrhi can't
        // auto-track the dependency — explicit transition needed.
        if (auto* gib = m_updates.GetNewClasGeometryIndicesBuffer())
        {
            commandList->setBufferState(gib, nvrhi::ResourceStates::ShaderResource);
            commandList->commitBarriers();
        }
    }
}

void ClusterLodStreaming::BuildNewClas(nvrhi::ICommandList*             commandList,
                                       const shaderio::StreamingUpdate& update)
{
    // ----- CLAS Implicit build for newly loaded clusters ---------------------
    // Reads the IndirectTriangleClasArgs that stream_update_scene wrote into
    // m_clasIndirectArgsBuffer (u_NewClasBuilds) and emits temporary CLAS plus
    // their addresses/sizes; FinalizeResidency moves them to the
    // allocator-chosen resident addresses.
    if (update.newClasCount)
    {
        assert(m_clasIndirectArgsBuffer && m_clasScratchBuffer);
        namespace cluster = nvrhi::rt::cluster;
        cluster::OperationParams p              = {};
        p.maxArgCount                           = update.newClasCount;
        p.type                                  = cluster::OperationType::ClasBuild;
        p.mode                                  = cluster::OperationMode::ImplicitDestinations;
        p.flags                                 = m_config.clasBuildFlags;
        p.clas                                  = m_clasTriangleInput;
        p.clas.maxTotalTriangleCount            = m_clasTriangleInput.maxTriangleCount * update.newClasCount;
        p.clas.maxTotalVertexCount              = m_clasTriangleInput.maxVertexCount  * update.newClasCount;

        cluster::OperationSizeInfo s            = m_device->getClusterOperationSizeInfo(p);

        cluster::OperationDesc desc             = {};
        desc.params                             = p;
        desc.scratchSizeInBytes                 = s.scratchSizeInBytes;
        desc.inIndirectArgsBuffer               = m_clasIndirectArgsBuffer;
        desc.inOutAddressesBuffer               = m_updates.GetNewClasAddressesBuffer();
        desc.outSizesBuffer                     = m_updates.GetNewClasSizesBuffer();
        desc.outAccelerationStructuresBuffer    = m_clasScratchBuffer;
        commandList->executeMultiIndirectClusterOperation(desc);
    }
}

void ClusterLodStreaming::PrepareClasAllocator(nvrhi::ICommandList*             commandList,
                                               const shaderio::StreamingUpdate& update)
{
    // ----- Persistent allocator: build freegaps + binning -------------------
    if (m_config.usePersistentClasAllocator
        && update.patchGroupsCount)
    {
        assert(m_buildFreegapsPso && m_streamingBindingSet);
        // build_freegaps — ONE wave per sector; the shader derives its
        // sectorID from the same threads/kWaveSize split.
        const uint32_t sectorCount = m_shaderData.clasAllocator.sectorCount;
        {
            nvrhi::ComputeState state;
            state.pipeline = m_buildFreegapsPso;
            state.bindings = { m_streamingBindingSet };
            commandList->setComputeState(state);
            commandList->dispatch(GetThreadGroupCount(sectorCount,
                                                      shaderio::kStreamAllocatorBuildFreegapsThreads
                                                          / shaderio::kWaveSize),
                                  1u, 1u);
        }

        // If we need to allocate new group clusters (loaded new groups)
        // then we need the full detail binned list of free gaps,
        // otherwise we are fine with just knowing the counters.
        if (update.patchGroupsCount > update.patchUnloadGroupsCount)
        {
            assert(m_setupPso && m_setupBindingSet && m_setupInsertionPso && m_freegapsInsertPso);
            // stream_dispatch_setup computes the indirect launch grid for
            // freegaps_insert and resets freeGapsCounter for setup_insertion.
            {
                nvrhi::ComputeState state;
                state.pipeline = m_setupPso;
                state.bindings = { m_setupBindingSet, m_residentPersistentSet };
                commandList->setComputeState(state);
                const uint32_t setup = uint32_t(shaderio::StreamSetup::AllocatorFreeInsert);
                commandList->setPushConstants(&setup, sizeof(setup));
                commandList->dispatch(1u, 1u, 1u);
            }

            // setup_insertion — one thread per size class.
            {
                nvrhi::ComputeState state;
                state.pipeline = m_setupInsertionPso;
                state.bindings = { m_streamingBindingSet };
                commandList->setComputeState(state);
                commandList->dispatch(GetThreadGroupCount(m_shaderData.clasAllocator.maxAllocationSize,
                                                          shaderio::kStreamAllocatorSetupInsertionThreads),
                                      1u, 1u);
            }

            // freegaps_insert — bins all free gaps by size into the
            // appropriate free list ranges. stream_dispatch_setup wrote the indirect
            // dispatch args into SceneStreaming::clasAllocator.
            {
                nvrhi::ComputeState state;
                state.pipeline = m_freegapsInsertPso;
                state.bindings = { m_streamingBindingSet };
                state.indirectParams = m_shaderBuffer;
                commandList->setComputeState(state);
                commandList->dispatchIndirect(
                    uint32_t(offsetof(shaderio::SceneStreaming, clasAllocator)
                        + offsetof(shaderio::StreamingAllocator, dispatchFreeGapsInsert)));
            }

            DebugDumpAllocatorFreelist(commandList, update);
        }
    }
    else if (!m_config.usePersistentClasAllocator
             && update.patchGroupsCount)
    {
        // ----- Compaction allocator: old resident CLAS ----------------------
        // Unloads punch holes in the pool, so stream_compact_defrag_old defrags
        // — it walks every OLD resident group's clusters, atomic-bumps
        // update.moveClasSize from 0, and emits (src,dst) pairs for the
        // pool→pool MOVE in FinalizeResidency.  Without unloads the resident CLAS
        // are already tightly packed, so the defrag is skipped and stream_dispatch_setup
        // only seeds the cursor from the GPU-persistent clasCompactionUsedSize
        // so compaction_new appends after the resident set.
        //
        // A zero loadActiveGroupsOffset (no old groups) dispatches 0 thread groups.
        if (update.patchUnloadGroupsCount)
        {
            assert(m_compactionOldPso && m_streamingBindingSet);
            nvrhi::ComputeState state;
            state.pipeline = m_compactionOldPso;
            state.bindings = { m_streamingBindingSet, m_compactionRawSet, m_descriptorTable };
            commandList->setComputeState(state);
            commandList->dispatch(GetThreadGroupCount(update.loadActiveGroupsOffset,
                                                      shaderio::kStreamCompactionOldClasThreads),
                                  1u, 1u);
        }
        else
        {
            assert(m_setupPso && m_setupBindingSet);
            nvrhi::ComputeState state;
            state.pipeline = m_setupPso;
            state.bindings = { m_setupBindingSet, m_residentPersistentSet };
            commandList->setComputeState(state);
            const uint32_t setup = uint32_t(shaderio::StreamSetup::CompactionOldNoUnloads);
            commandList->setPushConstants(&setup, sizeof(setup));
            commandList->dispatch(1u, 1u, 1u);
        }
    }
}

void ClusterLodStreaming::ApplyResidencyUpdate(nvrhi::ICommandList* commandList)
{
    // Prior traversal we run the update task.
    // This modifies the device address array of geometry groups so that
    // traversal knows whether a geometry group is resident or not and where to
    // find it.  It also handles the unloading by invalidating such addresses.
    //
    // For ray tracing we are building new clusters for newly loaded
    // cluster groups.
    //
    // This function is called by the renderer.

    const shaderio::StreamingUpdate& update = m_shaderData.update;

    // Allocation phase (part 1): CLAS dealloc for unloaded groups + the
    // scene-global residency/address update.  Timed into Cluster Lod/Allocation.
    stats::clusterAccelSamplers.clusterLodAllocUnloadUpdateTime.Start(commandList);
    UpdateSceneResidency(commandList, update);
    stats::clusterAccelSamplers.clusterLodAllocUnloadUpdateTime.Stop();

    // rasterization ends here
    if (!m_requiresClas)
        return;

    // CLAS Build phase: per-triangle alpha-mask geometry indices + the Implicit
    // CLAS build for newly loaded clusters.  Timed into Cluster Lod/CLAS Build.
    stats::clusterAccelSamplers.clusterLodClasBuildTime.Start(commandList);
    FillMixedClusterGeometryIndices(commandList, update);
    BuildNewClas(commandList, update);
    stats::clusterAccelSamplers.clusterLodClasBuildTime.Stop();

    // Allocation phase (part 2): persistent freegaps/binning OR compaction
    // defrag of old resident CLAS.  Timed into Cluster Lod/Allocation.
    stats::clusterAccelSamplers.clusterLodAllocFreegapsTime.Start(commandList);
    PrepareClasAllocator(commandList, update);
    stats::clusterAccelSamplers.clusterLodAllocFreegapsTime.Stop();
}

void ClusterLodStreaming::FinalizeResidency(nvrhi::ICommandList* commandList, bool runAgeFilter)
{
    // After traversal was performed, this function filters resident cluster
    // groups by age to append to the unload request list.
    // The traversal itself will have appended load requests and reset the age
    // of used cluster groups.
    //
    // For ray tracing we compact all resident clusters and append (also
    // compacted) the newly built clusters from the previous `ApplyResidencyUpdate`
    // step.
    //
    // This function is called by the renderer.

    const shaderio::StreamingUpdate& update = m_shaderData.update;

    // Allocation phase (part 3): age-filter eviction pass.  Timed into
    // Cluster Lod/Allocation.
    stats::clusterAccelSamplers.clusterLodAllocAgeTime.Start(commandList);

    // ----- Age filter (scene-global) -----------------------------------------
    // One dispatch over the active suffix: the activeGroups UAV is bound WHOLE
    // and the shader adds resident.persistentGroupsCount to skip the persistent
    // low-detail prefix.  Evicted (geometryID, groupID) pairs are atomically
    // appended to request.unloadGeometryGroups, which StageResidencyUpdate reads back.
    if (m_shaderData.resident.activeGroupsCount && runAgeFilter)
    {
        assert(m_agefilterPso && m_streamingBindingSet);
        nvrhi::ComputeState state;
        state.pipeline = m_agefilterPso;
        state.bindings = { m_streamingBindingSet };
        commandList->setComputeState(state);

        commandList->dispatch(GetThreadGroupCount(m_shaderData.resident.activeGroupsCount,
                                                  shaderio::kStreamAgeFilterGroupsThreads),
                              1u, 1u);
    }
    stats::clusterAccelSamplers.clusterLodAllocAgeTime.Stop();

    // rasterization ends here
    if (!m_requiresClas)
        return;

    const uint32_t patchLoadGroupsCount =
        update.patchGroupsCount - update.patchUnloadGroupsCount;

    // Allocation phase (part 4): persistent load_groups OR compaction old-CLAS
    // move + new-CLAS compaction.  Timed into Cluster Lod/Allocation.
    stats::clusterAccelSamplers.clusterLodAllocLoadTime.Start(commandList);

    // ----- Persistent allocator: load_groups ---------------------------------
    if (m_config.usePersistentClasAllocator)
    {
        if (patchLoadGroupsCount)
        {
            assert(m_loadGroupsPso && m_streamingBindingSet);
            if (DebugClusterLodLoggingEnabled())
            {
                donut::log::info("ClusterLodStreaming load_groups frame=%u task=%u patchLoad=%u patchOffset=%u newClas=%u",
                                 m_frameIndex,
                                 update.taskIndex,
                                 patchLoadGroupsCount,
                                 update.patchUnloadGroupsCount,
                                 update.newClasCount);
            }
            nvrhi::ComputeState state;
            state.pipeline = m_loadGroupsPso;
            state.bindings = { m_streamingBindingSet, m_descriptorTable };
            commandList->setComputeState(state);
            commandList->dispatch(GetThreadGroupCount(patchLoadGroupsCount,
                                                      shaderio::kStreamAllocatorLoadGroupsThreads),
                                  1u, 1u);
        }
    }
    else
    {
        // ----- Compaction allocator: move old CLAS, compact new -------------
        namespace cluster = nvrhi::rt::cluster;

        // (1) Relocate the OLD resident CLAS to the compacted destinations
        // stream_compact_defrag_old computed in ApplyResidencyUpdate.  src and dst
        // both live in the persistent pool, so overlap is allowed (flags =
        // None).  Only unload frames ran the defrag, so only they have moves.
        const uint32_t oldClasCount = update.loadActiveClustersOffset;
        if (update.patchUnloadGroupsCount && oldClasCount)
        {
            assert(m_resident.GetClasDataBuffer());
            // Make stream_compact_defrag_old's UAV writes to the move arrays
            // visible, and put the pool in AS-build state for the in-place move.
            commandList->setBufferState(m_resident.GetClasDataBuffer(),
                                        nvrhi::ResourceStates::AccelStructBuildBlas);
            commandList->commitBarriers();

            cluster::OperationParams p = {};
            p.maxArgCount              = oldClasCount;
            p.type                     = cluster::OperationType::Move;
            p.mode                     = cluster::OperationMode::ExplicitDestinations;
            p.flags                    = cluster::OperationFlags::None;   // old CLAS may overlap themselves
            p.move.type                = cluster::OperationMoveType::ClusterLevel;
            p.move.maxBytes            = uint32_t(std::min<size_t>(uint32_t(~0u),
                                            std::min<size_t>(size_t(m_clasSingleMaxSize) * oldClasCount,
                                                             size_t(m_config.maxClasMegaBytes) * 1024 * 1024)));

            cluster::OperationSizeInfo s = m_device->getClusterOperationSizeInfo(p);

            cluster::OperationDesc desc          = {};
            desc.params                          = p;
            desc.scratchSizeInBytes              = s.scratchSizeInBytes;
            desc.inIndirectArgCountBuffer        = m_shaderBuffer;
            desc.inIndirectArgCountOffsetInBytes = offsetof(shaderio::SceneStreaming, update)
                                                 + offsetof(shaderio::StreamingUpdate, moveClasCounter);
            desc.inIndirectArgsBuffer            = m_updates.GetMoveClasSrcAddressesBufferTyped();
            desc.inOutAddressesBuffer            = m_updates.GetMoveClasDstAddressesBufferTyped();
            desc.outAccelerationStructuresBuffer = m_resident.GetClasDataBuffer();
            commandList->executeMultiIndirectClusterOperation(desc);

            // Wait for the move to finish reading moveClas{Src,Dst} before
            // stream_compact_append_new overwrites them from index 0.
            commandList->setBufferState(m_resident.GetClasDataBuffer(),
                                        nvrhi::ResourceStates::AccelStructBuildBlas);
            commandList->commitBarriers();
        }

        // (2) Assign compacted destinations for the NEWLY built CLAS, appending
        // after the old set (continues the update.moveClasSize cursor).  Writes
        // moveClas{Src,Dst}Addresses[0..newClasCount-1]; the scratch→pool MOVE
        // below consumes them (count = newClasCount).
        if (update.newClasCount)
        {
            assert(m_compactionNewPso && m_streamingBindingSet);
            nvrhi::ComputeState state;
            state.pipeline = m_compactionNewPso;
            state.bindings = { m_streamingBindingSet, m_compactionRawSet };
            commandList->setComputeState(state);
            commandList->dispatch(GetThreadGroupCount(update.newClasCount,
                                                      shaderio::kStreamCompactionNewClasThreads),
                                  1u, 1u);
        }
    }
    stats::clusterAccelSamplers.clusterLodAllocLoadTime.Stop();

    // CLAS Move new (scratch → persistent destinations).  Attributed per the
    // allocator that owns the move: the persistent allocator's move places
    // freshly built CLAS into the resident pool (counts as Cluster Lod/CLAS
    // Build); the Streaming Compact allocator's move is allocator overhead
    // (counts as Cluster Lod/Allocation).
    auto& clusterLodClasMoveTimer = m_config.usePersistentClasAllocator
        ? stats::clusterAccelSamplers.clusterLodClasMovePersistentTime
        : stats::clusterAccelSamplers.clusterLodClasMoveCompactionTime;
    clusterLodClasMoveTimer.Start(commandList);

    // ----- CLAS Move new (scratch → persistent destinations) -----------------
    // Moves the newly-built CLAS out of m_clasScratchBuffer to the per-cluster
    // destinations recorded in m_updates.{moveClasSrc,moveClasDst}Addresses,
    // which point into the persistent pool.  NoOverlap is safe here: src and
    // dst are two disjoint buffers (unlike the compaction-old path).
    if (update.newClasCount)
    {
        assert(m_clasScratchBuffer && m_resident.GetClasDataBuffer());
        RTXMGBuffer<uint64_t>& moveSrcAddresses = m_updates.GetMoveClasSrcAddressesBufferTyped();
        RTXMGBuffer<uint64_t>& moveDstAddresses = m_updates.GetMoveClasDstAddressesBufferTyped();

        DebugReadbackMoveArgs(commandList, moveSrcAddresses, moveDstAddresses, update);

        // The Move operation reads source CLAS objects by GVA from
        // moveSrcBuffer, so the cluster-op wrapper cannot infer this source
        // storage buffer. Make the preceding CLAS-build writes visible.
        commandList->setBufferState(m_clasScratchBuffer,
                                    nvrhi::ResourceStates::AccelStructRead);
        commandList->commitBarriers();

        namespace cluster = nvrhi::rt::cluster;
        cluster::OperationParams p     = {};
        p.maxArgCount                  = update.newClasCount;
        p.type                         = cluster::OperationType::Move;
        p.mode                         = cluster::OperationMode::ExplicitDestinations;
        p.flags                        = cluster::OperationFlags::NoOverlap;
        p.move.type                    = cluster::OperationMoveType::ClusterLevel;
        p.move.maxBytes                = uint32_t(m_clasScratchNewClasSize);

        cluster::OperationSizeInfo s   = m_device->getClusterOperationSizeInfo(p);

        cluster::OperationDesc desc                  = {};
        desc.params                                  = p;
        desc.scratchSizeInBytes                      = s.scratchSizeInBytes;
        if (m_config.usePersistentClasAllocator)
        {
            // Persistent: count = moveClasCounter (load_groups zeroes it on
            // alloc failure to skip the move).  Host seeds it to newClusterCount.
            desc.inIndirectArgCountBuffer            = m_shaderBuffer;
            desc.inIndirectArgCountOffsetInBytes     = offsetof(shaderio::SceneStreaming, update)
                                                     + offsetof(shaderio::StreamingUpdate, moveClasCounter);
        }
        // else (compaction): leave inIndirectArgCountBuffer null so the op uses
        // p.maxArgCount (== newClasCount) directly.  moveClasCounter is unusable
        // here — it holds the OLD-CLAS move count from compaction_old.
        desc.inIndirectArgsBuffer                    = moveSrcAddresses;
        desc.inOutAddressesBuffer                    = moveDstAddresses;
        desc.outAccelerationStructuresBuffer         = m_resident.GetClasDataBuffer();
        commandList->executeMultiIndirectClusterOperation(desc);

        // The following BLAS build consumes these CLAS by GPU virtual address,
        // so nvrhi cannot infer this source buffer from the build desc.
        commandList->setBufferState(m_resident.GetClasDataBuffer(),
                                    nvrhi::ResourceStates::AccelStructBuildBlas);
        commandList->commitBarriers();
    }
    clusterLodClasMoveTimer.Stop();

    // Allocation phase (part 5): status / max-sized publish.  Timed into
    // Cluster Lod/Allocation.
    stats::clusterAccelSamplers.clusterLodAllocStatusTime.Start(commandList);

    // ----- Status / max-sized publish ----------------------------------------
    // Stores the worst-case budget into FrameRequest for the next frame's load
    // throttling.
    if (m_setupPso && m_setupBindingSet)
    {
        nvrhi::ComputeState state;
        state.pipeline = m_setupPso;
        state.bindings = { m_setupBindingSet, m_residentPersistentSet };
        commandList->setComputeState(state);
        // Persistent → AllocatorStatus (publishes clasAllocatedMaxSizedLeft);
        // compaction → CompactionStatus (persists resident.clasCompactionUsedSize
        // and publishes request.clasCompactionUsedSize/Count).
        const uint32_t setup = uint32_t(m_config.usePersistentClasAllocator
                                        ? shaderio::StreamSetup::AllocatorStatus
                                        : shaderio::StreamSetup::CompactionStatus);
        commandList->setPushConstants(&setup, sizeof(setup));
        commandList->dispatch(1u, 1u, 1u);
    }
    stats::clusterAccelSamplers.clusterLodAllocStatusTime.Stop();
}

void ClusterLodStreaming::CaptureFrameRequests(nvrhi::ICommandList* commandList)
{
    // Copy the GPU's authoritative StreamingFrameRequest for this frame back to
    // the host-mapped request buffer, for the next StageResidencyUpdate to pop.

    m_requests.DownloadFrameRequests(commandList,
                                     m_shaderData.request,
                                     m_shaderBuffer,
                                     offsetof(shaderio::SceneStreaming, request));

    m_pendingSubmittedTasks.push_back({
        PendingSubmittedTask::TaskQueue::Requests,
        m_shaderData.request.taskIndex,
    });

    // Refresh the per-(geometry, LOD) resident CLAS-bytes aggregate, only on
    // frames a residency report asked for it — the readback copies the whole
    // per-group CLAS-sizes buffer and sweeps the resident set on the CPU.
    if (m_requiresClas && m_config.usePersistentClasAllocator && m_residentClasStatsRequested)
    {
        m_residentClasStatsRequested = false;
        m_resident.DownloadResidentClasBytes(
            commandList, m_clasAllocator.GetShaderData().granularityByteShift, m_residentClasBytes);
    }

    m_frameIndex++;

    // For benchmarking: track when total geometry residency peaks (useful
    // to see how many frames it takes streaming to converge).  Gated because
    // the log line it feeds only prints under this flag.
    if (DebugClusterLodLoggingEnabled())
    {
        size_t geoSize = GetGeometrySize(false);
        if (geoSize > m_peakGeometrySize)
        {
            m_peakGeometrySize = geoSize;
            m_peakFrameIndex   = m_frameIndex;
        }
        else if (m_frameIndex == m_peakFrameIndex + 2)
        {
            donut::log::info("streaming: geometry peak frame %u", m_peakFrameIndex);
        }
    }
}

void ClusterLodStreaming::SignalTasksSubmitted()
{
    // Fences each task against the graphics queue's last submission, so this
    // must run after the frame's command list has been executed — signalling
    // early would let the host recycle task memory the GPU is still reading.
    for (const PendingSubmittedTask& task : m_pendingSubmittedTasks)
    {
        switch (task.taskQueue)
        {
        case PendingSubmittedTask::TaskQueue::Requests:
            m_requestsTaskQueue.Push(task.taskIndex);
            break;
        case PendingSubmittedTask::TaskQueue::Updates:
            m_updatesTaskQueue.Push(task.taskIndex);
            break;
        case PendingSubmittedTask::TaskQueue::Storage:
            m_storageTaskQueue.Push(task.taskIndex);
            break;
        }
    }

    m_pendingSubmittedTasks.clear();
}

void ClusterLodStreaming::GetStats(StreamingStats& stats) const
{
    stats = m_stats;

    const StreamingStorage::PoolStats pool = m_storage.GetStats();
    stats.reservedDataBytes  = pool.reservedDataBytes;
    stats.usedDataBytes      = pool.usedDataBytes;
    stats.allocatedDataBytes = pool.allocatedDataBytes;

    const StreamingResident::ResidencyStats resident = m_resident.GetStats();
    stats.residentGroups      = resident.residentGroups;
    stats.residentClusters    = resident.residentClusters;
    stats.residentTriangles   = resident.residentTriangles;
    stats.persistentGroups    = resident.persistentGroups;
    stats.persistentClusters  = resident.persistentClusters;
    stats.persistentTriangles = resident.persistentTriangles;

    stats.operationsBytes     = GetOperationsSize();
    stats.persistentDataBytes = GetPersistentDataBytes();
    stats.persistentClasBytes = m_clasLowDetailBuffer.GetVramBytes();
    stats.residentNormals = !m_config.stripResidentNormals;

    if (m_config.allowBlasCaching)
    {
        uint32_t cached = 0;
        for (const PersistentGeometry& g : m_persistentGeometries)
            if (g.cachedBlasLevel != shaderio::kTraversalInvalidLodLevel)
                ++cached;
        const rtxmg::BufferSubAllocator::Report report = m_cachedBlasAllocator.getReport();
        stats.cachedBlasCount          = cached;
        stats.cachedBlasBytes          = report.requestedSize;
        stats.allocatedCachedBlasBytes = report.allocatedSize;
        stats.maxCachedBlasBytes       = uint64_t(m_config.maxBlasCachingMegaBytes) << 20;
    }
}

float ClusterLodStreaming::GetLoadFactor() const
{
    const StreamingStorage::PoolStats pool = m_storage.GetStats();

    // Geometry pool: bytes sub-allocated out of the pool budget.
    float pct = m_stats.maxDataBytes
                    ? float(double(pool.usedDataBytes) / double(m_stats.maxDataBytes))
                    : 0.0f;

    // CLAS pool: BYTES occupancy (used + granule-padding waste), NOT the
    // worst-case max-sized-gap metric.  maxSizedLeft is the right signal for
    // load-scheduling guarantees but ratchets as a load measure — fragmentation
    // decays it even while byte usage falls, pinning the factor high and sending
    // the adaptive LoD error into unbounded growth (coarsening cannot
    // defragment).  The pool is fixed-size, so occupancy has to be
    // fragmentation-neutral.
    if (m_stats.reservedClasBytes)
    {
        pct = std::max(pct, float(double(m_stats.usedClasBytes + m_stats.wastedClasBytes)
                                  / double(m_stats.reservedClasBytes)));
    }
    return pct;
}

bool ClusterLodStreaming::GetResidencyReport(ResidencyReport& out)
{
    m_residentClasStatsRequested = true;

    out.stripResidentPositions = m_config.stripResidentPositions;
    out.stripResidentNormals   = m_config.stripResidentNormals;
    out.residentClasBytes      = m_residentClasBytes;

    out.cachedBlasLevels.resize(m_persistentGeometries.size());
    for (size_t i = 0; i < m_persistentGeometries.size(); ++i)
        out.cachedBlasLevels[i] = m_persistentGeometries[i].cachedBlasLevel;

    const uint64_t epoch = m_resident.GetResidencyEpoch();
    if (!out.residentGroups.empty() && out.residencyEpoch == epoch)
        return false;

    out.residencyEpoch = epoch;
    out.residentGroups.clear();
    m_resident.CollectActiveGeometryGroups(out.residentGroups);
    out.pinnedGroupsCount = m_resident.GetLowDetailGroupsCount();
    return true;
}

size_t ClusterLodStreaming::GetClasSize(bool reserved) const
{
    if (reserved)
    {
        return m_clasLowDetailSize + m_stats.reservedClasBytes;
    }
    else
    {
        return m_clasLowDetailSize + m_stats.usedClasBytes;
    }
}

size_t ClusterLodStreaming::GetBlasSize(bool reserved) const
{
    size_t size = m_blasSize;

    if (m_requiresClas && m_config.allowBlasCaching)
    {
        const rtxmg::BufferSubAllocator::Report report = m_cachedBlasAllocator.getReport();

        if (reserved)
        {
            size += report.reservedSize + report.freeSize;
        }
        else
        {
            size += report.requestedSize;
        }
    }

    return size;
}

size_t ClusterLodStreaming::GetGeometrySize(bool reserved) const
{
    const StreamingStorage::PoolStats pool = m_storage.GetStats();

    if (reserved)
    {
        return GetPersistentDataBytes() + pool.reservedDataBytes;
    }
    else
    {
        return GetPersistentDataBytes() + pool.usedDataBytes;
    }
}

ClusterLodStreaming::~ClusterLodStreaming()
{
    // m_resident's IDPool must release its live IDs before its destructor runs
    // (IDPool::deinit asserts on non-zero m_usedIDs).
    if (m_device)
        Deinit();
}

// ----- ClusterLodResources getter overrides --------------------------------
// Forward to the StreamingResident sub-manager.  The base class's
// typed resident-table wrappers stay empty on the streaming class (Preloaded
// populates them directly); the equivalent streaming data lives on m_resident.
const RTXMGBuffer<uint64_t>& ClusterLodStreaming::GetResidentClasAddressesBuffer() const
{
    return m_resident.GetClasAddressesBuffer();
}

const RTXMGBuffer<shaderio::ClusterAddress>& ClusterLodStreaming::GetResidentClustersBuffer() const
{
    return m_resident.GetClustersBuffer();
}

const RTXMGBuffer<shaderio::StreamingGroup>& ClusterLodStreaming::GetResidentGroupsBuffer() const
{
    return m_resident.GetGroupsBuffer();
}

// traversal_run.hlsl load-emit bindings.
nvrhi::IBuffer* ClusterLodStreaming::GetStreamingShaderBuffer() const
{
    return m_shaderBuffer;
}

nvrhi::IBuffer* ClusterLodStreaming::GetStreamingLoadGroupsBuffer() const
{
    return m_requests.GetRequestBuffer();
}

// BLAS merging — age-filter buffers for traversal_blas_merging.  The buffer is
// bound WHOLE; the merge kernel adds resident.persistentGroupsCount in-shader to
// skip the persistent low-detail prefix (matching stream_age_groups's u8 binding
// in UpdateBindings).
nvrhi::IBuffer* ClusterLodStreaming::GetActiveGroupsBuffer() const
{
    return m_resident.GetActiveGroupsBuffer();
}
nvrhi::IBuffer* ClusterLodStreaming::GetGroupIDsBuffer() const
{
    return m_resident.GetGroupIDsBuffer();
}
nvrhi::IBuffer* ClusterLodStreaming::GetUnloadRequestBuffer() const
{
    return m_requests.GetRequestBuffer();
}
uint64_t ClusterLodStreaming::GetUnloadRequestRingBytes() const
{
    return m_requests.GetRequestSlotSize() * uint64_t(kStreamingMaxActiveTasks);
}

bool ClusterLodStreaming::UpdateClasRequired(bool state, nvrhi::ICommandList* commandList)
{
    bool result = true;
    if (state != m_requiresClas)
    {
        if (state)
        {
            result = InitClas(m_device, commandList);
        }
        else
        {
            DeinitClas();
        }
    }

    LogDebugRendererBeginFrame();

    return result;
}

void ClusterLodStreaming::Deinit()
{
    if (!m_device)
    {
        return;
    }

    DeinitClas();
    DeinitShadersAndPipelines();

    m_resident.Deinit();
    m_storage.Deinit();
    m_updates.Deinit();
    m_requests.Deinit();

    m_requestsTaskQueue.Deinit();
    m_storageTaskQueue.Deinit();
    m_updatesTaskQueue.Deinit();

    // The BaseGeometry slices and their descriptor handles release through their
    // destructors when m_persistentGeometries clears.
    m_persistentGeometries.clear();
    m_lowDetailGroupsDataSRVHandle.Reset();
    m_lowDetailGroupsData.Release();
    m_lowDetailClasGeometryIndices.Release();
    m_persistentGeometrySize = 0;

    m_shaderGeometriesBuffer.Release();
    m_renderInstancesBuffer.Release();   // inherited
    m_shaderBuffer.Release();
    m_streamingDummyBuffer.Release();
    m_shaderGeometries.clear();
    m_renderInstances.clear();
    m_geometries.clear();
    m_shaderData = {};

    m_shaderFactory = nullptr;
    m_device = nullptr;
}

void ClusterLodStreaming::Reset(nvrhi::ICommandList* commandList)
{
    assert(m_device);
    m_device->waitForIdle();

    m_peakFrameIndex   = ~0u;
    m_peakGeometrySize = 0;

    // Recycle the task queues' EventQuery handles.  StreamingTaskQueue holds
    // nvrhi::EventQueryHandle members, so a deinit + init pair is needed
    // rather than a plain default-construct reset.
    m_requestsTaskQueue.Deinit();
    m_storageTaskQueue.Deinit();
    m_updatesTaskQueue.Deinit();
    m_requestsTaskQueue.Init(m_device);
    m_storageTaskQueue.Init(m_device);
    m_updatesTaskQueue.Init(m_device);

    // reset resident objects to just roots
    m_resident.Reset(m_shaderData.resident);
    m_updates.Reset();

    // reset dynamic storage
    m_storage.Reset();

    // need to reset internal clock
    m_frameIndex = 1;
    m_pendingSubmittedTasks.clear();

    ResetGeometryStreamingState(commandList);
    if (m_requiresClas && m_config.allowBlasCaching)
    {
        ResetCachedBlas(commandList);
    }
    if (m_requiresClas && m_config.usePersistentClasAllocator)
    {
        m_clasAllocator.ClearManagementBuffer(commandList);
    }
}

bool ClusterLodStreaming::InitShadersAndPipelines(donut::engine::ShaderFactory* shaderFactory,
                                                  nvrhi::IDevice*               device)
{
    using donut::engine::ShaderMacro;

    auto loadCS = [&](nvrhi::ShaderHandle& out,
                      const std::string&   path,
                      std::vector<ShaderMacro> defines = {}) -> bool
    {
        out = shaderFactory->CreateShader(path.c_str(), "main",
                                          defines.empty() ? nullptr : &defines,
                                          nvrhi::ShaderType::Compute);
        if (!out)
        {
            donut::log::error("ClusterLodStreaming: failed to load %s", path.c_str());
            return false;
        }
        return true;
    };

    // HAS_ALPHA_TEST gates the mixed-cluster task-append in
    // stream_update_scene.hlsl.  Both blobs are loaded; dispatch picks by
    // m_hasAlphaMask.
    auto sceneMacros = [](const char* hasAlpha) {
        return std::vector<ShaderMacro>{ {"HAS_ALPHA_TEST", hasAlpha} };
    };

    // Shaders that use the cluster-operation layouts come in one build per cluster API.
    auto clusterApiPath = [device](const char* path) { return GetClusterApiShaderPath(device, path); };

    bool ok = true;
    ok &= loadCS(m_shaders.computeAgeFilterGroups,    "cluster_lod/stream_age_groups.hlsl");
    ok &= loadCS(m_shaders.computeSetup,              clusterApiPath("cluster_lod/stream_dispatch_setup.hlsl"));
    ok &= loadCS(m_shaders.computeUpdateScene[0], clusterApiPath("cluster_lod/stream_update_scene.hlsl"), sceneMacros("0"));
    ok &= loadCS(m_shaders.computeUpdateScene[1], clusterApiPath("cluster_lod/stream_update_scene.hlsl"), sceneMacros("1"));

    // Mixed-cluster alpha-mask geometry-indices fill shader, dispatched
    // indirectly from ApplyResidencyUpdate between stream_update_scene (which
    // appends per-cluster tasks) and the CLAS build.
    ok &= loadCS(m_shaders.computeUpdateClasGeometryIndices,
                 "cluster_lod/stream_fill_clas_geometry_indices.hlsl");

    if (m_config.usePersistentClasAllocator)
    {
        ok &= loadCS(m_shaders.computeAllocatorBuildFreeGaps,  "cluster_lod/stream_allocator_scan_gaps.hlsl");
        ok &= loadCS(m_shaders.computeAllocatorFreeGapsInsert, "cluster_lod/stream_allocator_bin_gaps.hlsl");
        ok &= loadCS(m_shaders.computeAllocatorLoadGroups,     clusterApiPath("cluster_lod/stream_allocator_alloc_groups.hlsl"));
        ok &= loadCS(m_shaders.computeAllocatorSetupInsertion, "cluster_lod/stream_allocator_bin_offsets.hlsl");
        ok &= loadCS(m_shaders.computeAllocatorUnloadGroups,   "cluster_lod/stream_allocator_free_groups.hlsl");
    }
    else
    {
        // Compaction allocator (non-persistent): the two compaction shaders
        // defrag old resident CLAS to the pool base and append newly built
        // CLAS after them.
        ok &= loadCS(m_shaders.computeCompactionClasOld, "cluster_lod/stream_compact_defrag_old.hlsl");
        ok &= loadCS(m_shaders.computeCompactionClasNew, clusterApiPath("cluster_lod/stream_compact_append_new.hlsl"));
    }

    if (!ok)
    {
        return false;
    }

    // ---- BindingLayouts -----------------------------------------------------
    // One unified layout (m_streamingBindingLayout) plus one BindingSet rebuilt
    // per frame in UpdateBindings() covers the allocator family,
    // stream_update_scene and stream_age_groups; load/unload_groups and
    // compaction_old pair it with the bindless heap layout, stream_dispatch_setup with
    // the push-constant variant.  Every per-frame scalar sub-struct (update /
    // resident / frame request) and the allocator header lives in ONE aggregate
    // at u7, read as streamingRW[0].<sub-struct>.
    //
    // Buffers shared by two shaders are bound ONLY as UAVs (HLSL can read from
    // RWStructuredBuffer, and a buffer can't be in SRV and UAV state at once in
    // one BindingSet).  ActiveGroups (u8) is additionally bound WHOLE, with
    // both readers adding resident.persistentGroupsCount to skip the persistent
    // low-detail prefix — Vulkan requires 16B-aligned storage-buffer descriptor
    // offsets, which an arbitrary prefix group count cannot satisfy.
    //
    // Slot map:
    //   t0  Patches                 (alloc + update_scene)
    //   t1  NewClasSizes            (alloc)
    //   t2  NewClasAddresses        (alloc)
    //   t3  GeometryPatches         (update_scene cached-write; BLAS caching)
    //   u0  ResidentGroups          (update_scene + agefilter)
    //   u1  AllocatorMem RawBuffer  (alloc — dummy in compaction path)
    //   u2  ResidentClasAddresses   (alloc)
    //   u3  ResidentClasSizes       (alloc)
    //   u4  ResidentGroupClasSizes  (alloc — dummy in compaction path)
    //   u5  MoveClasSrcAddresses    (alloc)
    //   u6  MoveClasDstAddresses    (alloc)
    //   u7  streamingRW (aggregate) (all three families)
    //   u8  ActiveGroups            (update_scene + agefilter; whole buffer)
    //   u9  GroupIDs                (update_scene + agefilter)
    //   u10 ResidentClusters        (update_scene)
    //   u11 NewClasBuilds           (update_scene)
    //   u12 UnloadGeometryGroups    (agefilter, task-slot ring of the request
    //                                buffer)
    //   u13 NewClasGeometryIndices  (update_scene + geometry-indices fill)
    //   u14 NewClasResidentIDs      (update_scene + compaction_new)
    //   u15 Geometries              (update_scene cached-write + agefilter +
    //                                allocator unload)
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        // Vulkan: descriptor set index = registerSpace (space0 -> set 0).
        // Required because this layout shares pipelines with the space-1/2
        // layouts below, and nvrhi requires the flag to match across all
        // layouts of a pipeline.  Ignored on D3D12.
        layoutDesc.registerSpaceIsDescriptorSet = true;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0),  // t0 = t_Patches
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),  // t1 = t_NewClasSizes
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),  // t2 = t_NewClasAddresses
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3),  // t3 = t_GeometryPatches (BLAS caching)
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0),  // u0 = u_ResidentGroups
            nvrhi::BindingLayoutItem::RawBuffer_UAV(1),         // u1 = u_AllocatorMem
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2),  // u2 = u_ResidentClasAddresses
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(3),  // u3 = u_ResidentClasSizes
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(4),  // u4 = u_ResidentGroupClasSizes
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(5),  // u5 = u_MoveClasSrcAddresses
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(6),  // u6 = u_MoveClasDstAddresses
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(7),  // u7 = streamingRW (SceneStreaming aggregate)
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(8),  // u8 = u_ActiveGroups (whole; shader adds persistentGroupsCount)
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(9),  // u9 = u_GroupIDs
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(10), // u10 = u_ResidentClusters
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(11), // u11 = u_NewClasBuilds
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(12), // u12 = u_UnloadGeometryGroups (agefilter)
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(13), // u13 = u_NewClasGeometryIndices
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(14), // u14 = u_NewClasResidentIDs (compaction: update_scene write / compaction_new read)
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(15), // u15 = u_Geometries (BLAS caching: update_scene writes cachedBlas*; agefilter/unload read)
        };
        m_streamingBindingLayout = device->createBindingLayout(layoutDesc);

        // stream_dispatch_setup.hlsl shares the unified layout but adds a 4-byte
        // push constant at b0 carrying the branch ID (shaderio::StreamSetup).
        // Set via commandList->setPushConstants() before each dispatch.
        layoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(uint32_t)));
        m_setupBindingLayout = device->createBindingLayout(layoutDesc);

        // Bindless heap layout for the ResourceDescriptorHeap[] per-geom
        // GroupAddress tables and per-block group header reads.  Must be
        // layout-compatible with the renderer's global bindless layout on
        // Vulkan (the table bound at dispatch is allocated from THAT layout),
        // so use the shared canonical desc rather than a local one.
        m_updateSceneBindlessLayout = device->createBindlessLayout(rtxmg::MakeGlobalBindlessLayoutDesc());

        // Compaction allocator: raw view of the SceneStreaming aggregate for the
        // 64-bit moveClasSize InterlockedAdd64, in its own register space so it
        // never collides with the mutable bindless heap.
        {
            nvrhi::BindingLayoutDesc rawLayoutDesc;
            rawLayoutDesc.visibility    = nvrhi::ShaderType::Compute;
            rawLayoutDesc.registerSpace = 1;
            rawLayoutDesc.registerSpaceIsDescriptorSet = true; // Vulkan: space1 -> set 1
            rawLayoutDesc.bindings      = { nvrhi::BindingLayoutItem::RawBuffer_UAV(0) };
            m_compactionRawLayout = device->createBindingLayout(rawLayoutDesc);
        }

        // GPU-persistent allocator scalars (StreamingResidentPersistent), bound
        // only to stream_dispatch_setup, in their own register space to avoid collisions
        // with the unified layout + bindless heap.
        {
            nvrhi::BindingLayoutDesc persistentLayoutDesc;
            persistentLayoutDesc.visibility    = nvrhi::ShaderType::Compute;
            persistentLayoutDesc.registerSpace = 2;
            persistentLayoutDesc.registerSpaceIsDescriptorSet = true; // Vulkan: space2 -> set 2
            persistentLayoutDesc.bindings      = { nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0) };
            m_residentPersistentLayout = device->createBindingLayout(persistentLayoutDesc);
        }
    }

    // ---- ComputePipelines ---------------------------------------------------
    {
        nvrhi::ComputePipelineDesc psoDesc;

        if (m_config.usePersistentClasAllocator)
        {
            psoDesc.bindingLayouts = { m_streamingBindingLayout };

            psoDesc.CS = m_shaders.computeAllocatorBuildFreeGaps;
            m_buildFreegapsPso = device->createComputePipeline(psoDesc);
            psoDesc.CS = m_shaders.computeAllocatorSetupInsertion;
            m_setupInsertionPso = device->createComputePipeline(psoDesc);
            psoDesc.CS = m_shaders.computeAllocatorFreeGapsInsert;
            m_freegapsInsertPso = device->createComputePipeline(psoDesc);

            psoDesc.bindingLayouts = { m_streamingBindingLayout, m_updateSceneBindlessLayout };
            psoDesc.CS = m_shaders.computeAllocatorLoadGroups;
            m_loadGroupsPso = device->createComputePipeline(psoDesc);
            psoDesc.CS = m_shaders.computeAllocatorUnloadGroups;
            m_unloadGroupsPso = device->createComputePipeline(psoDesc);
        }
        else
        {
            // Compaction allocator PSOs.  compaction_old reads the
            // resident group blob via the bindless heap (same layout as
            // load_groups); compaction_new touches only the unified set.
            psoDesc.bindingLayouts = { m_streamingBindingLayout, m_compactionRawLayout, m_updateSceneBindlessLayout };
            psoDesc.CS = m_shaders.computeCompactionClasOld;
            m_compactionOldPso = device->createComputePipeline(psoDesc);

            psoDesc.bindingLayouts = { m_streamingBindingLayout, m_compactionRawLayout };
            psoDesc.CS = m_shaders.computeCompactionClasNew;
            m_compactionNewPso = device->createComputePipeline(psoDesc);
        }

        psoDesc.bindingLayouts = { m_setupBindingLayout, m_residentPersistentLayout };
        psoDesc.CS = m_shaders.computeSetup;
        m_setupPso = device->createComputePipeline(psoDesc);

        // Both HAS_ALPHA_TEST permutations; dispatch picks by m_hasAlphaMask.
        psoDesc.bindingLayouts = { m_streamingBindingLayout, m_updateSceneBindlessLayout };
        for (int i = 0; i < 2; ++i)
        {
            psoDesc.CS = m_shaders.computeUpdateScene[i];
            m_updateScenePso[i] = device->createComputePipeline(psoDesc);
        }

        // Mixed-cluster geometry-indices fill — same layouts, since it reads
        // cluster headers + per-triangle material bytes from the bindless
        // ByteAddressBuffer named by each task slot's stamped srvIndex.
        psoDesc.CS = m_shaders.computeUpdateClasGeometryIndices;
        m_updateClasGeometryIndicesPso = device->createComputePipeline(psoDesc);

        psoDesc.bindingLayouts = { m_streamingBindingLayout };
        psoDesc.CS = m_shaders.computeAgeFilterGroups;
        m_agefilterPso = device->createComputePipeline(psoDesc);
    }

    return true;
}

void ClusterLodStreaming::DeinitShadersAndPipelines()
{
    m_updateScenePso[0]           = nullptr;
    m_updateScenePso[1]           = nullptr;
    m_updateClasGeometryIndicesPso = nullptr;
    m_buildFreegapsPso            = nullptr;
    m_setupInsertionPso           = nullptr;
    m_freegapsInsertPso           = nullptr;
    m_unloadGroupsPso             = nullptr;
    m_loadGroupsPso               = nullptr;
    m_compactionOldPso            = nullptr;
    m_compactionNewPso            = nullptr;
    m_setupPso                    = nullptr;
    m_agefilterPso                = nullptr;
    m_shaders                     = {};
}

void ClusterLodStreaming::InitClasSizing(nvrhi::IDevice* device, uint32_t maxNewPerFrameClusters)
{
    namespace cluster = nvrhi::rt::cluster;

    m_clasTriangleInput = {};

    m_clasTriangleInput.maxTriangleCount          = m_maxClusterTriangles;
    m_clasTriangleInput.maxVertexCount            = m_maxClusterVertices;
    // maxUniqueGeometryCount bounds packed *elements*, not index values, and the
    // encoder varies CullDisable independently of the alpha slot -- a mixed
    // cluster can emit {0,Opaque}, {0,Opaque|CullDisable}, {1,None} and
    // {1,CullDisable}.  Sidedness has no scene-wide flag, so assume it varies.
    if (m_hasAlphaMask)
    {
        m_clasTriangleInput.maxUniqueGeometryCount    = 4;
        m_clasTriangleInput.maxGeometryIndex          = 1;
    }
    else
    {
        m_clasTriangleInput.maxUniqueGeometryCount    = 2;
        m_clasTriangleInput.maxGeometryIndex          = 0;
    }

    m_clasTriangleInput.minPositionTruncateBitCount = m_clasPositionTruncateBits;
    m_clasTriangleInput.vertexFormat              = nvrhi::Format::RGB32_FLOAT;
    m_clasTriangleInput.indexFormat               = cluster::OperationIndexFormat::IndexFormat8bit;

    // (1) Implicit build, maxNewPerFrameClusters
    {
        cluster::OperationParams p              = {};
        p.maxArgCount                           = maxNewPerFrameClusters;
        p.flags                                 = m_config.clasBuildFlags;
        p.type                                  = cluster::OperationType::ClasBuild;
        p.mode                                  = cluster::OperationMode::ImplicitDestinations;
        p.clas                                  = m_clasTriangleInput;
        p.clas.maxTotalTriangleCount            = m_maxClusterTriangles * maxNewPerFrameClusters;
        p.clas.maxTotalVertexCount              = m_maxClusterVertices  * maxNewPerFrameClusters;

        cluster::OperationSizeInfo s            = device->getClusterOperationSizeInfo(p);
        m_clasScratchNewClasSize  = s.resultMaxSizeInBytes;
        m_clasScratchNewClasSize  = rtxmg::align_up(uint32_t(m_clasScratchNewClasSize),
                                                    m_clasScratchAlignment);
    }

    // (2) Explicit build of a single worst-case CLAS — sizes m_clasSingleMaxSize.
    {
        cluster::OperationParams p              = {};
        p.maxArgCount                           = 1;
        p.flags                                 = m_config.clasBuildFlags;
        p.type                                  = cluster::OperationType::ClasBuild;
        p.mode                                  = cluster::OperationMode::ExplicitDestinations;
        p.clas                                  = m_clasTriangleInput;
        p.clas.maxTotalTriangleCount            = m_maxClusterTriangles;
        p.clas.maxTotalVertexCount              = m_maxClusterVertices;
        cluster::OperationSizeInfo s            = device->getClusterOperationSizeInfo(p);
        m_clasSingleMaxSize = s.resultMaxSizeInBytes;
    }
}

void ClusterLodStreaming::InitClasAllocator(nvrhi::IDevice* device, uint32_t clusterByteAlignment)
{
    // ----- Persistent GPU CLAS allocator -------------------------------------
    if (m_config.usePersistentClasAllocator)
    {
        // maxAllocationByteSize = worst-case CLAS bytes for a single group's
        // clusters; granularityByteSize = clusterByteAlignment scaled by the
        // config-driven shift.
        const uint32_t maxAllocationByteSize = uint32_t(m_clasSingleMaxSize)
                                              * m_bakerConfig.clusterGroupSize;
        const uint32_t granularityByteSize   = clusterByteAlignment 
                                              << m_config.clasAllocatorGranularityShift;
        m_clasAllocator.Init(device,
                             m_config.maxClasMegaBytes,
                             maxAllocationByteSize,
                             granularityByteSize,
                             m_config.clasAllocatorSectorSizeShift,
                             m_shaderData.clasAllocator);

        m_clasOperationsSize += m_clasAllocator.GetOperationsSize();
        m_stats.maxSizedReserved = m_clasAllocator.GetMaxSized();
    }
    else
    {
        m_stats.maxSizedReserved =
            uint32_t(m_stats.reservedClasBytes
                     / (m_clasSingleMaxSize * m_bakerConfig.clusterGroupSize));
    }
}

ClusterLodStreaming::LowDetailClasArgs
ClusterLodStreaming::BuildLowDetailClasArgs(nvrhi::IDevice*      device,
                                            nvrhi::ICommandList* commandList)
{
    namespace cluster = nvrhi::rt::cluster;

    // ----- Resident pool + persistent prefix ---------------------------------
    m_resident.InitClas(device, m_config, m_shaderData.resident);

    const uint32_t loGroupsCount           = m_resident.GetLowDetailGroupsCount();
    const uint32_t loClustersCount         = m_resident.GetLowDetailClustersCount();
    const uint32_t loMaxGroupClustersCount = m_resident.GetLowDetailMaxGroupClusters();
    const std::vector<StreamingResident::Group>& groups = m_resident.GetGroups();

    m_clasOperationsSize += m_resident.GetClasOperationsSize();
    assert(loGroupsCount == uint32_t(m_persistentGeometries.size())
           && "every geometry must contribute exactly one persistent low-detail group");

    // Zero-init the GPU-persistent allocator scalars once.  The GPU owns the
    // buffer thereafter — the host never re-uploads it, so the cursor/budget
    // survive each frame's wholesale SceneStreaming upload.
    if (m_resident.GetResidentPersistentBuffer())
    {
        const shaderio::StreamingResidentPersistent zero = {};
        commandList->writeBuffer(m_resident.GetResidentPersistentBuffer(),
                                 &zero, sizeof(zero));
    }

    // ----- Populate low-detail CLAS + BLAS args ------------------------------
    // One BLAS arg per resident low-detail group (one per geometry, in geometry
    // order) and one IndirectTriangleClasArgs per cluster.  Geometry, group blob
    // and group VA all come off the resident Group itself rather than being
    // re-derived from a parallel geometry index.
    std::vector<cluster::nv::IndirectTriangleClasArgs> clasArgs(loClustersCount);
    std::vector<cluster::IndirectArgs>             blasArgs(loGroupsCount);
    const nvrhi::GpuVirtualAddress                 residentClasAddressesVA =
        m_resident.GetClasAddressesBuffer().GetGpuVirtualAddress();

    uint32_t maxTri = 0, maxVtx = 0, totalTri = 0, totalVtx = 0;
    uint32_t clusterCursor = 0;

    std::vector<uint8_t> devBlob;  // decompression scratch, reused per group

    // Mixed-cluster entries of every group + their (cluster-cursor → offset)
    // map.  Usually the lowest LoD is single-material and needs neither.
    std::vector<nvrhi::rt::cluster::GeometryIndexAndFlags> mixedEntries;
    std::vector<std::pair<uint32_t, uint32_t>> mixedClusterCursorOffsets;

    for (uint32_t g = 0; g < loGroupsCount; ++g)
    {
        const StreamingResident::Group& rg         = groups[g];
        const uint32_t                  geometryID = rg.geometryGroup.geometryID;
        const uint32_t                  groupID    = rg.geometryGroup.groupID;
        const GeometryView&             geom       = m_geometries[geometryID];

        const GroupInfo&  groupInfo = geom.groupInfos[groupID];
        const GroupView   srcGroupView(geom.groupData, groupInfo);
        assert(groupInfo.clusterCount == rg.clusterCount);

        // The args address the decompressed device blob, but a compressed
        // group's stored cluster.triangles points at its compressed vertex
        // stream.  Same handling as ClusterLodPreloaded.
        GroupView groupView = srcGroupView;
        if (groupInfo.uncompressedSizeBytes != 0u)
        {
            GroupInfo devInfo       = groupInfo;
            devInfo.offsetBytes     = 0;
            devInfo.sizeBytes       = groupInfo.uncompressedSizeBytes;
            devInfo.vertexDataCount = groupInfo.uncompressedVertexDataCount;

            devBlob.assign(groupInfo.GetDeviceSize(), uint8_t(0));
            DecompressGroup(groupInfo, srcGroupView, devBlob.data(), devBlob.size());
            groupView = GroupView(devBlob, devInfo);
        }

        blasArgs[g].clusterCount     = rg.clusterCount;
        blasArgs[g].clusterAddressStride = 8;
        blasArgs[g].clusterAddresses = residentClasAddressesVA
                                       + uint64_t(rg.clusterResidentID) * sizeof(nvrhi::GpuVirtualAddress);

        const nvrhi::GpuVirtualAddress groupBaseVA = rg.deviceAddress;

        for (uint32_t c = 0; c < groupInfo.clusterCount; ++c)
        {
            const shaderio::Cluster& cl       = groupView.clusters[c];
            const uint32_t           triCount = cl.triangleCountMinusOne + 1u;
            const uint32_t           vtxCount = cl.vertexCountMinusOne  + 1u;

            maxTri    = std::max(maxTri, triCount);
            maxVtx    = std::max(maxVtx, vtxCount);
            totalTri += triCount;
            totalVtx += vtxCount;

            const uint32_t clusterHdrOff =
                uint32_t(sizeof(shaderio::Group))
                + uint32_t(sizeof(shaderio::Cluster)) * c;
            const nvrhi::GpuVirtualAddress clusterVA = groupBaseVA + clusterHdrOff;

            cluster::nv::IndirectTriangleClasArgs& arg = clasArgs[clusterCursor];
            arg = {};
            // CLAS encodes the allocator-issued cluster resident ID; the hit
            // shader's GetClusterID() reads this back and indexes directly
            // into m_resident's clasAddresses/clusters tables.
            arg.clusterId                         = rg.clusterResidentID + c;
            arg.clusterFlags                      = 0;
            arg.triangleCount                     = triCount;
            arg.vertexCount                       = vtxCount;
            arg.positionTruncateBitCount          = m_clasPositionTruncateBits;
            arg.indexFormat                       = uint32_t(cluster::OperationIndexFormat::IndexFormat8bit);
            arg.opacityMicromapIndexFormat        = 0;
            arg.indexBufferStride                 = 1;
            arg.vertexBufferStride                = uint16_t(sizeof(float) * 3);
            arg.indexBuffer                       = clusterVA + cl.triangles;
            arg.vertexBuffer                      = clusterVA + cl.vertices;
            arg.opacityMicromapArray              = 0;
            arg.opacityMicromapIndexBuffer        = 0;

            // Uniform clusters encode baseGeometryIndexAndFlags from
            // Cluster::stateBits; mixed ones get per-triangle entries in
            // m_lowDetailClasGeometryIndices.  With materials off, every
            // cluster is forced opaque/single-sided/material 0, so it is never
            // mixed and no per-triangle buffer is needed.  The alpha-mask
            // geometry slot only exists when the scene has alpha-masked
            // materials (maxGeometryIndex 0 otherwise).
            const uint8_t stateBitsMask =
                m_hasAlphaMaskScene ? uint8_t(~0u) : uint8_t(~shaderio::ClusterState::AlphaMasked);
            const uint8_t effStateBits = (m_enableMaterials ? cl.stateBits : uint8_t(0)) & stateBitsMask;
            const uint8_t effLocalMat  = m_enableMaterials ? cl.localMaterialID : uint8_t(0);
            const bool requiresMixedGeometryBuffer =
                (effStateBits & shaderio::ClusterState::AlphaMaskedMixed) != 0u ||
                (effStateBits & shaderio::ClusterState::TwoSidedMixed)    != 0u;
            if (requiresMixedGeometryBuffer)
            {
                arg.baseGeometryIndexAndFlags         = {};
                arg.geometryIndexAndFlagsBufferStride = uint16_t(sizeof(nvrhi::rt::cluster::GeometryIndexAndFlags));
                const uint32_t entryOffset = static_cast<uint32_t>(mixedEntries.size());
                mixedClusterCursorOffsets.emplace_back(clusterCursor, entryOffset);
                const uint8_t* clusterMaterialBytes =
                    groupView.GetClusterIndices(c) + size_t(triCount) * 3u;
                if (effLocalMat == shaderio::kPerTriangleMaterials)
                {
                    for (uint32_t t = 0; t < triCount; ++t)
                        mixedEntries.push_back(
                            ClasEncodePerTriangleGeometryIndexAndFlags(clusterMaterialBytes[t]));
                }
                else
                {
                    // Fall back to the uniform state for every triangle.
                    const auto uniformEncoded =
                        ClasEncodeBaseGeometryIndexAndFlagsFromState(effStateBits);
                    for (uint32_t t = 0; t < triCount; ++t)
                        mixedEntries.push_back(uniformEncoded);
                }
                // arg.geometryIndexAndFlagsBuffer patched below once the
                // GPU buffer is created.
            }
            else
            {
                arg.baseGeometryIndexAndFlags         =
                    ClasEncodeBaseGeometryIndexAndFlagsFromState(effStateBits);
                arg.geometryIndexAndFlagsBufferStride = 0;
                arg.geometryIndexAndFlagsBuffer       = 0;
            }

            ++clusterCursor;
        }
    }
    assert(clusterCursor == loClustersCount);

    // Upload the mixed-cluster entries + patch pointers.
    m_lowDetailClasGeometryIndices.Release();
    if (!mixedEntries.empty())
    {
        m_lowDetailClasGeometryIndices.Create(
            uint32_t(mixedEntries.size()), "ClusterLodStreamingLoDetailGeomIndices", device);
        commandList->writeBuffer(m_lowDetailClasGeometryIndices,
                                 mixedEntries.data(),
                                 mixedEntries.size() * sizeof(nvrhi::rt::cluster::GeometryIndexAndFlags));
        const nvrhi::GpuVirtualAddress baseVA =
            m_lowDetailClasGeometryIndices.GetGpuVirtualAddress();
        for (auto [cur, off] : mixedClusterCursorOffsets)
            clasArgs[cur].geometryIndexAndFlagsBuffer =
                baseVA + uint64_t(off) * sizeof(nvrhi::rt::cluster::GeometryIndexAndFlags);
    }

    LowDetailClasArgs result;
    result.clasArgs.swap(clasArgs);
    result.blasArgs.swap(blasArgs);
    result.groupsCount           = loGroupsCount;
    result.clustersCount         = loClustersCount;
    result.maxGroupClustersCount = loMaxGroupClustersCount;
    result.maxTriangleCount      = maxTri;
    result.maxVertexCount        = maxVtx;
    result.totalTriangleCount    = totalTri;
    result.totalVertexCount      = totalVtx;
    return result;
}

void ClusterLodStreaming::BuildLowDetailClas(nvrhi::IDevice*          device,
                                             nvrhi::ICommandList*     commandList,
                                             const LowDetailClasArgs& lowDetail)
{
    namespace cluster = nvrhi::rt::cluster;
    const std::vector<cluster::nv::IndirectTriangleClasArgs>& clasArgs = lowDetail.clasArgs;
    const uint32_t loClustersCount = lowDetail.clustersCount;
    const uint32_t maxTri   = lowDetail.maxTriangleCount;
    const uint32_t maxVtx   = lowDetail.maxVertexCount;
    const uint32_t totalTri = lowDetail.totalTriangleCount;
    const uint32_t totalVtx = lowDetail.totalVertexCount;

    // ----- GetSizes pass to query per-cluster CLAS byte sizes ----------------
    const nvrhi::BufferHandle clasArgsBuffer = CreateAndUploadClusterArgs(
        clasArgs,
        GetGenericDesc(loClustersCount, uint32_t(sizeof(cluster::nv::IndirectTriangleClasArgs)),
                       "ClusterLodStreamingClasArgsBuffer")
            .setIsAccelStructBuildInput(true),
        commandList);

    ClusterOperationSizesBuffer           tempSizes;
    RTXMGBuffer<nvrhi::GpuVirtualAddress> tempAddrs;
    tempSizes.Create(loClustersCount, "ClusterLodStreamingClasSizesTemp",     device);
    tempAddrs.Create(loClustersCount, "ClusterLodStreamingClasAddressesTemp", device);

    cluster::OperationParams clasParams = {};
    clasParams.maxArgCount                      = loClustersCount;
    clasParams.type                             = cluster::OperationType::ClasBuild;
    clasParams.mode                             = cluster::OperationMode::GetSizes;
    clasParams.flags                            = m_config.clasBuildFlags;
    clasParams.clas.vertexFormat                = nvrhi::Format::RGB32_FLOAT;
    // Alpha-mask scenes reserve a 2-entry geometry-index range so the CLAS
    // build accounts for the per-cluster mixed-geometry metadata.  Mirrors the
    // persistent-CLAS sizing of m_clasTriangleInput above.
    // See initClas: the bound is on elements, and sidedness varies freely.
    clasParams.clas.maxGeometryIndex            = m_hasAlphaMaskScene ? 1u : 0u;
    clasParams.clas.maxUniqueGeometryCount      = m_hasAlphaMaskScene ? 4u : 2u;
    clasParams.clas.maxTriangleCount            = maxTri;
    clasParams.clas.maxVertexCount              = maxVtx;
    clasParams.clas.maxTotalTriangleCount       = totalTri;
    clasParams.clas.maxTotalVertexCount         = totalVtx;
    clasParams.clas.minPositionTruncateBitCount = m_clasPositionTruncateBits;
    clasParams.clas.indexFormat                 = cluster::OperationIndexFormat::IndexFormat8bit;

    cluster::OperationSizeInfo clasSizeInfo = device->getClusterOperationSizeInfo(clasParams);

    {
        cluster::OperationDesc getSizesDesc = {};
        getSizesDesc.params               = clasParams;
        getSizesDesc.scratchSizeInBytes   = clasSizeInfo.scratchSizeInBytes;
        getSizesDesc.inIndirectArgsBuffer = clasArgsBuffer;
        getSizesDesc.outSizesBuffer       = tempSizes.GetBuffer();
        commandList->executeMultiIndirectClusterOperation(getSizesDesc);
    }

    // compute size, storage of lo-res geometry and destination addresses etc.

    uint64_t totalClasSize = 0;
    const std::vector<uint64_t> clasSizes = tempSizes.Download(commandList);
    // A failed map returns an empty vector; DebugReadbackMoveArgs bounds the
    // same pattern, these two did not.
    if (clasSizes.size() < loClustersCount)
        donut::log::fatal("ClusterLodStreaming: CLAS size readback returned %zu of %u entries.",
                   clasSizes.size(), loClustersCount);
    for (uint32_t c = 0; c < loClustersCount; ++c)
    {
        assert(clasSizes[c] > 0 && "CLAS GetSizes returned 0 for a cluster");
        totalClasSize += clasSizes[c];
    }
    m_clasLowDetailSize = totalClasSize;

    {
        auto desc = nvrhi::BufferDesc()
                        .setByteSize(totalClasSize)
                        .setFormat(nvrhi::Format::UNKNOWN)
                        .setCanHaveUAVs(true)
                        .setCanHaveRawViews(true)
                        .setIsAccelStructStorage(true)
                        .setInitialState(nvrhi::ResourceStates::AccelStructWrite)
                        .setKeepInitialState(true)
                        .setDebugName("ClusterLodStreamingLowDetailClas");
        m_clasLowDetailBuffer.Create(desc, device);
    }

    std::vector<nvrhi::GpuVirtualAddress> clasAddrs(loClustersCount);
    const nvrhi::GpuVirtualAddress lowDetailClasBaseVA =
        m_clasLowDetailBuffer.GetBuffer()->getGpuVirtualAddress();
    uint64_t writeOffset = 0;
    for (uint32_t c = 0; c < loClustersCount; ++c)
    {
        clasAddrs[c]  = lowDetailClasBaseVA + writeOffset;
        writeOffset  += clasSizes[c];
    }
    commandList->writeBuffer(tempAddrs.GetBuffer(), clasAddrs.data(),
                             loClustersCount * sizeof(nvrhi::GpuVirtualAddress));

    // ----- ExplicitDestinations build → m_clasLowDetailBuffer ----------------
    {
        clasParams.mode = cluster::OperationMode::ExplicitDestinations;

        cluster::OperationDesc buildDesc = {};
        buildDesc.params               = clasParams;
        buildDesc.scratchSizeInBytes   = clasSizeInfo.scratchSizeInBytes;
        buildDesc.inIndirectArgsBuffer = clasArgsBuffer;
        buildDesc.inOutAddressesBuffer = tempAddrs.GetBuffer();
        commandList->executeMultiIndirectClusterOperation(buildDesc);
    }

    // The low-detail BLAS build reads source CLAS objects through addresses
    // in blasArgsBuffer, so the cluster-op wrapper cannot infer this storage.
    commandList->setBufferState(m_clasLowDetailBuffer.GetBuffer(),
                                nvrhi::ResourceStates::AccelStructBuildBlas);
    commandList->commitBarriers();

    // ----- Copy CLAS sizes + addresses into m_resident's tables --------------
    // The persistent prefix occupies [0, loClustersCount) in both tables;
    // streaming loads bump-allocate cluster IDs past it.
    commandList->writeBuffer(m_resident.GetClasAddressesBuffer().GetBuffer(),
                             clasAddrs.data(),
                             loClustersCount * sizeof(nvrhi::GpuVirtualAddress),
                             0);
    std::vector<uint32_t> residentClasSizes(loClustersCount);
    for (uint32_t c = 0; c < loClustersCount; ++c)
        residentClasSizes[c] = uint32_t(clasSizes[c]);
    commandList->writeBuffer(m_resident.GetClasSizesBuffer(),
                             residentClasSizes.data(),
                             loClustersCount * sizeof(uint32_t),
                             0);

    if (m_config.usePersistentClasAllocator)
    {
        m_clasAllocator.ClearManagementBuffer(commandList);
    }
}

void ClusterLodStreaming::BuildLowDetailBlas(nvrhi::IDevice*          device,
                                             nvrhi::ICommandList*     commandList,
                                             const LowDetailClasArgs& lowDetail)
{
    namespace cluster = nvrhi::rt::cluster;
    const std::vector<cluster::IndirectArgs>& blasArgs = lowDetail.blasArgs;
    const size_t   numGeom                 = m_persistentGeometries.size();
    const uint32_t loClustersCount         = lowDetail.clustersCount;
    const uint32_t loMaxGroupClustersCount = lowDetail.maxGroupClustersCount;

    // ----- Build per-geometry low-detail BLASes ------------------------------
    // One BLAS per geometry, each referencing that geometry's persistent
    // cluster CLAS (clusterCount = persistent group's clusterCount).  All
    // BLASes share m_clasLowDetailBlasBuffer (storage allocated to the
    // worst-case Implicit-build result size).
    RTXMGBuffer<cluster::IndirectArgs> blasArgsBuffer;
    {
        auto desc = GetGenericDesc(uint32_t(numGeom),
                                   uint32_t(sizeof(cluster::IndirectArgs)),
                                   "ClusterLodStreamingBlasArgsBuffer")
                        .setIsAccelStructBuildInput(true);
        blasArgsBuffer.Create(desc, device);
        commandList->writeBuffer(blasArgsBuffer.GetBuffer(), blasArgs.data(),
                                 numGeom * sizeof(cluster::IndirectArgs));
    }

    cluster::OperationParams blasParams = {};
    blasParams.maxArgCount               = uint32_t(numGeom);
    blasParams.type                      = cluster::OperationType::BlasBuild;
    blasParams.mode                      = cluster::OperationMode::ImplicitDestinations;
    blasParams.flags                     = cluster::OperationFlags::FastTrace;
    blasParams.blas.maxClasPerBlasCount  = loMaxGroupClustersCount;
    blasParams.blas.maxTotalClasCount    = loClustersCount;

    cluster::OperationSizeInfo blasSizeInfo = device->getClusterOperationSizeInfo(blasParams);

    {
        auto desc = nvrhi::BufferDesc()
                        .setByteSize(blasSizeInfo.resultMaxSizeInBytes)
                        .setFormat(nvrhi::Format::UNKNOWN)
                        .setCanHaveUAVs(true)
                        .setIsAccelStructStorage(true)
                        .setInitialState(nvrhi::ResourceStates::AccelStructWrite)
                        .setKeepInitialState(true)
                        .setDebugName("ClusterLodStreamingLowDetailBlas");
        m_clasLowDetailBlasBuffer.Create(desc, device);
    }
    m_blasSize = blasSizeInfo.resultMaxSizeInBytes;

    RTXMGBuffer<nvrhi::GpuVirtualAddress> blasAddrsBuffer;
    blasAddrsBuffer.Create(uint32_t(numGeom), "ClusterLodStreamingBlasAddresses", device);
    ClusterOperationSizesBuffer           blasSizesBuffer;
    blasSizesBuffer.Create(uint32_t(numGeom), "ClusterLodStreamingBlasSizes",    device);

    {
        cluster::OperationDesc blasBuildDesc = {};
        blasBuildDesc.params                          = blasParams;
        blasBuildDesc.scratchSizeInBytes              = blasSizeInfo.scratchSizeInBytes;
        blasBuildDesc.inIndirectArgsBuffer            = blasArgsBuffer.GetBuffer();
        blasBuildDesc.inOutAddressesBuffer            = blasAddrsBuffer.GetBuffer();
        blasBuildDesc.outSizesBuffer                  = blasSizesBuffer.GetBuffer();
        blasBuildDesc.outAccelerationStructuresBuffer = m_clasLowDetailBlasBuffer.GetBuffer();
        commandList->executeMultiIndirectClusterOperation(blasBuildDesc);
    }

    // Download BLAS addresses, populate m_shaderGeometries.
    const std::vector<nvrhi::GpuVirtualAddress> blasAddrs = blasAddrsBuffer.Download(commandList);
    if (blasAddrs.size() < numGeom)
        donut::log::fatal("ClusterLodStreaming: BLAS address readback returned %zu of %zu entries.",
                   blasAddrs.size(), size_t(numGeom));
    for (size_t g = 0; g < numGeom; ++g)
    {
        m_shaderGeometries[g].lowDetailBlasAddress = blasAddrs[g];
        // lowDetailClusterID is already populated by InitGeometries (= the
        // persistent group's allocator-issued cluster base ID).
    }
    commandList->writeBuffer(m_shaderGeometriesBuffer.GetBuffer(),
                             m_shaderGeometries.data(),
                             m_shaderGeometries.size() * sizeof(shaderio::Geometry));
}

bool ClusterLodStreaming::CreatePerFrameClasBuffers(nvrhi::IDevice* device,
                                                    uint32_t        maxNewPerFrameClusters)
{
    namespace cluster = nvrhi::rt::cluster;

    // ----- Per-frame CLAS-build buffers --------------------------------------
    // m_clasIndirectArgsBuffer holds the GPU-written array of
    // IndirectTriangleClasArgs that stream_update_scene fills and the
    // per-frame Implicit build consumes. Sized to the worst-case per-frame load.
    m_maxNewClustersPerFrame = maxNewPerFrameClusters;
    {
        nvrhi::BufferDesc d;
        d.structStride          = GetTriangleClasArgsBytes(device);
        d.byteSize              = uint64_t(m_maxNewClustersPerFrame) * d.structStride;
        d.isAccelStructBuildInput = true;
        d.canHaveUAVs           = true;
        d.canHaveRawViews       = true;
        d.initialState          = nvrhi::ResourceStates::Common;
        d.keepInitialState      = true;
        d.debugName             = "ClusterLodStreamingClasIndirectArgs";
        m_clasIndirectArgsBuffer = device->createBuffer(d);
        if (!m_clasIndirectArgsBuffer)
            return false;
    }

    // m_clasScratchBuffer = OperationDesc::outAccelerationStructuresBuffer for
    // the per-frame Implicit build.  Transient build/move scratch is separate
    // and nvrhi-managed (OperationDesc takes only scratchSizeInBytes).
    {
        nvrhi::BufferDesc d;
        d.byteSize             = m_clasScratchNewClasSize;
        d.canHaveUAVs          = true;
        d.canHaveRawViews      = true;
        d.isAccelStructStorage = true;
        d.initialState         = nvrhi::ResourceStates::AccelStructWrite;
        d.keepInitialState     = true;
        d.debugName            = "ClusterLodStreamingClasScratch";
        m_clasScratchBuffer = device->createBuffer(d);
        if (!m_clasScratchBuffer)
            return false;
    }

    // Indirect-dispatch args for the alpha-mask geometry-indices pass — only
    // alpha-mask scenes ever queue mixed-cluster tasks.
    if (m_hasAlphaMask)
    {
        nvrhi::BufferDesc d;
        d.byteSize           = 3u * sizeof(uint32_t);  // dispatch grid {X,Y,Z}
        d.structStride       = sizeof(uint32_t);
        d.isDrawIndirectArgs = true;
        d.initialState       = nvrhi::ResourceStates::Common;
        d.keepInitialState   = true;
        d.debugName          = "ClusterLodStreamingGeomIndicesDispatchArgs";
        m_clasGeometryIndicesDispatchBuffer = device->createBuffer(d);
        if (!m_clasGeometryIndicesDispatchBuffer)
            return false;
    }

    m_clasOperationsSize += rtxmg::BufferVramBytes(m_clasIndirectArgsBuffer)
                          + rtxmg::BufferVramBytes(m_clasScratchBuffer)
                          + rtxmg::BufferVramBytes(m_clasGeometryIndicesDispatchBuffer);

    return true;
}

bool ClusterLodStreaming::InitClas(nvrhi::IDevice* device, nvrhi::ICommandList* commandList)
{
    // Persistent low-detail CLAS + per-geometry low-detail BLAS build, plus the
    // per-frame CLAS buffers and the GPU CLAS allocator.

    const auto tClas0 = std::chrono::steady_clock::now();

    // m_requiresClas is still false here, so Reset skips the CLAS-allocator and
    // cached-BLAS resets; BuildLowDetailClas re-issues the allocator one.
    Reset(commandList);

    m_stats.reservedClasBytes = uint64_t(m_config.maxClasMegaBytes) * 1024 * 1024;
    m_clasOperationsSize      = 0;
    m_blasSize                = 0;
    m_requiresClas            = true;

    namespace cluster = nvrhi::rt::cluster;

    // ----- Cluster-property alignment constants ------------------------------
    // nvrhi exposes no cluster-AS-properties query, so use the NVAPI/D3D12
    // constants directly:
    //
    //   Vulkan name                          NVAPI / D3D12 constant                                Value
    //   -----------------------------------  ----------------------------------------------------  -----
    //   clusterByteAlignment                 NVAPI_D3D12_RAYTRACING_CLAS_BYTE_ALIGNMENT            128
    //   clusterBottomLevelByteAlignment      D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT 256
    //   clusterScratchByteAlignment          D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT 256
    constexpr uint32_t kClusterByteAlignment            = 128u;
    constexpr uint32_t kClusterBottomLevelByteAlignment = 256u;
    constexpr uint32_t kClusterScratchByteAlignment     = 256u;
    m_clasScratchAlignment = kClusterScratchByteAlignment;

    // ----- Cached-BLAS pool --------------------------------------------------
    // Only the alignment is fixed here; the pool itself is created lazily by
    // AllocateCachedBlas, so a run that never caches never pays for it.
    m_cachedBlasAlignment = kClusterBottomLevelByteAlignment;

    // ----- Sub-manager per-frame CLAS buffers --------------------------------
    // Creates m_newClas*Buffer + m_moveClas*Buffer sized to maxPerFrameLoad ×
    // clusterGroupSize — the inputs/outputs to the per-frame Implicit build
    // (ApplyResidencyUpdate) and Move ops (FinalizeResidency).
    const uint32_t maxNewPerFrameClusters =
        m_bakerConfig.clusterGroupSize * m_config.maxPerFrameLoadRequests;

    m_updates.InitClas(device, m_config, m_bakerConfig, m_maxClusterTriangles);
    m_clasOperationsSize += m_updates.GetClasOperationsSize();

    // StreamingUpdates::InitClas caches the geometry-indices buffer's GVA onto
    // its own shader data, so ApplyTask carries it through every frame;
    // stream_update_scene reads it as update.newClasGeometryIndicesBufferVA_*.
    if (auto* gib = m_updates.GetNewClasGeometryIndicesBuffer())
    {
        donut::log::info("ClusterLodStreaming alpha-mask: newClasGeometryIndicesBufferVA=0x%016llx hasAlphaMask=%d sceneMaxClusterTriangles=%u",
                         (unsigned long long)gib->getGpuVirtualAddress(),
                         int(m_hasAlphaMask),
                         m_maxClusterTriangles);
    }
    donut::log::info("ClusterLodStreaming initClas: reset + per-frame CLAS buffers %.0f ms",
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - tClas0).count());

    InitClasSizing(device, maxNewPerFrameClusters);
    InitClasAllocator(device, kClusterByteAlignment);

    const LowDetailClasArgs lowDetail = BuildLowDetailClasArgs(device, commandList);
    BuildLowDetailClas(device, commandList, lowDetail);
    BuildLowDetailBlas(device, commandList, lowDetail);

    return CreatePerFrameClasBuffers(device, maxNewPerFrameClusters);
}

void ClusterLodStreaming::DeinitClas()
{
    // No reset is required, we just destroy all clas related resources.
    // What was fitting so far, is guaranteed to fit still.

    m_resident.DeinitClas();
    m_updates.DeinitClas();
    if (m_config.usePersistentClasAllocator)
    {
        m_clasAllocator.Deinit();
    }

    m_clasLowDetailBuffer.Release();
    m_clasLowDetailBlasBuffer.Release();   // inherited from ClusterLodResourcesBase
    m_clasGeometryIndicesDispatchBuffer = nullptr;
    m_stats.reservedClasBytes = 0;

    if (m_config.allowBlasCaching)
    {
        for (auto& persistentGeometry : m_persistentGeometries)
        {
            if (persistentGeometry.cachedBlasAllocation)
            {
                m_cachedBlasAllocator.subFree(persistentGeometry.cachedBlasAllocation);
            }
        }

        m_cachedBlasAllocator.deinit();
    }

    for (auto& shaderGeometry : m_shaderGeometries)
    {
        shaderGeometry.lowDetailBlasAddress = 0;
    }

    m_clasOperationsSize      = 0;
    m_clasLowDetailSize       = 0;
    m_clasSingleMaxSize       = 0;
    m_clasScratchNewClasSize  = 0;
    m_blasSize                = 0;

    m_requiresClas = false;
}

}  // namespace rtxmg
