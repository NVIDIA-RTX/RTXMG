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

#pragma once

#include <algorithm>
#include <cassert>
#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <nvrhi/nvrhi.h>  // for nvrhi::rt::cluster::OperationFlags / EventQuery / CommandQueue

#include <optional>

#include <donut/core/math/math.h>                 // for donut::math::uint2 (m_groupClasSizesBuffer element)
#include <donut/engine/DescriptorTableManager.h>  // StreamingStorage bindless block SRVs

#include "rtxmg/cluster_lod/baker.h"             // for BakerConfig (StreamingUpdates::InitClas — clusterGroupSize / clusterTriangles)
#include "rtxmg/cluster_lod/geometry_group.h"    // for rtxmg::GeometryGroup
#include "rtxmg/cluster_lod/shaderio.h"          // for shaderio::StreamingResident
#include "rtxmg/cluster_lod/streaming_task_queue.h"  // for kStreamingMaxActiveTasks (per-task ring sizes)
#include "rtxmg/utils/alignment.h"               // for rtxmg::align_up
#include "rtxmg/utils/buffer.h"                  // for RTXMGBuffer<T>
#include "rtxmg/utils/buffer_sub_allocator.h"    // for rtxmg::BufferSubAllocator + BufferSubAllocation
#include "rtxmg/utils/cluster_operation.h"       // for ClusterOperationSizesBuffer
#include "rtxmg/utils/id_pool.h"                 // for rtxmg::IDPool


namespace rtxmg {

struct StreamingConfig
{
    bool usePersistentClasAllocator = true;
    bool allowBlasCaching           = true;
    bool debugClusterLod            = false;

    // Debug override (--nomat): force every cluster opaque/single-sided/
    // material 0, bypassing the alpha-mask geometryIndexAndFlags path.
    bool enableMaterials            = true;

    // Max group load requests consumed per frame (--maxframeloadrequests / UI);
    // the staging buffers scale with it, and large bursts stall frames.
    uint32_t maxPerFrameLoadRequests   = 256;
    uint32_t maxPerFrameUnloadRequests = 1024;  // compile-time tuning only

    // Slot cap on resident groups (--maxresidentgroups), separate from the
    // geometry/CLAS byte pools: when full, streaming wedges with those pools
    // still under budget, so size it for the scene's working set.
    uint32_t maxGroups   = 1 << 17;
    uint32_t maxClusters = 0;  // if 0 then computed from maxGroups

    size_t maxTransferMegaBytes    = 32;  // compile-time tuning only
    // Geometry and CLAS are budgeted as one ~4 GB footprint, split evenly.
    size_t maxGeometryMegaBytes    = 2048;
    // Geometry pool sub-allocation block size; maxGeometryMegaBytes is clamped to
    // at least one block (a smaller budget would pin the residency watermark).
    size_t geometryBlockMegaBytes  = 128;
    size_t maxClasMegaBytes        = 2048;  // see maxGeometryMegaBytes
    // Ceiling on the lazily-grown cached-BLAS pool (16 MiB blocks).  Running out
    // only means a geometry stays uncached and rebuilds per frame, so this is
    // sized to observed use rather than worst case; raise it in the VRAM window.
    size_t maxBlasCachingMegaBytes = 64;

    // clasBuildFlags is compile-time tuning only; clasPositionTruncateBits has
    // a CLI flag.
    nvrhi::rt::cluster::OperationFlags clasBuildFlags = nvrhi::rt::cluster::OperationFlags::FastTrace;
    uint32_t                           clasPositionTruncateBits = 0;

    // Strip vertex positions from resident group blobs at upload time: the hit
    // shader fetches them from the AS and the CLAS build consumes a transient
    // upload-ring copy, so the geometry pool drops 12 of ~24 bytes/vertex.
    // Forced off at init when the device lacks RayTracingPositionFetch.
    bool stripResidentPositions = true;

    // Also strip the packed normal+tangent words (another 4 B/vertex) when the
    // renderer's cluster-LoD Vertex Normals toggle is off — the stripped cluster
    // headers drop ClusterAttribute::VertexNormal/VertexTangent and the hit
    // shader falls back to facet shading.  Independent of stripResidentPositions.
    bool stripResidentNormals = false;

    // For the persistent allocator; both are compile-time tuning only — the UI
    // Memory tab reads them for display and nothing writes them.
    uint32_t clasAllocatorSectorSizeShift = 10;
    // granularity of allocator in multiples of clas alignment
    uint32_t clasAllocatorGranularityShift = 0;
};

//////////////////////////////////////////////////////////////////////////
//
// StreamingRequests
//
// Requests from the device to be handled by the streaming manager.
// Device fills it, host reacts by updating.
//
// This mostly provides the storage space for the geometry groups to be
// loaded and unloaded, both on device and the host copy.  The load/unload
// arrays live in a per-task ring that shaders index with taskSlotStride.

class StreamingRequests
{
public:
    struct TaskInfo
    {
        const shaderio::StreamingFrameRequest* shaderData;
        const GeometryGroup*                   loadGeometryGroups;
        const GeometryGroup*                   unloadGeometryGroups;
    };

    void Init(nvrhi::IDevice* device, const StreamingConfig& config,
              uint32_t groupCountAlignment, uint32_t clusterCountAlignment);
    void Deinit();

    size_t GetOperationsSize() const;

    // for updates:
    // within same frame
    // first prepare request
    void ApplyTask(shaderio::StreamingFrameRequest& shaderData,
                   uint32_t                          taskIndex,
                   uint32_t                          frameIndex);

    // then trigger readback for that request.
    // srcBuffer + offset locate the live shaderio::StreamingFrameRequest header
    // (typically the scene-wide CBV owned by ClusterLodStreaming).
    void DownloadFrameRequests(nvrhi::ICommandList*                   commandList,
                               const shaderio::StreamingFrameRequest& shaderData,
                               nvrhi::IBuffer*                        srcBuffer,
                               uint64_t                               srcBufferOffset);

    // later frame, get results when cmd update completed
    const TaskInfo& GetCompletedTask(uint32_t taskIndex) const { return m_taskInfos[taskIndex]; }

    // Request ring accessors.  Load groups start at the slot base; unload groups
    // start at m_unloadGroupsOffset within each slot.
    nvrhi::IBuffer* GetRequestBuffer()      const { return m_requestBuffer.GetBuffer(); }
    uint64_t        GetRequestSlotSize()    const { return m_requestSize; }

private:
    nvrhi::IDevice*       m_device = nullptr;
    RTXMGBuffer<uint8_t>  m_requestBuffer;       // device UAV + transfer-src
    RTXMGBuffer<uint8_t>  m_requestHostBuffer;   // cpuAccess = Read, persistently mapped
    void*                 m_hostMapping = nullptr;

    uint64_t m_requestSize        = 0;  // bytes per slot (loadGroups + unloadGroups)
    uint64_t m_shaderDataOffset   = 0;  // byte offset of the per-task header-snapshot array
    uint64_t m_loadGroupsOffset   = 0;  // within-slot offset
    uint64_t m_unloadGroupsOffset = 0;  // within-slot offset

    shaderio::StreamingFrameRequest m_shaderData = {};
    TaskInfo                        m_taskInfos[kStreamingMaxActiveTasks];
};

//////////////////////////////////////////////////////////////////////////
//
// StreamingResident
//
// This class holds the persistent table of resident groups and clusters.
// Each group is assigned a groupResidentID, an immutable index in the group table,
// as well as a range of indices starting at clusterResidentID for the cluster table.
//
// The active groups — the ones that can be loaded and unloaded — are a tightly
// packed array of groupResidentIDs so shaders can easily iterate them.  The
// always-resident lowest-detail groups sit at its head as a pinned prefix
// (see `UploadInitialState`); the shader-visible counts skip that prefix.
//
// Loads append at the end of the list; an unload pops the last element into the
// freed spot.  As small optimization, we only upload the range of indices that
// has changed, not the entire list.
//
// The object also contains the clas buffer in which resident clusters are stored.
// `initClas`/`deinitClas` are separated out, so that a pure rasterization renderer
// can avoid extra memory cost.

class StreamingResident
{
public:
    struct Group
    {
        GeometryGroup       geometryGroup;
        uint32_t            activeIndex;
        uint32_t            groupResidentID : 24;
        uint32_t            lodLevel : 8;
        uint32_t            clusterResidentID;
        uint16_t                   clusterCount;
        uint16_t                   triangleCount;
        uint64_t                   deviceAddress;
        shaderio::GroupAddress     groupAddress;
        rtxmg::BufferSubAllocation storageHandle;  // host-only: carries bytes + allocator metadata
    };

    // GPU resident-table buffer setup
    void Init(nvrhi::IDevice* device, const StreamingConfig& config,
              uint32_t groupCountAlignment, uint32_t clusterCountAlignment);
    void Deinit();
    void Reset(shaderio::StreamingResident& shaderData);

    // CLAS pool buffer setup.  Separated from Init() so a pure rasterization
    // renderer can skip the CLAS-storage cost.
    void InitClas(nvrhi::IDevice*              device,
                  const StreamingConfig&       config,
                  shaderio::StreamingResident& outShaderData);
    void DeinitClas();

    size_t GetOperationsSize() const;
    size_t GetClasOperationsSize() const;

    // The residency slice of StreamingStats, owned rather than written into the
    // caller's aggregate: the composition happens in ClusterLodStreaming::GetStats.
    struct ResidencyStats
    {
        uint32_t residentGroups      = 0;
        uint32_t residentClusters    = 0;
        uint32_t residentTriangles   = 0;
        uint32_t persistentGroups    = 0;
        uint32_t persistentClusters  = 0;
        uint32_t persistentTriangles = 0;
    };
    ResidencyStats GetStats() const;

    // Per-buffer accessors.  Bound as RWStructuredBuffer<T> UAVs, or
    // StructuredBuffer<T> SRVs where hit paths read them.
    const RTXMGBuffer<shaderio::StreamingGroup>& GetGroupsBuffer() const { return m_groupsBuffer; }
    nvrhi::IBuffer* GetGroupIDsBuffer()       const { return m_groupIDsBuffer.GetBuffer(); }
    const RTXMGBuffer<shaderio::ClusterAddress>& GetClustersBuffer() const { return m_clustersBuffer; }
    nvrhi::IBuffer* GetActiveGroupsBuffer()   const { return m_activeGroupsBuffer.GetBuffer(); }

    // CLAS pool accessors.
    const RTXMGBuffer<uint64_t>& GetClasAddressesBuffer() const { return m_clasAddressesBuffer; }
    nvrhi::IBuffer* GetClasSizesBuffer()      const { return m_clasSizesBuffer.GetBuffer(); }
    nvrhi::IBuffer* GetGroupClasSizesBuffer() const { return m_groupClasSizesBuffer.GetBuffer(); }
    nvrhi::IBuffer* GetClasDataBuffer()       const { return m_clasDataBuffer.GetBuffer(); }
    // GPU-persistent CLAS-allocator scalars: the host zero-inits this once and
    // never re-uploads it, so only stream_dispatch_setup maintains its contents.
    nvrhi::IBuffer* GetResidentPersistentBuffer() const { return m_residentPersistentBuffer.GetBuffer(); }

    const Group* FindGroup(GeometryGroup geometryGroup) const;
    const Group& GetGroup(uint32_t groupResidentID) const { return m_groups[groupResidentID]; }
    // Indexed by groupResidentID; the persistent low-detail groups are added
    // first, so they occupy [0, GetLowDetailGroupsCount()).
    const std::vector<Group>& GetGroups() const { return m_groups; }
    // Appends in active-list order, so the pinned low-detail groups land in
    // out[0, GetLowDetailGroupsCount()) — the Inspector's pinned-vs-streamable
    // split is positional.
    void CollectActiveGeometryGroups(std::vector<GeometryGroup>& out) const;

    // Monotonic change counter of the resident set.  The Inspector re-runs its
    // O(resident) aggregation only when this moves.
    uint64_t GetResidencyEpoch() const { return m_residencyEpoch; }

    // Aggregates resident CLAS bytes per (geometryID, lodLevel) into
    // out[geometryID][lodLevel] for the Inspector.  Leaves `out` unchanged until
    // the 1-frame async readback lands, and when the CLAS-sizes buffer doesn't
    // exist (the compaction allocator leaves it null).
    void DownloadResidentClasBytes(nvrhi::ICommandList* commandList,
                                   uint32_t granularityByteShift,
                                   std::vector<std::array<uint64_t, shaderio::kMaxLodLevels>>& out);

    // for updates:
    uint32_t GetLoadActiveGroupsOffset() const;
    uint32_t GetLoadActiveClustersOffset() const;

    // Length of the always-resident persistent prefix in m_activeGroupsBuffer.
    // Uploaded as StreamingResident::persistentGroupsCount: the buffer is bound
    // whole and shaders add this base to reach the streamable suffix.
    uint32_t GetLowDetailGroupsCount() const { return m_lowDetailGroupsCount; }
    uint32_t GetLowDetailClustersCount() const { return m_lowDetailClustersCount; }
    uint32_t GetLowDetailMaxGroupClusters() const { return m_lowDetailMaxGroupClusters; }

    // first handle adding & removing
    bool   CanAllocateGroup(uint32_t numClusters) const;
    Group* AddGroup(GeometryGroup geometryGroup, uint32_t clusterCount, uint32_t triangleCount);
    void   RemoveGroup(uint32_t groupResidentID);

    // Run once, after every persistent low-detail group has been added: captures
    // the persistent-prefix counts and uploads their table entries.
    void UploadInitialState(nvrhi::ICommandList*         commandList,
                            shaderio::StreamingResident& outShaderData);

    // Per-frame three-stage update sequence (one per task slot):
    //   1) UploadActiveGroupsDelta: memcpy delta-changes to host staging, then
    //      GPU-copy to m_activeUpdateBuffer. Returns transferred bytes.
    //   2) ApplyTask:               copy task's CBV snapshot to caller's shaderData.
    //   3) CommitActiveGroupsDelta: final GPU copy from m_activeUpdateBuffer to
    //      m_activeGroupsBuffer at the host-visible update-range offset.
    size_t UploadActiveGroupsDelta(nvrhi::ICommandList* commandList, uint32_t taskIndex);
    void   ApplyTask(shaderio::StreamingResident& shaderData, uint32_t taskIndex, uint32_t frameIndex);
    void   CommitActiveGroupsDelta(nvrhi::ICommandList* commandList, uint32_t taskIndex);

private:
    // tracks range of indices that were manipulated since last update, triggered by unloads.
    struct UpdateRange
    {
        uint32_t lo = uint32_t(~0);
        uint32_t hi = 0;

        void Update(uint32_t index)
        {
            lo = std::min(lo, index);
            hi = std::max(hi, index);
        }

        uint32_t Count() const { return hi == 0 && lo == ~0u ? 0 : 1 + hi - lo; }
    };

    // uint64_t is GeometryGroup::key
    std::unordered_map<uint64_t, uint32_t> m_mapGeometryGroup2Residency;

    rtxmg::IDPool m_groupAllocator;
    rtxmg::IDPool m_clusterAllocator;

    uint32_t m_maxClusters = 0;
    uint32_t m_maxGroups   = 0;

    std::vector<Group> m_groups;

    // index into above
    std::vector<uint32_t> m_activeGroupIndices;

    uint32_t m_lowDetailGroupsCount      = 0;
    uint32_t m_lowDetailClustersCount    = 0;
    uint32_t m_lowDetailTrianglesCount   = 0;
    uint32_t m_lowDetailMaxGroupClusters = 0;

    uint32_t m_activeGroupsCount    = 0;
    uint32_t m_activeClustersCount  = 0;
    uint32_t m_activeTrianglesCount = 0;

    uint64_t m_residencyEpoch       = 0;  // see GetResidencyEpoch()

    shaderio::StreamingResident m_shaderData = {};
    UpdateRange                 m_groupIndicesUpdateRange;

    // Resident-table buffers (one typed RTXMGBuffer per region).
    RTXMGBuffer<shaderio::StreamingGroup> m_groupsBuffer;          // size = maxGroups
    RTXMGBuffer<uint32_t>                 m_groupIDsBuffer;        // size = maxGroups
    RTXMGBuffer<shaderio::ClusterAddress> m_clustersBuffer;        // size = maxClusters
    RTXMGBuffer<uint32_t>                 m_activeGroupsBuffer;    // size = maxGroups
    RTXMGBuffer<uint32_t>                 m_activeUpdateBuffer;    // size = maxGroups * kStreamingMaxActiveTasks
    RTXMGBuffer<uint32_t>                 m_activeHostBuffer;      // cpuAccess = Write, size = maxGroups * kStreamingMaxActiveTasks

    // CLAS pool buffers (typed metadata buffers; ByteAddressBuffer for payload).
    // shaderio::StreamingFrameRequest carries the per-frame readback mirror of
    // the allocator scalars, m_residentPersistentBuffer the canonical GPU copy.
    RTXMGBuffer<uint64_t>                 m_clasAddressesBuffer;   // size = maxClusters
    RTXMGBuffer<uint32_t>                 m_clasSizesBuffer;       // size = maxClusters
    // One uint2 (allocSize, wastedByteSize) per group, indexed by groupResidentID.
    RTXMGBuffer<donut::math::uint2>       m_groupClasSizesBuffer;  // size = maxGroups, allocator-only.
                                                                   // 8-byte stride; HLSL view is RWStructuredBuffer<uint2>.
    RTXMGBuffer<uint8_t>                  m_clasDataBuffer;        // size = m_maxClasBytes, AS storage
    uint64_t                              m_maxClasBytes = 0;
    // Single-element GPU-persistent allocator scalars (host zero-inits only).
    RTXMGBuffer<shaderio::StreamingResidentPersistent> m_residentPersistentBuffer;

    // Per-frame upload task ring; shaderData is the CBV snapshot the task will
    // upload on apply.
    struct TaskInfo
    {
        struct BufferCopy
        {
            uint64_t srcOffset = 0;  // within m_activeUpdateBuffer
            uint64_t dstOffset = 0;  // within m_activeGroupsBuffer
            uint64_t size      = 0;
        };
        BufferCopy                  region;
        shaderio::StreamingResident shaderData = {};
    };
    TaskInfo m_taskInfos[kStreamingMaxActiveTasks];
};

//////////////////////////////////////////////////////////////////////////
//
// StreamingAllocator
//
// This class implements a persistent allocator on the GPU that
// allows to do allocation management in shaders.
//
// The compute kernels scan the memory for free gaps and make
// them available as list for different gap sizes up to
// `maxAllocationByteSize`. The memory's use is encoded in
// a giant bitfield where each bit represents `granularityByteSize`
// bytes.
//
// We use it in the sample to manage clas memory, as the host
// doesn't know the size of the clas after building and we want
// to avoid detailed readbacks and host to be involved in the
// allocation process.

class StreamingAllocator
{
public:
    void Init(nvrhi::IDevice*               device,
              size_t                        totalMegaBytes,
              uint32_t                      maxAllocationByteSize,
              uint32_t                      granularityByteSize,
              uint32_t                      sectorSizeShift,
              shaderio::StreamingAllocator& shaderData);
    void Deinit();

    size_t   GetOperationsSize() const;
    uint32_t GetMaxSized() const;

    // ClearManagementBuffer zeroes every allocator sub-array (a full reset);
    // ClearFreeSizeRanges runs each frame, before build_freegaps refills them.
    void ClearManagementBuffer(nvrhi::ICommandList* cmd);
    void ClearFreeSizeRanges(nvrhi::ICommandList* cmd);

    // The 7 allocator sub-arrays live contiguously inside m_managementBuffer (a
    // single RWByteAddressBuffer), addressed by the per-region byte offsets in
    // GetShaderData().
    nvrhi::IBuffer* GetManagementBuffer() const { return m_managementBuffer.GetBuffer().Get(); }
    RTXMGBuffer<uint8_t>& GetManagementBufferTyped() { return m_managementBuffer; }

    // The caller copies this into its aggregate's .clasAllocator field, which it
    // uploads wholesale each frame; shaders read streamingRW[0].clasAllocator.
    const shaderio::StreamingAllocator& GetShaderData() const { return m_shaderData; }

private:
    shaderio::StreamingAllocator m_shaderData = {};

    RTXMGBuffer<uint8_t>                      m_managementBuffer;

    // Zero blob sized to the freeSizeRanges sub-region, written each frame to
    // clear the per-class counters build_freegaps refills: nvrhi's
    // clearBufferUInt is whole-buffer only and this is a sub-range.
    std::vector<uint8_t> m_freeSizeRangesZeroBlob;
};

//////////////////////////////////////////////////////////////////////////
//
// StreamingUpdates
//
// Provides storage for update tasks that modify the per-geometry group pointers
// and are performed on the device. These patches reflect changes made to
// the `StreamingResident` table.
//
// Furthermore we might need extra information when building new clas as part
// of loading new groups within an update task.
//
// We also track how much groups & clusters were scheduled for loading, this helps us
// estimate the clas memory space we have left when handling a new request to
// load new groups. Cause at that point in time we have to estimate using
// the worst-case size for all those "yet to be built clas".
//
// Note:
// Giving back the memory of unloading tasks must be delayed until after
// an update has completed on the device, otherwise we risk taking memory
// from a frame that was scheduled in the past of the host but might still
// be executing on the device.

class StreamingUpdates
{
public:
    struct TaskInfo
    {
        uint32_t                          loadCount;
        uint32_t                          unloadCount;
        uint32_t                          newClusterCount;
        uint32_t                          loadActiveGroupsOffset;
        uint32_t                          loadActiveClustersOffset;
        uint32_t                          geometryCachedCount;
        uint32_t                          geometryCachedClustersCount;
        shaderio::StreamingPatch*         loadPatches;
        shaderio::StreamingPatch*         unloadPatches;
        // Host-only handles carrying block location + size + allocator metadata,
        // so completing an update can free them without a parallel size array.
        rtxmg::BufferSubAllocation*       unloadHandles;
        shaderio::StreamingGeometryPatch* geometryPatches;
    };

    struct NewInfo
    {
        uint32_t groups   = 0;
        uint32_t clusters = 0;
    };

    void Init(nvrhi::IDevice*        device,
              const StreamingConfig& config,
              uint32_t               geometryCount,
              uint32_t               groupCountAlignment,
              uint32_t               clusterCountAlignment);
    void InitClas(nvrhi::IDevice*        device,
                  const StreamingConfig& config,
                  const BakerConfig&     bakerConfig,
                  uint32_t               maxClusterTriangles);
    void DeinitClas();
    void Deinit();

    size_t   GetOperationsSize() const;
    size_t   GetClasOperationsSize() const;
    uint32_t GetMaxCachedBlasBuilds() const;

    void Reset();

    // Groups/clusters already scheduled to land after frameIndex, i.e. the
    // worst-case CLAS space a new load request must still leave room for.
    NewInfo GetFutureNew(uint64_t frameIndex) const;

    // first run update
    TaskInfo& GetNewTask(uint32_t taskIndex);
    // returns number of bytes transferred
    size_t UploadTaskPatches(nvrhi::ICommandList* cmd, uint32_t taskIndex);
    // then apply if upload completed
    void ApplyTask(shaderio::StreamingUpdate& shaderData, uint32_t taskIndex, uint32_t frameIndex);

    // later frame, must have applied task completed
    const TaskInfo& GetCompletedTask(uint32_t taskIndex) const { return m_taskInfos[taskIndex]; }

    // Accessors required for binding; the per-task byte offset into the patches
    // buffer is exposed by GetPatchesByteOffsetForTask.
    nvrhi::IBuffer* GetPatchesBuffer()                const { return m_patchesBuffer.GetBuffer().Get(); }
    nvrhi::IBuffer* GetNewClasBuildsBuffer()          const { return m_newClasBuildsBuffer.GetBuffer().Get(); }
    nvrhi::IBuffer* GetNewClasAddressesBuffer()       const { return m_newClasAddressesBuffer.GetBuffer().Get(); }
    nvrhi::IBuffer* GetNewClasSizesBuffer()           const { return m_newClasSizesBuffer.GetBuffer().Get(); }
    nvrhi::IBuffer* GetNewClasResidentIDsBuffer()     const { return m_newClasResidentIDsBuffer.GetBuffer().Get(); }
    nvrhi::IBuffer* GetNewClasGeometryIndicesBuffer() const { return m_newClasGeometryIndicesBuffer.GetBuffer().Get(); }
    nvrhi::IBuffer* GetMoveClasDstAddressesBuffer()   const { return m_moveClasDstAddressesBuffer.GetBuffer().Get(); }
    nvrhi::IBuffer* GetMoveClasSrcAddressesBuffer()   const { return m_moveClasSrcAddressesBuffer.GetBuffer().Get(); }

    RTXMGBuffer<uint64_t>& GetNewClasAddressesBufferTyped() { return m_newClasAddressesBuffer; }
    RTXMGBuffer<uint64_t>& GetMoveClasDstAddressesBufferTyped() { return m_moveClasDstAddressesBuffer; }
    RTXMGBuffer<uint64_t>& GetMoveClasSrcAddressesBufferTyped() { return m_moveClasSrcAddressesBuffer; }

    uint64_t GetPatchesByteOffsetForTask(uint32_t taskIndex) const;

    // BLAS caching: 16B-stride StreamingGeometryPatch ring, which needs its own
    // buffer because a 16B view can't alias the 32B StreamingPatch ring.  Null
    // when caching is not configured.
    nvrhi::IBuffer* GetGeometryPatchesBuffer() const { return m_geometryPatchesBuffer.GetBuffer().Get(); }
    uint64_t        GetGeometryPatchesByteOffsetForTask(uint32_t taskIndex) const;

private:
    bool m_useBlasCaching = false;

    RTXMGBuffer<shaderio::StreamingPatch> m_patchesBuffer;
    RTXMGBuffer<shaderio::StreamingPatch> m_patchesHostBuffer;
    void*                                 m_patchesHostMapping = nullptr;

    // One slot of m_geometryPatchesPerTask entries per task; allocated only when
    // caching is configured.
    RTXMGBuffer<shaderio::StreamingGeometryPatch> m_geometryPatchesBuffer;
    RTXMGBuffer<shaderio::StreamingGeometryPatch> m_geometryPatchesHostBuffer;
    void*                                         m_geometryPatchesHostMapping = nullptr;
    uint32_t                                      m_geometryPatchesPerTask     = 0;

    std::vector<rtxmg::BufferSubAllocation> m_unloadHandles;
    TaskInfo                        m_taskInfos[kStreamingMaxActiveTasks] = {};

    shaderio::StreamingUpdate m_shaderData = {};

    uint32_t m_clusterCountAlignment     = 0;
    uint32_t m_scheduleIndex             = 0;
    NewInfo  m_pendingNew                = {};
    NewInfo  m_scheduledNew[kStreamingMaxActiveTasks]      = {};
    uint64_t m_scheduledNewFrame[kStreamingMaxActiveTasks] = {};

    // Per-region typed CLAS build/move arrays, each bound at its own slot.
    RTXMGBuffer<shaderio::ClasBuildInfo> m_newClasBuildsBuffer;
    RTXMGBuffer<uint64_t>                m_newClasAddressesBuffer;
    ClusterOperationSizesBuffer          m_newClasSizesBuffer;
    RTXMGBuffer<uint32_t>                m_newClasResidentIDsBuffer;
    RTXMGBuffer<uint32_t>                m_newClasGeometryIndicesBuffer;
    RTXMGBuffer<uint64_t>                m_moveClasDstAddressesBuffer;
    RTXMGBuffer<uint64_t>                m_moveClasSrcAddressesBuffer;

    nvrhi::IDevice* m_device = nullptr;
};

//////////////////////////////////////////////////////////////////////////
//
// StreamingStorage
//
// Storage contains the geometric data for the active resident groups
// (persistent resident groups are stored in the `ClusterLodStreaming` class directly).
// It also contains scratch space to handle the uploads from host to device.
// The resident group has an immutable device memory location over its lifetime.
//
// rtxmg::BufferSubAllocator owns the block buffers, descriptor handles and
// OffsetAllocator bookkeeping (see rtxmg/utils/buffer_sub_allocator.h).

class StreamingStorage
{
public:
    struct TaskInfo
    {
        size_t usedMemory;
        size_t baseOffset;
        size_t regionCount;
    };

    void Init(nvrhi::IDevice*                        device,
              donut::engine::DescriptorTableManager* descriptorTable,
              const StreamingConfig&                 config);
    void Deinit();
    void Reset();

    // freeing is not done during regular transfer tasks.
    void Free(rtxmg::BufferSubAllocation& handle);

    // The geometry-pool slice of StreamingStats; see StreamingResident::ResidencyStats.
    struct PoolStats
    {
        uint64_t reservedDataBytes  = 0;
        uint64_t usedDataBytes      = 0;
        uint64_t allocatedDataBytes = 0;
    };
    PoolStats GetStats() const;
    size_t GetOperationsSize() const;
    size_t GetMaxDataSize() const;

    // for transfer task:
    // first get operation
    TaskInfo& GetNewTask(uint32_t taskIndex);
    // first test if space is available
    bool CanTransfer(const TaskInfo& operation, size_t size) const;
    // then allocate. Returns false on OOM; deviceAddress receives the GPU
    // virtual address of the allocation.  `group` is unused.
    bool Allocate(rtxmg::BufferSubAllocation& handle, GeometryGroup group, size_t sz, uint64_t& deviceAddress);
    // and get transfer space — returns a CPU pointer into the upload ring
    // for the caller to memcpy bytes into; the matching device-side copy
    // is recorded for UploadPendingTransfers to issue.
    void* AppendTransfer(TaskInfo& operation, const rtxmg::BufferSubAllocation& dstHandle, size_t bytes);
    // Allocate ring space WITHOUT a pool copy — for transient data the GPU reads
    // straight from the ring by GPU VA (returned in gpuVA), e.g. the CLAS-build
    // position staging of position-stripped blobs.  Lives until the task's fence.
    void* AppendHostRead(TaskInfo& operation, size_t bytes, uint64_t& gpuVA);
    // at end of updates trigger cmd update; returns number of copyBuffer
    // operations issued (one per CopyRegion).
    uint32_t UploadPendingTransfers(nvrhi::ICommandList* commandList);

    const rtxmg::BufferSubAllocator& GetAllocator() const { return m_dataAllocator; }
    rtxmg::BufferSubAllocator&       GetAllocator()       { return m_dataAllocator; }

private:
    // Per-target-buffer batch of CopyRegions; UploadPendingTransfers issues one
    // copyBuffer call per region (nvrhi has no multi-region copyBuffer).
    struct CopyInfo
    {
        nvrhi::IBuffer* targetBuffer;
        size_t          regionOffset;
        size_t          regionCount;
    };

    struct CopyRegion
    {
        uint64_t srcOffset;
        uint64_t dstOffset;
        uint64_t size;
    };

    nvrhi::IDevice*                        m_device          = nullptr;
    donut::engine::DescriptorTableManager* m_descriptorTable = nullptr;

    size_t m_maxSceneBytes    = 0;
    size_t m_maxTransferBytes = 0;
    size_t m_blockBytes       = 0;

    rtxmg::BufferSubAllocator m_dataAllocator;

    RTXMGBuffer<uint8_t>      m_transferHostBuffer;  // CPU-mapped upload ring
    void*                     m_transferHostMapping = nullptr;

    std::vector<CopyInfo>     m_copyInfos;
    std::vector<CopyRegion>   m_copyRegions;

    TaskInfo m_taskOperations[kStreamingMaxActiveTasks];
};

}  // namespace rtxmg
