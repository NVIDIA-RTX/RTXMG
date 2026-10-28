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

/*

  Shader Description
  ==================

  This compute shader handles updating the scene.
  Previous requests to load/unload have been completed and
  are provided for patching the scene.

  Effectively we are manipulating each geometry's `streamingGroupAddresses`
  entry for the loaded/unloaded group, publishing a valid blob address on load
  or a sentinel on unload.

  Furthermore when ray tracing is required we prepare building
  new CLAS for the loaded groups' clusters.  The CLAS allocator
  (`stream_allocator_alloc_groups.hlsl`, or `stream_compact_append_new.hlsl`
  in the compaction scheme) then sets up the move from the temporary build
  destinations to the final ones.

  A thread represents a single patch operation, which takes care of
  one group.

  D3D12 bindings:
  ---------------

  * Per-buffer state is read via indexed access on discretely-bound buffers at
    fixed register slots, part of the unified m_streamingBindingLayout in
    [streaming.cpp] (shared with allocator family + stream_age_groups):
        t0  t_StreamingPatches       sub-range of m_updates patches
        t3  t_GeometryPatches        per-geom cached-BLAS patch ring
        u15 u_Geometries             shader-side per-geom Geometry (UAV:
                                     cached-BLAS fields written here)
        u0  u_ResidentGroups         scene-global StreamingGroup table
        u7  streamingRW              SceneStreaming aggregate (atomics)
        u8  u_ActiveGroups           whole buffer; shader adds
                                     persistentGroupsCount (shared with agefilter)
        u9  u_GroupIDs               scene-global resident->orig groupID
                                     (shared with agefilter)
        u10 u_ResidentClusters       per-cluster ClusterAddress UAV
        u11 u_NewClasBuilds          IndirectTriangleClasArgs[] UAV
    Per-frame scalars (patchUnloadGroupsCount / patchGroupsCount /
    loadActiveGroupsOffset) come from streamingRW[0].update.X — no
    per-dispatch CBV.

  * The per-geom residency write goes to a per-geom bindless
    RWStructuredBuffer<GroupAddress> at `geom.streamingGroupAddressesUAV`.
    traversal_run.hlsl gates residency on
    `GroupAddress.srvIndex != shaderio::kStreamingInvalidSrvIndex`; while invalid,
    `GroupAddress.byteOffset` stores the last requested frame for dedupe.

  * Each cluster's residency is published as a
    ClusterAddress{srvIndex, byteOffset} write into u_ResidentClusters.
    byteOffset points at the Cluster header; the hit shader derives
    vertex/index byte bases from the header offsets.

  * The CLAS-build inner loop writes per-cluster
    `donut::cluster::IndirectTriangleClasArgs` into a discretely-bound
    UAV `u_NewClasBuilds` (register u11), which the host binds at
    `m_clasIndirectArgsBuffer`.  No CPU-side fill — the GPU is the only
    writer.  The follow-up CLAS Implicit build in ApplyResidencyUpdate reads
    from the same buffer.

  * Counter updates use InterlockedAdd (out-param form).  16-bit fields on
    StreamingPatch / StreamingGroup are native uint16_t (SM6.2+ 16-bit types
    enabled by ShaderMake), no bit-shift unpack.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "feature_gates.hlsli"
#include "rtxmg/cluster_lod/shaders/cluster_lod_material_resolve.hlsli"
#include <nvrhi/nvrhiHLSL.h>
#include <donut/shaders/cluster_args.hlsli>

////////////////////////////////////////////

// Unified streaming binding layout — see streaming.cpp's
// InitShadersAndPipelines comment for the full slot map.  Allocator-family
// and agefilter shaders share this layout; slots not consumed here are
// bound to either real buffers (when the persistent allocator family is
// live) or dummy buffers (in the compaction path).
StructuredBuffer<shaderio::StreamingPatch>                     t_StreamingPatches       : register(t0);
// BLAS caching: per-geom cached-BLAS patch ring, written by the host's
// HandleBlasCaching and published into u_Geometries below.  Geometries is bound
// as a UAV so it can be read and written through one binding (no state conflict).
StructuredBuffer<shaderio::StreamingGeometryPatch>             t_GeometryPatches        : register(t3);
RWStructuredBuffer<shaderio::Geometry>                         u_Geometries             : register(u15);
RWStructuredBuffer<shaderio::StreamingGroup>                   u_ResidentGroups         : register(u0);
RWStructuredBuffer<shaderio::SceneStreaming>                   streamingRW              : register(u7);
RWStructuredBuffer<uint>                                       u_ActiveGroups           : register(u8);
RWStructuredBuffer<uint>                                       u_GroupIDs               : register(u9);
RWStructuredBuffer<shaderio::ClusterAddress>                   u_ResidentClusters       : register(u10);
RWStructuredBuffer<donut::cluster::IndirectTriangleClasArgs> u_NewClasBuilds        : register(u11);
// Scene-wide mixed-cluster geometry-indices task queue; per-task slot =
// sceneMaxClusterTriangles entries (one per triangle of the resulting CLAS).
// The first two entries of a slot temporarily carry the cluster reference for
// stream_fill_clas_geometry_indices, which then overwrites the whole slot with
// the real per-triangle entries — see the append site below.
// Packed uint32 (low 24 = geometryIndex, high 3 = ClusterGeometryFlags), the
// same byte layout as GeometryIndexAndFlags but written as plain uint: DXC's
// bit-field-struct stores left raw material bytes in the buffer.
RWStructuredBuffer<uint>                                       u_NewClasGeometryIndices : register(u13);
// Per-frame map from CLAS-build argIdx → scene-global clusterResidentID,
// consumed by stream_compact_append_new.hlsl.
RWStructuredBuffer<uint>                                       u_NewClasResidentIDs     : register(u14);

////////////////////////////////////////////

[numthreads(shaderio::kStreamUpdateSceneThreads, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint threadID = gid.x;

    const uint patchUnloadGroupsCount = streamingRW[0].update.patchUnloadGroupsCount;
    const uint patchGroupsCount       = streamingRW[0].update.patchGroupsCount;
    const uint loadActiveGroupsOffset = streamingRW[0].update.loadActiveGroupsOffset;

    // BLAS caching: publish each (re)built/invalidated cached BLAS into the
    // shader-visible Geometry.  HandleBlasCaching emits at most one patch per
    // geometry, so each Geometry is written by exactly one thread and the
    // bitfield RMW on cachedBlasLodLevel is race-free.  This runs BEFORE the
    // patchGroupsCount bail: the grid is sized to max(patchGroupsCount,
    // patchCachedBlasCount), so a cached patch can land on a lane past it.
    if (threadID < streamingRW[0].update.patchCachedBlasCount)
    {
        shaderio::StreamingGeometryPatch sgpatch = t_GeometryPatches[threadID];
        uint geometryID = sgpatch.geometryID;
        u_Geometries[geometryID].cachedBlasLodLevel  = sgpatch.cachedBlasLodLevel & 0xFFu;
        u_Geometries[geometryID].cachedBlasAddress   = sgpatch.cachedBlasAddress;
        // render-stats: cached level triangle / cluster counts (0 on invalidate)
        u_Geometries[geometryID].cachedBlasTriangles = sgpatch.cachedBlasTriangles;
        u_Geometries[geometryID].cachedBlasClusters  = sgpatch.cachedBlasClustersCount;
    }

    // The grid is rounded up to a kStreamUpdateSceneThreads multiple, so
    // padding lanes have no patch.  Bail before touching t_StreamingPatches:
    // it is bound as a tight sub-range of exactly patchGroupsCount entries, so
    // a pre-emptive read on a padding lane would be out of bounds.
    if (threadID >= patchGroupsCount)
        return;

    // works for both load and unload — patches are bound as a sub-range that
    // already starts at the task's first patch (host applies the byte offset
    // at binding-set creation), so t_StreamingPatches[threadID] is direct.
    shaderio::StreamingPatch spatch = t_StreamingPatches[threadID];

    // geometryID varies per thread → NonUniformResourceIndex.
    uint groupAddressUAVSlot = u_Geometries[spatch.geometryID].streamingGroupAddressesUAV;
    RWStructuredBuffer<shaderio::GroupAddress> groupAddresses =
        ResourceDescriptorHeap[NonUniformResourceIndex(groupAddressUAVSlot)];

    if (threadID < patchUnloadGroupsCount)
    {
        // ---- UNLOAD branch ------------------------------------------
        // Write the not-resident sentinel. The second component is the
        // request-dedupe frame and starts at 0 until traversal requests it.
        shaderio::GroupAddress deadAddress;
        deadAddress.srvIndex   = shaderio::kStreamingInvalidSrvIndex;
        deadAddress.byteOffset = 0u;
        groupAddresses[spatch.groupID] = deadAddress;
    }
    else
    {
        // ---- LOAD branch --------------------------------------------
        uint loadGroupIndex = threadID - patchUnloadGroupsCount;

        // Read the freshly streamed Group blob via the per-block bindless
        // ByteAddressBuffer at spatch.groupAddress.srvIndex, indexed by
        // GroupAddress{srvIndex, byteOffset}.
        ByteAddressBuffer groupBlob =
            ResourceDescriptorHeap[NonUniformResourceIndex(spatch.groupAddress.srvIndex)];
        // The CPU upload path has already patched the resident IDs into
        // the group blob. Publish the bindless blob address for traversal.
        shaderio::Group group = groupBlob.Load<shaderio::Group>(spatch.groupAddress.byteOffset);
        uint groupResidentID  = group.groupResidentID;
        shaderio::GroupAddress groupAddress;
        groupAddress = spatch.groupAddress;
        groupAddresses[spatch.groupID] = groupAddress;

        shaderio::StreamingGroup residentGroup;
        residentGroup.geometryID    = spatch.geometryID;
        residentGroup.lodLevel      = uint16_t(group.lodLevel);
        residentGroup.age           = uint16_t(0);
        residentGroup.groupAddress  = spatch.groupAddress;

        // update description in residency table
        u_ResidentGroups[groupResidentID] = residentGroup;

        // retain original groupID, used for unloading
        u_GroupIDs[groupResidentID] = spatch.groupID;

        // insert ourselves into the list of all active groups
        // (u_ActiveGroups is bound WHOLE — Vulkan storage-buffer descriptor
        // offsets must be 16B-aligned — so the persistent low-detail prefix
        // skip is applied in-shader via resident.persistentGroupsCount)
        u_ActiveGroups[streamingRW[0].resident.persistentGroupsCount
                       + loadActiveGroupsOffset + loadGroupIndex] = groupResidentID;

        // We might have a bit of divergence here, but shouldn't be a
        // mission critical issue.
        //
        // The loop below publishes one ClusterAddress per cluster into the
        // scene-global u_ResidentClusters table (the hit shader loads the
        // Cluster header from it and derives the vertex/index payload offsets),
        // and the per-cluster vertex/index GVAs the CLAS-build hardware needs.

        uint newBuildOffset = spatch.clasBuildOffset;
        const uint clusterCount = uint(group.clusterCount);
        // Running byte offset into the group's staged position array (upload
        // ring, spatch.positionsGpuAddress) — clusters back-to-back in header
        // order, fp32 xyz per vertex.  Only consumed when the resident blob
        // was uploaded position-stripped.
        uint stagedPosOffset = 0;
        for (uint c = 0; c < clusterCount; c++)
        {
            uint clusterResidentID = group.clusterResidentID + c;

            // Per-cluster header byte offset inside the group blob.
            const uint clusterHdrOff =
                uint(sizeof(shaderio::Group)) +
                uint(sizeof(shaderio::Cluster)) * c;
            shaderio::Cluster cluster =
                groupBlob.Load<shaderio::Cluster>(spatch.groupAddress.byteOffset + clusterHdrOff);

            // Write ClusterAddress for the hit-shader's payload reads.
            shaderio::ClusterAddress clusterAddress;
            clusterAddress.srvIndex   = spatch.groupAddress.srvIndex;
            clusterAddress.byteOffset = spatch.groupAddress.byteOffset + clusterHdrOff;
            u_ResidentClusters[clusterResidentID] = clusterAddress;

            // Per-cluster IndirectTriangleClasArgs for ApplyResidencyUpdate's
            // Implicit-destinations CLAS build.
            donut::cluster::IndirectTriangleClasArgs arg;
            arg.clusterId                         = clusterResidentID;
            arg.clusterFlags                      = 0;
            arg.triangleCount                     = uint16_t(uint(cluster.triangleCountMinusOne) + 1u);
            arg.vertexCount                       = uint16_t(uint(cluster.vertexCountMinusOne)   + 1u);
            arg.positionTruncateBitCount          = uint16_t(streamingRW[0].clasPositionTruncateBits);
#if !DONUT_CLUSTER_API_DXR
            // The DXR cluster API takes both index formats per operation instead of per argument.
            arg.indexFormat                       = 1u;  // OperationIndexFormat::IndexFormat8bit
            arg.opacityMicromapIndexFormat        = 0u;
#endif
            arg.baseGeometryIndexAndFlags         = (nvrhi::rt::cluster::GeometryIndexAndFlags)0;
            arg.indexBufferStride                 = uint16_t(1);
            arg.vertexBufferStride                = uint16_t(4 * 3);
            arg.geometryIndexAndFlagsBufferStride = uint16_t(0);
            arg.opacityMicromapIndexBufferStride  = uint16_t(0);
            arg.indexBuffer                       = spatch.groupGpuAddress + clusterHdrOff + cluster.triangles;
            // Stripped resident blob: positions live only in the transient
            // upload-ring range at spatch.positionsGpuAddress (alive until
            // this task's fence — covers the Implicit CLAS build).
            arg.vertexBuffer =
                (cluster.attributeBits & shaderio::ClusterAttribute::StrippedVertexPos)
                    ? spatch.positionsGpuAddress + stagedPosOffset
                    : spatch.groupGpuAddress + clusterHdrOff + cluster.vertices;
            stagedPosOffset += arg.vertexCount * 12u;
            arg.geometryIndexAndFlagsBuffer       = 0;
            arg.opacityMicromapArray              = 0;
            arg.opacityMicromapIndexBuffer        = 0;
#if DONUT_CLUSTER_API_DXR
            arg.opacityMicromapBaseLocation       = 0;
            arg.geometryIndexAndFlagsIndexBuffer  = 0;
            arg.reservedPadding                   = uint16_t(0);
#endif

            // Mixed clusters carry per-triangle alpha/two-sided state and
            // need the CLAS build's geometryIndexAndFlagsBuffer filled by
            // stream_fill_clas_geometry_indices.  Uniform clusters pack the
            // cluster-wide state into baseGeometryIndexAndFlags instead.
            //
            // With materials disabled (--nomat) stateBits reads as 0, which
            // forces every cluster opaque / single-sided / material 0.
            uint effStateBits =
                (streamingRW[0].materialsEnabled != 0u) ? cluster.stateBits : 0u;
#if !HAS_ALPHA_TEST
            // This permutation builds CLAS with maxGeometryIndex 0, so the
            // alpha-mask geometry slot must never be encoded.
            effStateBits &= ~shaderio::ClusterState::AlphaMasked;
#endif
            const bool useGeometryIndices =
                (effStateBits & shaderio::ClusterState::AlphaMaskedMixed) != 0u ||
                (effStateBits & shaderio::ClusterState::TwoSidedMixed)    != 0u;

#if HAS_ALPHA_TEST
            if (useGeometryIndices)
            {
                // Mixed-cluster geometry-indices task append: claim a task slot
                // in u_NewClasGeometryIndices and stash the cluster reference
                // (srvIndex + byteOffset) in its first two uint32s.
                // stream_fill_clas_geometry_indices consumes it and then
                // overwrites the whole slot with one uint32 per triangle — safe
                // because every lane reads the reference before any lane writes.
                uint taskIdx;
                InterlockedAdd(streamingRW[0].update.newClasGeometryIndicesTaskCounter, 1u, taskIdx);

                const uint slotUint32Base =
                    taskIdx * streamingRW[0].update.sceneMaxClusterTriangles;
                u_NewClasGeometryIndices[slotUint32Base + 0u] =
                    spatch.groupAddress.srvIndex;
                u_NewClasGeometryIndices[slotUint32Base + 1u] =
                    spatch.groupAddress.byteOffset + clusterHdrOff;

                // GVA of u_NewClasGeometryIndices is supplied by the
                // host in shaderio::StreamingUpdate (set once by
                // StreamingUpdates::InitClas).
                arg.baseGeometryIndexAndFlags         = (nvrhi::rt::cluster::GeometryIndexAndFlags)0;
                arg.geometryIndexAndFlagsBufferStride =
                    uint16_t(sizeof(nvrhi::rt::cluster::GeometryIndexAndFlags));
                arg.geometryIndexAndFlagsBuffer       =
                    streamingRW[0].update.newClasGeometryIndicesBufferVA
                    + uint64_t(slotUint32Base)
                    * uint64_t(sizeof(nvrhi::rt::cluster::GeometryIndexAndFlags));
            }
            else
#endif  // HAS_ALPHA_TEST
            {
                arg.baseGeometryIndexAndFlags =
                    ClasEncodeBaseGeometryIndexAndFlagsFromState(effStateBits);
            }

            u_NewClasBuilds[newBuildOffset + c] = arg;

            // Lets stream_compact_append_new map newID → clusterResidentID;
            // the persistent-allocator path never reads it.
            u_NewClasResidentIDs[newBuildOffset + c] = clusterResidentID;
        }
    }
}
