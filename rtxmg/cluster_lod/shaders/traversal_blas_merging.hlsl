/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

  Only used under streaming, with BLAS sharing plus merging enabled.
  One thread per resident active group.

  Replaces the standalone stream_age_groups when BLAS merging is on, doing two
  jobs in one dispatch:
    1. the streaming age filter, shared verbatim with that kernel through
       stream_age_groups.hlsli, and
    2. the merged-BLAS cluster gather: for each resident group whose geometry
       elected a merged proxy, append its highest-detail clusters (leaf clusters,
       or clusters whose generating group isn't resident) to the shared render-
       cluster list under the proxy instanceID, so the per-frame BLAS build
       produces ONE merged BLAS per geometry.

  Notes:
  * Shared traversal binding layout (traversal_common.hlsli); the age-filter
    slots u15..u17 exist only for this kernel.
  * The per-geom GroupAddress table + per-block group blob are read via bindless
    ResourceDescriptorHeap (matching traversal_run_groups' residency read).
    Group header (clusterCount / clusterResidentID) + per-cluster generating
    group are loaded from the group blob, not from a buffer-reference.
  * ReserveRenderClusters reserves the wave's contiguous block; the render-cluster
    list is an unordered append so block order is irrelevant.
  * The cached-keep-alive reads the as-built geometry.cachedBlasLodLevel via the
    t_Geometries SRV, and useBlasCaching comes from streamingRW at runtime (no
    compile permutation).
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "feature_gates.hlsli"
#include "rtxmg/cluster_lod/shaders/traversal_common.hlsli"

////////////////////////////////////////////

// The traversal layout reaches geometries through an SRV; the streaming layout
// binds the same table as a UAV.
uint GetCachedBlasLodLevel(uint geometryID)
{
    return t_Geometries[geometryID].cachedBlasLodLevel;
}

#include "stream_age_groups.hlsli"
#include "rtxmg/utils/wave_size.hlsli"

////////////////////////////////////////////

RTXMG_WAVE_SIZE_32
[numthreads(shaderio::kTraversalBlasMergingThreads, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint threadID = gid.x;

    // Pre-emptive load before the count guard: u_ActiveGroups is over-allocated
    // by one thread group past m_maxGroups (see StreamingResident::Init), so the
    // read stays in-bounds for the padding lanes.  It is bound WHOLE (Vulkan
    // storage-buffer descriptor offsets must be 16B-aligned), so the persistent
    // low-detail prefix is skipped in-shader.
    uint residentID = u_ActiveGroups[streamingRW[0].resident.persistentGroupsCount + threadID];
    bool isValid    = threadID < streamingRW[0].resident.activeGroupsCount;
    if (!isValid)
        return;

    uint geometryID = u_ResidentGroups[residentID].geometryID;

    // (1) age filter — identical to stream_age_groups.
    StreamingAgeFilter(residentID, geometryID, streamingRW[0].useBlasCaching != 0u);

    // (2) merged-BLAS gather: only for geometries that elected a merged proxy.
    uint mergedInstanceID = u_GeometryBuildInfos[geometryID].mergedInstanceID;
    if (mergedInstanceID == ~0u)
        return;

    shaderio::Geometry geometry = t_Geometries[geometryID];
    uint groupIndex = u_GroupIDs[residentID];

    // Resolve this resident group's blob via the per-geom GroupAddress table.
    StructuredBuffer<shaderio::GroupAddress> groupAddresses =
        ResourceDescriptorHeap[NonUniformResourceIndex(geometry.streamingGroupAddressesSRV)];
    shaderio::GroupAddress groupAddress = groupAddresses[groupIndex];
    ByteAddressBuffer groupBlob =
        ResourceDescriptorHeap[NonUniformResourceIndex(groupAddress.srvIndex)];
    shaderio::Group group = groupBlob.Load<shaderio::Group>(groupAddress.byteOffset);

    uint clusterGeneratingGroupByteOffset =
        groupAddress.byteOffset +
        uint(sizeof(shaderio::Group)) +
        uint(sizeof(shaderio::Cluster)) * uint(group.clusterCount);

    // Mark each cluster that belongs in the merged (highest-detail) BLAS: a leaf
    // (no generating group) or one whose generating group isn't resident.
    // One bit per cluster across four words, so group.clusterCount must fit in
    // GROUP_CLUSTER_COUNT — the host picks the permutation from the bake's
    // clusterGroupSize and errors above 128 rather than dropping clusters.
    uint4 renderClusterMask = uint4(0u, 0u, 0u, 0u);
    for (uint clusterIndex = 0u; clusterIndex < group.clusterCount; clusterIndex++)
    {
        uint clusterGeneratingGroup =
            groupBlob.Load<uint>(clusterGeneratingGroupByteOffset + 4u * clusterIndex);

        if (clusterGeneratingGroup == shaderio::kOriginalMeshGroup ||
            groupAddresses[clusterGeneratingGroup].srvIndex == shaderio::kStreamingInvalidSrvIndex)
        {
            renderClusterMask[GROUP_CLUSTER_COUNT > 32 ? clusterIndex / 32u : 0u] |=
                (1u << (clusterIndex & 31u));
        }
    }

    // This lane appends a whole block, so it reserves renderClusterCount slots
    // at once and writes them at consecutive offsets.
    uint renderClusterCount = countbits(renderClusterMask.x);
#if GROUP_CLUSTER_COUNT > 32
    renderClusterCount += countbits(renderClusterMask.y);
#endif
#if GROUP_CLUSTER_COUNT > 64
    renderClusterCount += countbits(renderClusterMask.z);
#endif
#if GROUP_CLUSTER_COUNT > 96
    renderClusterCount += countbits(renderClusterMask.w);
#endif
    uint offsetClusters = ReserveRenderClusters(renderClusterCount);
    uint perThreadBase  = offsetClusters;

    for (uint clusterIndex = 0u; clusterIndex < group.clusterCount; clusterIndex++)
    {
        bool isMerged = (renderClusterMask[GROUP_CLUSTER_COUNT > 32 ? clusterIndex / 32u : 0u]
                         & (1u << (clusterIndex & 31u))) != 0u;

        if (AppendRenderCluster(isMerged, offsetClusters, mergedInstanceID,
                                group.clusterResidentID + clusterIndex))
        {
            offsetClusters++;

#if TRACK_RENDER_STATS
            // Render-stats: same per-cluster tally as traversal_run_groups, but
            // all clusters of a merged geometry attribute to the merged proxy.
            shaderio::Cluster clHdr = groupBlob.Load<shaderio::Cluster>(
                groupAddress.byteOffset + uint(sizeof(shaderio::Group)) +
                uint(sizeof(shaderio::Cluster)) * clusterIndex);
            uint clusterTris = uint(clHdr.triangleCountMinusOne) + 1u;
            uint prevInst;
            InterlockedAdd(u_PerInstanceTriangles[mergedInstanceID], clusterTris, prevInst);
            uint clusterID = group.clusterResidentID + clusterIndex;
            uint prevSeen;
            InterlockedExchange(u_UniqueSeenClusters[clusterID], 1u, prevSeen);
            if (prevSeen == 0u)
            {
                uint64_t prevUnique;
                InterlockedAdd(u_Counters[0].uniqueTriangles, (uint64_t)clusterTris, prevUnique);
                uint prevUC;
                InterlockedAdd(u_Counters[0].uniqueClusters, 1u, prevUC);
            }
#endif
        }
    }

    uint dummy;
    InterlockedAdd(u_InstanceBuildInfos[mergedInstanceID].clusterReferencesCount,
                   offsetClusters - perThreadBase, dummy);
}
