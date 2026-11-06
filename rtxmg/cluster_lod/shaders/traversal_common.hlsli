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

// traversal_common.hlsli
// Shared bindings, pack/unpack helpers, and LOD metric utilities for the
// traversal-family shaders.
//
// Each shader in this directory documents what it does, but the order they run
// in is not discoverable from any of them, so it is written down here once.
// The host side is ClusterLodPass::Execute.
//
//   traversal_init          One thread per instance.  Frustum / screen-size /
//                           HiZ-culls the instance, then tests the root node's
//                           second-coarsest child.  If even that is fine enough
//                           the instance is never queued and renders from its
//                           persistent low-detail BLAS; otherwise the root is
//                           wave-appended to u_TraversalNodeQ.
//   traversal_setup         1x1x1, between every stage below.  Clamps the queue
//                           counters and refreshes the indirect dispatch grids.
//   traversal_run           One thread per CHILD of a queued node, wave-packed
//                           so nodes with differing child counts still fill
//                           consecutive lanes.  Each child either re-enters
//                           u_TraversalNodeQ or, being a leaf group, moves to
//                           u_TraversalGroupQ.  A leaf group that is not
//                           resident is dropped and a (geometryID, groupIndex)
//                           load request appended to u_LoadGeometryGroups.
//   traversal_run_groups    One thread per group, looping its clusters.  The
//                           continuous-LoD rule lives here: reaching this kernel
//                           already means the group's own error was too large to
//                           stop above it, so a cluster is emitted once its
//                           GENERATING group — the finer group it was simplified
//                           from — is fine enough.  An original-mesh or
//                           non-resident generating group forces the cluster out.
//   traversal_blas_merging  Optional and, despite the name, not part of the DAG
//                           walk: it gathers a merged per-geometry proxy's
//                           clusters into the same render list and runs the
//                           streaming age filter in place of stream_age_groups.
//
// Under USE_BLAS_SHARING, instance_classify_lod + blas_elect_sharing_provider +
// traversal_init_blas_sharing replace traversal_init.
//
// Downstream of all of the above: blas_reserve_clusters ->
// blas_insert_clusters -> the cluster BLAS build -> instance_assign_blas, which
// resolves each instance's final BLAS address for the host-side TLAS fill.

#pragma once

#include "rtxmg/cluster_lod/shaderio.h"
#include "rtxmg/cluster_lod/shader_registers.h"
#include "feature_gates.hlsli"

#pragma pack_matrix(row_major)

// ---------------------------------------------------------------------------
// Register bindings — slots come from shader_registers.h, which
// ClusterLodPass::CreateBindingLayout reads as well.
// ---------------------------------------------------------------------------

ConstantBuffer<shaderio::SceneBuildingConstants>   g_Constants    : register(b0);

RWStructuredBuffer<shaderio::SceneBuildingCounters> u_Counters       : CLOD_UAV(CLOD_U_COUNTERS);
RWStructuredBuffer<uint2>                           u_TraversalNodeQ : CLOD_UAV(CLOD_U_TRAVERSAL_NODE_Q);
RWStructuredBuffer<uint2>                           u_TraversalGroupQ: CLOD_UAV(CLOD_U_TRAVERSAL_GROUP_Q);
RWStructuredBuffer<uint2>                           u_RenderClusters : CLOD_UAV(CLOD_U_RENDER_CLUSTERS);
// per-instance cluster counts + BLAS build indices (written by traversal_run_groups, read by blas_setup)
RWStructuredBuffer<shaderio::InstanceBuildInfo>     u_InstanceBuildInfos  : CLOD_UAV(CLOD_U_INSTANCE_BUILD_INFOS);
// per-instance BLAS device addresses for TLAS fill; pre-filled with lowDetailBlasAddress
// by traversal_init, then overwritten with the dynamic BLAS address by instance_assign_blas.
RWStructuredBuffer<nvrhi::GpuVirtualAddress>        u_InstanceBlasAddrs   : CLOD_UAV(CLOD_U_INSTANCE_BLAS_ADDRS);
// scene-global StreamingGroup table.  traversal_run_groups clears
// residentGroups[group.groupResidentID].age on visit; the next agefilter
// dispatch reads the same slot.  The preload path binds a size-1 placeholder.
RWStructuredBuffer<shaderio::StreamingGroup>        u_ResidentGroups      : CLOD_UAV(CLOD_U_RESIDENT_GROUPS);
// Streaming state + the frame's load-request ring, written by traversal_run's
// load-emit branch.  The preload path binds size-1 placeholders.
RWStructuredBuffer<shaderio::SceneStreaming>        streamingRW           : CLOD_UAV(CLOD_U_STREAMING);
RWStructuredBuffer<uint2>                           u_LoadGeometryGroups  : CLOD_UAV(CLOD_U_LOAD_GEOMETRY_GROUPS);
// per-instance visibility bits (shaderio::InstanceVisibility):
// traversal_init / instance_classify_lod publish the cull result, and
// traversal_init_blas_sharing tags merged instances so traversal_run can skip
// cluster iteration for them.  Always bound; host clears it each frame.
RWStructuredBuffer<uint>                            u_InstanceVisibility  : CLOD_UAV(CLOD_U_INSTANCE_VISIBILITY);

// Render-stats (TRACK_RENDER_STATS): per-instance rendered-triangle tally and a
// per-resident-cluster "seen this frame" flag for the unique-triangle dedup.
// Always bound; only touched under the gate.
RWStructuredBuffer<uint>                            u_PerInstanceTriangles : CLOD_UAV(CLOD_U_PER_INSTANCE_TRIANGLES);
RWStructuredBuffer<uint>                            u_UniqueSeenClusters   : CLOD_UAV(CLOD_U_UNIQUE_SEEN_CLUSTERS);

// Streaming age-filter slots, read/written only by traversal_blas_merging (it
// replaces the standalone stream_age_groups dispatch when merging is on).  The
// preload path binds size-1 placeholders.
RWStructuredBuffer<uint>                            u_ActiveGroups         : CLOD_UAV(CLOD_U_ACTIVE_GROUPS);
RWStructuredBuffer<uint>                            u_GroupIDs             : CLOD_UAV(CLOD_U_GROUP_IDS);
RWStructuredBuffer<uint2>                           u_UnloadGeometryGroups : CLOD_UAV(CLOD_U_UNLOAD_GEOMETRY_GROUPS);

StructuredBuffer<shaderio::Geometry>               t_Geometries   : CLOD_SRV(CLOD_T_GEOMETRIES);
StructuredBuffer<shaderio::RenderInstance>         t_Instances    : CLOD_SRV(CLOD_T_RENDER_INSTANCES);

#if USE_BLAS_SHARING
// Per-geometry BLAS-sharing tables.  One entry per unique geometry, persistent
// allocation.  Histograms are accumulated by instance_classify_lod (atomics) and
// cleared to zero each frame host-side; build infos are written by
// blas_elect_sharing_provider and read by traversal_init_blas_sharing /
// instance_assign_blas.
RWStructuredBuffer<shaderio::GeometryBuildInfo>      u_GeometryBuildInfos : CLOD_UAV(CLOD_U_GEOMETRY_BUILD_INFOS);
RWStructuredBuffer<shaderio::GeometryBuildHistogram> u_GeometryHistograms : CLOD_UAV(CLOD_U_GEOMETRY_HISTOGRAMS);
#endif

// ---------------------------------------------------------------------------
// TraversalInfo pack / unpack  (uint2 ↔ TraversalInfo)
// ---------------------------------------------------------------------------

uint2 PackTraversalInfo(shaderio::TraversalInfo info)
{
    return uint2(info.instanceID, info.packedNode);
}

shaderio::TraversalInfo UnpackTraversalInfo(uint2 packed)
{
    shaderio::TraversalInfo info;
    info.instanceID = packed.x;
    info.packedNode = packed.y;
    return info;
}

// ---------------------------------------------------------------------------
// Wave-coalesced queue appends — one atomic per wave instead of one per thread.
// ---------------------------------------------------------------------------

// Seeds u_TraversalNodeQ with one instance's root node.  Shared by
// traversal_init and traversal_init_blas_sharing, which differ only in how they
// decide doEnqueue.
void EnqueueRootNode(uint instanceID, shaderio::Geometry geometry, bool doEnqueue)
{
    uint4 voteNodes = WaveActiveBallot(doEnqueue);
    uint  voteCount = countbits(voteNodes.x) + countbits(voteNodes.y)
                    + countbits(voteNodes.z) + countbits(voteNodes.w);

    uint offsetNodes = 0u;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(u_Counters[0].traversalNodeWriteCounter, voteCount, offsetNodes);
    }
    offsetNodes  = WaveReadLaneFirst(offsetNodes);
    offsetNodes += WavePrefixCountBits(doEnqueue);

    if (doEnqueue && offsetNodes < g_Constants.maxTraversalInfos)
    {
        // NonUniformResourceIndex because geometryID varies per thread.
        StructuredBuffer<shaderio::Node> nodes =
            ResourceDescriptorHeap[NonUniformResourceIndex(geometry.nodesSRV)];

        shaderio::TraversalInfo traversalInfo;
        traversalInfo.instanceID = instanceID;
        traversalInfo.packedNode = nodes[0].packed;

        u_TraversalNodeQ[offsetNodes] = PackTraversalInfo(traversalInfo);
    }
}

// Reserves this lane's `count` contiguous slots in the unordered render-cluster
// list and returns their base offset; lanes reserving nothing pass 0.
uint ReserveRenderClusters(uint count)
{
    uint offsetClusters = WavePrefixSum(count);   // exclusive prefix
    uint waveTotal      = WaveActiveSum(count);

    uint offsetBase = 0u;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(u_Counters[0].renderClusterCounter, waveTotal, offsetBase);
    }
    return WaveReadLaneFirst(offsetBase) + offsetClusters;
}

// Writes one reserved slot, dropping the cluster when the budget is spent.
// ClusterInfo == uint2{ instanceID, clusterID } (shaderio aliasing).
bool AppendRenderCluster(bool doAppend, uint offset, uint instanceID, uint clusterID)
{
    if (doAppend && offset < g_Constants.maxRenderClusters)
    {
        u_RenderClusters[offset] = uint2(instanceID, clusterID);
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// LOD metric helpers
// ---------------------------------------------------------------------------

// Extract uniform scale from the upper-left 3×3 of a row-major float3x4.
float ComputeUniformScale(float3x4 worldMatrix)
{
    float3 c0 = float3(worldMatrix[0][0], worldMatrix[1][0], worldMatrix[2][0]);
    float3 c1 = float3(worldMatrix[0][1], worldMatrix[1][1], worldMatrix[2][1]);
    float3 c2 = float3(worldMatrix[0][2], worldMatrix[1][2], worldMatrix[2][2]);
    return max(max(length(c0), length(c1)), length(c2));
}

// Extend a row-major float3x4 to float4x4 by appending row [0,0,0,1].
float4x4 ToFloat4x4(float3x4 m)
{
    return float4x4(m[0], m[1], m[2], float4(0.0f, 0.0f, 0.0f, 1.0f));
}

// Key LOD-metric evaluation — returns true when this node/group's error is still
// too large for its distance, i.e. KEEP DESCENDING.  Nodes use the result
// directly; clusters negate it against their GENERATING group's metric, so a
// cluster is emitted once the finer group it was simplified from tests false.
// instanceToEye = traversalViewMatrix * worldMatrix (both row-major).
// errorScale attenuates the LoD of culled instances: it scales the THRESHOLD,
// not the error, so > 1 descends less (coarser); 1 is the unculled case.
bool TestForTraversal(float4x4 instanceToEye, float uniformScale,
                      shaderio::TraversalMetric metric, float errorScale)
{
    float3 spherePos    = float3(metric.boundingSphereX,
                                  metric.boundingSphereY,
                                  metric.boundingSphereZ);
    float  minDist      = g_Constants.nearPlane;
    float3 eyeSpacePos  = mul(instanceToEye, float4(spherePos, 1.0f)).xyz;
    float  sphereDist   = length(eyeSpacePos);
    float  errDist      = max(minDist, sphereDist - metric.boundingSphereRadius * uniformScale);
    float  errOverDist  = metric.maxQuadricError * uniformScale / errDist;
    return errOverDist >= g_Constants.errorOverDistanceThreshold * errorScale;
}

bool TestForTraversal(float4x4 instanceToEye, float uniformScale,
                      shaderio::TraversalMetric metric)
{
    return TestForTraversal(instanceToEye, uniformScale, metric, 1.0f);
}
