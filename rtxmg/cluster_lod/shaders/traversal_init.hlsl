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

  This compute shader initializes the traversal queue with the
  root nodes of the lod hierarchy of rendered instances.

  A thread represents one instance.

  NOT compatible with USE_BLAS_SHARING; that variant ships as its own
  shader, traversal_init_blas_sharing.hlsl.

  D3D12 bindings:
  ---------------

  * SceneBuilding state is split into a per-frame
    ConstantBuffer<SceneBuildingConstants> (b0) plus a
    RWStructuredBuffer<SceneBuildingCounters> (u0).  Traversal queues are
    explicit RWStructuredBuffer<uint2> bindings; see traversal_common.hlsli.
  * geometry.nodes is read as a bindless StructuredBuffer<Node> via
    `ResourceDescriptorHeap[NonUniformResourceIndex(geometry.nodesSRV)]`.
  * Counter updates use `InterlockedAdd` (out-param form).
  * Wave intrinsics: `WaveActiveBallot`, `WaveIsFirstLane`,
    `WavePrefixCountBits`, `WaveActiveCountBits`.
  * Frustum culling is a RUNTIME switch — soft cull (`g_Constants.useCulling`,
    coarsen via culledErrorScale), hard cull (`useHardCull`, skip traversal →
    low-detail), and hard+remove (`hardCullForcesInvisible`, null the BLAS →
    invisible).
  * Per-instance BLAS addresses are written to `u_InstanceBlasAddrs[instanceID]`
    (pass.cpp builds the TLAS CPU-side and reads them out of this UAV;
    pre-filling with `lowDetailBlasAddress` gives the "low-detail BLAS unless
    promoted" default).
  * Per-instance build info is written through `u_InstanceBuildInfos[instanceID]`
    (clusterReferencesCount = 0, blasBuildIndex = shaderio::BlasBuildIndex::LowDetail).
    `InstanceBuildInfo` carries `clusterReferencesCount` + `blasBuildIndex` +
    `blasClasOffset`.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "feature_gates.hlsli"
#include "rtxmg/cluster_lod/shaders/traversal_common.hlsli"
#include "rtxmg/cluster_lod/shaders/culling.hlsli"  // frustum culling: IntersectFrustum
#include "rtxmg/utils/wave_size.hlsli"

////////////////////////////////////////////

// shaderio::BlasBuildIndex::LowDetail (= 0xFFFFFFFFu, from shaderio.h) is the sentinel
// for "use lowDetail BLAS, no dynamic BLAS scheduled this frame".
// blas_reserve_clusters promotes it to a real build index.  It must NOT collide
// with any value `atomicAdd(blasBuildCounter, 1)` produces (those start at 0).

// HiZ occlusion pyramid (previous-frame view-space max-depth) for IntersectHiz.
// One texture per level (the HiZBuffer is an array, not mips), in register space 1
// with its sampler, so the space-0 binding layout is untouched.  The test is also
// runtime-gated by g_Constants.hizNumLODs (0 => IntersectHiz returns "visible",
// e.g. when the Z pre-pass didn't run).
//
// CLUSTER_LOD_HIZ_OCCLUSION is a compile permutation: the PSO carries the space-1 HiZ
// layout and binds a HiZ set only in the =1 variant.
#if CLUSTER_LOD_HIZ_OCCLUSION
Texture2D<float> t_ClusterLodHiZ[HIZ_MAX_LODS] : register(t0, space1);
SamplerState     s_ClusterLodHiZSampler        : register(s0, space1);
#endif

////////////////////////////////////////////

RTXMG_WAVE_SIZE_32
[numthreads(shaderio::kTraversalInitThreads, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint threadID    = gid.x;
    uint instanceID  = threadID;
    uint instanceLoad = min(g_Constants.numRenderInstances - 1u, instanceID);
    bool isValid     = instanceID == instanceLoad;

    shaderio::RenderInstance instance = t_Instances[instanceLoad];
    uint geometryID = instance.geometryID;
    shaderio::Geometry geometry = t_Geometries[geometryID];

    uint blasBuildIndex = shaderio::BlasBuildIndex::LowDetail;

    float4 clipMin;
    float4 clipMax;
    bool   clipValid;

    // Runtime RT visibility test against last frame's VP: in-frustum AND
    // (near-plane straddle OR big-enough-on-screen AND not-occluded).  Every
    // term is conservative, so a soft-culled instance is only coarsened.  It
    // runs on the clamped instanceLoad data, hence harmless on padding threads;
    // isValid folds in later, at traverseInstance.
    bool isVisible = true;
    if (g_Constants.useCulling)
    {
        bool inFrustum = IntersectFrustum(g_Constants.cullViewProjMatrix,
                                          geometry.bbox.lo, geometry.bbox.hi,
                                          instance.worldMatrix,
                                          clipMin, clipMax, clipValid);
        bool bigEnough = IntersectSize(clipMin, clipMax, 1.0f, g_Constants.viewportf.xy);
#if CLUSTER_LOD_HIZ_OCCLUSION
        bool notOccluded = IntersectHiz(clipMin, clipMax, t_ClusterLodHiZ, s_ClusterLodHiZSampler,
                                        g_Constants.hizNumLODs, g_Constants.hizInvSize,
                                        g_Constants.viewportf.xy);
#else
        bool notOccluded = true;
#endif
        isVisible = inFrustum && (!clipValid || (bigEnough && notOccluded));
    }

    uint visibilityState = isVisible ? shaderio::InstanceVisibility::Visible : 0u;

    // hardCull DROPS an off-screen instance from traversal entirely; soft cull
    // keeps it traversed and just coarsens via culledErrorScale.
    const bool hardCull = g_Constants.useCulling && g_Constants.useHardCull;
    // "Traverse" means "build this instance's own BLAS" — a culledOut instance
    // is still drawn, from its low-detail BLAS, unless hardCullForcesInvisible
    // removes it (see the BLAS-address finalization below).
    const bool culledOut        = hardCull && !isVisible;
    bool       traverseInstance = isValid && !culledOut;

    bool traverseRootNode = traverseInstance;

    // NonUniformResourceIndex because geometryID varies per thread.
    StructuredBuffer<shaderio::Node> nodes =
        ResourceDescriptorHeap[NonUniformResourceIndex(geometry.nodesSRV)];

    if (traverseInstance)
    {
        // We test if we are only using the furthest lod.
        // If that is true, then we can skip lod traversal completely and
        // straight enqueue the lowest detail cluster directly.

        uint rootNodePacked = nodes[0].packed;

        uint childOffset        = Node_nodeChildOffset(rootNodePacked);
        uint childCountMinusOne = Node_nodeChildCountMinusOne(rootNodePacked);

        // test if the second to last lod needs to be traversed
        uint childNodeIndex = (childCountMinusOne > 1u) ? (childCountMinusOne - 1u) : 0u;
        shaderio::Node childNode = nodes[childOffset + childNodeIndex];
        shaderio::TraversalMetric traversalMetric = childNode.traversalMetric;

        float3x4 worldMatrix = t_Instances[instanceID].worldMatrix;
        float    uniformScale = ComputeUniformScale(worldMatrix);

        float    errorScale  = 1.0f;
        // A frustum-culled instance stays in the TLAS but uses coarser LoD
        // (the root coarse node is preferred).  Runtime-gated on useCulling.
        if (g_Constants.useCulling && visibilityState == 0u)
            errorScale = g_Constants.culledErrorScale;

        float4x4 transform = mul(g_Constants.traversalViewMatrix, ToFloat4x4(worldMatrix));

        // if there is no need to traverse the pen ultimate lod level,
        // then just insert the last lod level node's cluster directly.
        if (!TestForTraversal(transform, uniformScale, traversalMetric, errorScale))
        {
            // Nothing to enqueue — low-detail BLAS substitution happens
            // implicitly via u_InstanceBlasAddrs below.
            traverseRootNode = false;
        }
    }

    EnqueueRootNode(instanceID, geometry, traverseRootNode);

    // Per-instance build-info seed + low-detail BLAS address pre-fill; pass.cpp
    // consumes both downstream (instanceBuildInfos by blas_setup,
    // instanceBlasAddrs by TLAS fill).
    if (isValid)
    {
        // Publish per-instance visibility so traversal_run can read it this same
        // frame for its culledErrorScale decision.
        if (g_Constants.useCulling)
            u_InstanceVisibility[instanceID] = visibilityState;

        shaderio::InstanceBuildInfo info;
        info.clusterReferencesCount = 0u;
        info.blasBuildIndex         = blasBuildIndex;
        info.blasClasOffset         = 0u;
        // lodLevelMin/Max are only consumed on the USE_BLAS_SHARING path (which
        // replaces this shader with traversal_init_blas_sharing); leave them
        // invalid here.
        info.lodLevelMin            = shaderio::kTraversalInvalidLodLevel;
        info.lodLevelMax            = shaderio::kTraversalInvalidLodLevel;
        info._lodReserved           = 0u;
        u_InstanceBuildInfos[instanceID] = info;

        // Resolve this instance's BLAS address.  A culledOut instance normally
        // falls back to the low-detail BLAS; with hardCullForcesInvisible it is
        // REMOVED instead — a null BLAS makes the TLAS build drop it.  The TLAS
        // is rebuilt every frame, so no frameIndex guard is needed.
        if (culledOut && g_Constants.hardCullForcesInvisible)
        {
            u_InstanceBlasAddrs[instanceID] = (nvrhi::GpuVirtualAddress)0;
        }
        else
        {
            // Default: point this instance at its low-detail BLAS.  Promoted to a
            // dynamic BLAS GVA by instance_assign_blas once traversal_run_groups
            // has scheduled enough clusters (only happens for traversed instances).
            u_InstanceBlasAddrs[instanceID] = geometry.lowDetailBlasAddress;
        }
    }
}
