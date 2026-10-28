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
  Only used with USE_BLAS_SHARING.  Replaces
  traversal_init.hlsl on the sharing path.

  Initializes the traversal queue with the root nodes of the instances that
  actually need a per-instance BLAS.  Using the LoD range from
  instance_classify_lod and the sharing decision from blas_elect_sharing_provider,
  each instance is classified into one of four categories:

    1. low-detail only       -> use pre-built low-detail BLAS, no traversal
    2. sharing provider       -> traverse + build its own BLAS (others reuse it)
    3. sharing consumer        -> reuse provider's BLAS (ShareBit), no traversal
    4. view-dependent          -> traverse + build its own per-instance BLAS

  One thread per instance.

  Notes:
  * Geometry buffers are read via bindless StructuredBuffer; wave intrinsics
    coalesce the node-queue append (see traversal_init.hlsl for the same idioms).
  * geometryLodLevelMax is derived from geometry.lodLevelsCount - 1.
  * USE_BLAS_CACHING is a per-PSO permutation; frustum culling and BLAS merging
    are runtime switches (g_Constants.useCulling / useBlasMerging).
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "feature_gates.hlsli"
#include "rtxmg/cluster_lod/shaders/traversal_common.hlsli"
#include "rtxmg/utils/wave_size.hlsli"

RTXMG_WAVE_SIZE_32
[numthreads(shaderio::kTraversalInitThreads, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint instanceID   = gid.x;
    uint instanceLoad = min(g_Constants.numRenderInstances - 1u, instanceID);
    bool isValid      = instanceID == instanceLoad;

    shaderio::RenderInstance instance = t_Instances[instanceLoad];
    uint geometryID = instance.geometryID;
    shaderio::Geometry geometry = t_Geometries[geometryID];

    // by default all instances fall back to the lowest-detail BLAS
    uint blasBuildIndex = shaderio::BlasBuildIndex::LowDetail;

    // instance LoD range, computed in instance_classify_lod
    shaderio::InstanceBuildInfo instanceInfo = u_InstanceBuildInfos[instanceLoad];
    uint instanceLevelMin    = instanceInfo.lodLevelMin;
    uint geometryLodLevelMax = geometry.lodLevelsCount - 1u;

    // geometry's sharing election, computed in blas_elect_sharing_provider
    uint cachedLevel     = u_GeometryBuildInfos[geometryID].cachedLevel;
    uint shareLevelMax   = u_GeometryBuildInfos[geometryID].shareLevelMax;
    uint shareInstanceID = u_GeometryBuildInfos[geometryID].shareInstanceID;
    // ~0u when merging is off or no proxy was elected.
    uint mergedInstanceID = u_GeometryBuildInfos[geometryID].mergedInstanceID;
    // Read on the CLAMPED index: the dispatch rounds the thread count up to a
    // kTraversalInitThreads multiple, so overhang threads have
    // instanceID >= numRenderInstances.  The writes below stay on instanceID
    // but are isValid-gated.
    uint visibilityState  = u_InstanceVisibility[instanceLoad];

    // instance_classify_lod published visibilityState this frame.  Hard culling
    // drops off-screen instances from traversal (low-detail BLAS fallback);
    // soft culling keeps them, just coarser, so they stay traversable.
    const bool hardCull  = g_Constants.useCulling && g_Constants.useHardCull;
    const bool isVisible = (visibilityState & shaderio::InstanceVisibility::Visible) != 0u;

    bool traverseRootNode = false;
    if (isValid && (isVisible || !hardCull))
    {
        // An instance can be both the share provider and the merged proxy.
        if (g_Constants.useBlasMerging && mergedInstanceID == instanceID)
        {
            traverseRootNode = true;
            u_InstanceVisibility[instanceID] = visibilityState | shaderio::InstanceVisibility::UsesMerged;
        }

        if (instanceLevelMin == geometryLodLevelMax)
        {
            // (1) low-detail BLAS only — nothing to traverse
            traverseRootNode = false;
        }
        else if (shareInstanceID == instanceID)
        {
            // (2) this instance is the shared provider — traverse + build
            traverseRootNode = true;
        }
#if USE_BLAS_CACHING
        else if (instanceLevelMin >= cachedLevel)
        {
            // use the geometry's cached BLAS instead
            blasBuildIndex   = geometryID | shaderio::BlasBuildIndex::CacheBit;
            traverseRootNode = false;
        }
#endif
        else if (instanceLevelMin >= shareLevelMax)
        {
            // (3) consumer — reuse the provider's BLAS
            blasBuildIndex   = shareInstanceID | shaderio::BlasBuildIndex::ShareBit;
            traverseRootNode = false;

#if TRACK_RENDER_STATS
            // Diagnostic: count instances that reused a shared BLAS.
            uint dummy;
            InterlockedAdd(u_Counters[0].numSharingConsumers, 1u, dummy);
#endif
        }
        else
        {
            // (4) view-dependent — regular per-instance traversal
            traverseRootNode = true;

            // View-dependent instance of a merged geometry: tag it so
            // traversal_run skips cluster iteration, and route its BLAS to the
            // merged proxy (unless this instance IS the proxy, which builds it).
            if (g_Constants.useBlasMerging && mergedInstanceID != ~0u)
            {
                u_InstanceVisibility[instanceID] |= shaderio::InstanceVisibility::UsesMerged;
                if (mergedInstanceID != instanceID)
                {
                    blasBuildIndex = mergedInstanceID | shaderio::BlasBuildIndex::ShareBit;
                }
            }
        }
    }

    EnqueueRootNode(instanceID, geometry, traverseRootNode);

    if (isValid)
    {
        u_InstanceBuildInfos[instanceID].clusterReferencesCount = 0u;
        u_InstanceBuildInfos[instanceID].blasBuildIndex         = blasBuildIndex;
    }
}
