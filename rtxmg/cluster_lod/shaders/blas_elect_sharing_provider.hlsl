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
  Only used with USE_BLAS_SHARING.

  Evaluates each geometry's LoD histogram (accumulated by
  instance_classify_lod) to elect a single canonical "sharing provider"
  instance whose freshly-built BLAS can be reused by many lower-detail
  instances of the same geometry.  One thread per geometry.

  Implementation notes
  --------------------
  * Histograms and build infos are read/written via u_GeometryHistograms /
    u_GeometryBuildInfos (see traversal_common.hlsli).
  * USE_BLAS_CACHING is a per-PSO permutation; BLAS merging is a runtime
    switch (g_Constants.useBlasMerging).
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "feature_gates.hlsli"
#include "rtxmg/cluster_lod/shaders/traversal_common.hlsli"

bool TestForBlasSharing(shaderio::Geometry geometry)
{
#if USE_BLAS_CACHING
    return geometry.instancesCount >= 1u;
#else
    return geometry.instancesCount >= 2u;
#endif
}

[numthreads(shaderio::kGeometryBlasSharingThreads, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint geometryID = gid.x;

    if (geometryID < g_Constants.numGeometries)
    {
        shaderio::Geometry geometry = t_Geometries[geometryID];

#if USE_BLAS_CACHING
        uint cachedLevel       = geometry.cachedBlasLodLevel;
        uint cachedLevelNeeded = shaderio::kTraversalInvalidLodLevel;
#endif
        uint shareLevelMin    = shaderio::kTraversalInvalidLodLevel;
        uint shareLevelMax    = shaderio::kTraversalInvalidLodLevel;
        uint shareInstanceID  = ~0u;
        uint mergedInstanceID = ~0u;

        if (TestForBlasSharing(geometry))
        {
            // number of per-frame per-instance builds that are NOT sharing
            uint numBuildInstances = 0u;
            // previous iteration's instance count (for tolerance roll-back)
            uint numPrevInstances  = 0u;

            uint lodLevelsCount    = geometry.lodLevelsCount;
            bool allowMerge        = true;

            uint sharingToleranceLevel = max(1u, lodLevelsCount - min(g_Constants.sharingTolerantLevels, lodLevelsCount));
            uint sharingMinLevel       = lodLevelsCount - min(g_Constants.sharingEnabledLevels, lodLevelsCount);

            for (uint lodLevel = 0u; lodLevel < lodLevelsCount; lodLevel++)
            {
                uint numInstances = u_GeometryHistograms[geometryID].lodLevelMinHistogram[lodLevel];
                bool terminates   = u_GeometryHistograms[geometryID].lodLevelMaxHistogram[lodLevel] > 0u;

#if USE_BLAS_CACHING
                if (numInstances > 0u && lodLevel >= cachedLevel)
                {
                    if (cachedLevelNeeded == shaderio::kTraversalInvalidLodLevel)
                    {
                        cachedLevelNeeded = lodLevel;
                        // sharing is unnecessary at/after the cached level, but an
                        // election below it still has consumers in between
                        if (shareLevelMax == shaderio::kTraversalInvalidLodLevel)
                        {
                            shareLevelMax = shaderio::kTraversalInvalidLodLevel - 1u;
                        }
                        // only merge if instances existed before the cached level
                        allowMerge        = numBuildInstances != 0u;
                    }
                }
#endif

                // Elect the merged proxy: the first instance that terminates
                // early hosts the geometry's merged BLAS.
                if (g_Constants.useBlasMerging && mergedInstanceID == ~0u && terminates && allowMerge)
                {
                    uint packedLodInstance = u_GeometryHistograms[geometryID].lodLevelMaxPackedInstance[lodLevel];
                    mergedInstanceID       = packedLodInstance & ((1u << 27) - 1u);
                }

                // still looking for the sharing level, and some instance ends here?
                if (lodLevel >= sharingMinLevel &&
                    shareLevelMax == shaderio::kTraversalInvalidLodLevel &&
                    terminates)
                {
                    uint packedLodInstance = u_GeometryHistograms[geometryID].lodLevelMaxPackedInstance[lodLevel];
                    shareLevelMax   = lodLevel;
                    shareLevelMin   = packedLodInstance >> 27;
                    shareInstanceID = packedLodInstance & ((1u << 27) - 1u);

                    // from some level onwards, add one level of tolerance
                    if (lodLevel >= sharingToleranceLevel && shareLevelMin < lodLevel)
                    {
                        // pretend we "end" one lod level earlier
                        shareLevelMax     = lodLevel - 1u;
                        numBuildInstances -= numPrevInstances;
                    }
                }

                // if not sharing, append to running number of per-instance builds,
                // these might be merged or not later
                //
                // != shaderio::kTraversalInvalidLodLevel would be the default condition
                // but given BLAS_CACHING uses shaderio::kTraversalInvalidLodLevel - 1, we also want
                // to account for that
                if (shareLevelMax > shaderio::kTraversalInvalidLodLevel - 1u)
                {
                    numBuildInstances += numInstances;
                }

                numPrevInstances = numInstances;
            }
        }

#if USE_BLAS_CACHING
        // influences the streaming age filter (kept alive only if used)
        u_GeometryBuildInfos[geometryID].cachedLevel      = cachedLevelNeeded;
#else
        u_GeometryBuildInfos[geometryID].cachedBuildIndex = ~0u;
        u_GeometryBuildInfos[geometryID].cachedLevel      = shaderio::kTraversalInvalidLodLevel;
#endif
        u_GeometryBuildInfos[geometryID].shareLevelMin   = shareLevelMin;
        u_GeometryBuildInfos[geometryID].shareLevelMax   = shareLevelMax;
        u_GeometryBuildInfos[geometryID].shareInstanceID = shareInstanceID;
        // Always published (~0u when merging is off or no proxy was elected);
        // traversal_init_blas_sharing + traversal_blas_merging read it.
        u_GeometryBuildInfos[geometryID].mergedInstanceID = mergedInstanceID;

#if TRACK_RENDER_STATS
        // Diagnostic: count geometries that elected a sharing provider.
        if (shareInstanceID != ~0u)
        {
            uint dummy;
            InterlockedAdd(u_Counters[0].numSharingProviders, 1u, dummy);
        }
        // Diagnostic: count geometries with a live merged BLAS this frame.
        if (mergedInstanceID != ~0u)
        {
            uint dummyMerged;
            InterlockedAdd(u_Counters[0].numMergedBlas, 1u, dummyMerged);
        }
#endif
    }
}
