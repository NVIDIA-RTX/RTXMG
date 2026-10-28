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
  Only used with USE_BLAS_CACHING && streaming.

  Seeds one per-geometry cached-BLAS build.  One thread group per cached-BLAS
  patch (dispatched with exactly patchCachedBlasCount thread groups).  The
  threads cooperatively gather all CLAS references of the geometry's chosen
  cached LoD level (from its fully-resident groups) into the shared
  blasClusterAddresses pool, and write the per-build IndirectArgs slot +
  geometryBuildInfos.cachedBuildIndex so the subsequent cluster-BLAS build
  (shared with the per-instance builds) produces the cached BLAS.

  Implementation notes
  --------------------
  * Buffers are discretely bound (slots match the m_cachingBuildLayout in
    blas_pass.cpp); CLAS addresses come from t_ResidentClasAddrs (the same
    scene-global table blas_insert_clusters uses); geometry.lodLevels /
    streamingGroupAddresses are bindless StructuredBuffers indexed via the
    per-geometry SRV indices, and each resident group's Group header is
    loaded from its bindless group-data block (GroupAddress{srvIndex,byteOffset}),
    matching stream_update_scene's residency read model.
  * The cluster-references VA is the
    nvrhi::rt::cluster::IndirectArgs.clusterAddresses VA = base-of-pool +
    referencesOffset*8, exactly as blas_reserve_clusters seeds per-instance args.
  * Cluster gather is a two-pass wave-cooperative compaction: pass 1 reserves
    ONE contiguous block per wave from the groupshared cursor (one atomic per
    wave, not per group); pass 2 densely packs each group's clusters inside that
    block with WavePrefixSum / WaveActiveSum and no atomics.  A BLAS's CLAS list
    is an unordered set, so per-wave block order is irrelevant.
  * geometries[g].cachedBlasLodLevel/Address are NOT written here —
    stream_update_scene applies them pre-traversal, ahead of their only
    consumer, blas_elect_sharing_provider.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "rtxmg/cluster_lod/shader_registers.h"
#include "rtxmg/cluster_lod/blas_build_params.h"
#include <nvrhi/nvrhiHLSL.h>
#include "rtxmg/utils/wave_size.hlsli"

////////////////////////////////////////////

ConstantBuffer<BlasBuildParams>                      g_Params              : register(b0);

StructuredBuffer<shaderio::Geometry>                 t_Geometries          : CLOD_SRV(CLOD_T_GEOMETRIES);
StructuredBuffer<uint64_t>                           t_ResidentClasAddrs   : CLOD_SRV(CLOD_T_RESIDENT_CLAS_ADDRS);
StructuredBuffer<shaderio::StreamingGeometryPatch>   t_GeometryPatches     : CLOD_SRV(CLOD_T_GEOMETRY_PATCHES);

RWStructuredBuffer<shaderio::SceneBuildingCounters>  u_Counters            : CLOD_UAV(CLOD_U_COUNTERS);
RWStructuredBuffer<nvrhi::rt::cluster::IndirectArgs> u_BlasArgs            : CLOD_UAV(CLOD_U_BLAS_ARGS);
RWStructuredBuffer<shaderio::GeometryBuildInfo>      u_GeometryBuildInfos  : CLOD_UAV(CLOD_U_GEOMETRY_BUILD_INFOS);
RWStructuredBuffer<uint64_t>                         u_BlasClasAddrs       : CLOD_UAV(CLOD_U_BLAS_CLAS_ADDRS);

////////////////////////////////////////////

groupshared uint s_clusterOffset;  // pool-subrange cursor; each wave reserves a block

RTXMG_WAVE_SIZE_32
[numthreads(shaderio::kBlasCachingSetupBuildThreads, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint patchID       = groupID.x;       // one thread group per cached-BLAS patch
    uint localThreadID = gtid.x;

    shaderio::StreamingGeometryPatch sgpatch = t_GeometryPatches[patchID];

    uint cachedBlasLodLevel      = sgpatch.cachedBlasLodLevel;
    uint cachedBlasClustersCount = sgpatch.cachedBlasClustersCount;
    uint geometryID              = sgpatch.geometryID;

    // Rare event: the cached BLAS was fully disabled this frame (invalidate
    // patch).  Nothing to build.
    if (cachedBlasLodLevel == shaderio::kTraversalInvalidLodLevel)
        return;

    shaderio::Geometry geometry = t_Geometries[geometryID];

    StructuredBuffer<shaderio::LodLevel> lodLevels =
        ResourceDescriptorHeap[NonUniformResourceIndex(geometry.lodLevelsSRV)];
    shaderio::LodLevel lodLevelInfo = lodLevels[cachedBlasLodLevel];

    if (localThreadID == 0)
    {
        // host's HandleBlasCaching guaranteed there is room for these offsets.
        uint referencesOffset;
        InterlockedAdd(u_Counters[0].blasClasCounter, cachedBlasClustersCount, referencesOffset);
        uint buildOffset;
        InterlockedAdd(u_Counters[0].blasBuildCounter, 1u, buildOffset);

        u_GeometryBuildInfos[geometryID].cachedBuildIndex = buildOffset;

        // Seed the per-build IndirectArgs slot (mirrors blas_reserve_clusters).
        // Unlike there, clusterCount is set to the known total up front — the
        // gather below appends without incrementing it.
        u_BlasArgs[buildOffset].clusterCount     = cachedBlasClustersCount;
        u_BlasArgs[buildOffset].clusterAddressStride = 8;
        u_BlasArgs[buildOffset].clusterAddresses =
            g_Params.blasClasAddressesBaseVA + uint64_t(referencesOffset) * uint64_t(8);

        s_clusterOffset = referencesOffset;
    }

    GroupMemoryBarrierWithGroupSync();

    StructuredBuffer<shaderio::GroupAddress> groupAddresses =
        ResourceDescriptorHeap[NonUniformResourceIndex(geometry.streamingGroupAddressesSRV)];

    // ---- Pass 1: per-wave reservation -----------------------------------
    // Each lane sums the clusterCounts of the groups it will process (strided);
    // the wave then reserves one contiguous block for their total.
    // HandleBlasCaching only picks fully-loaded levels, so the invalid-srv guard
    // is purely defensive.
    uint waveClusterCount = 0u;
    for (uint i = localThreadID; i < lodLevelInfo.groupCount; i += shaderio::kBlasCachingSetupBuildThreads)
    {
        shaderio::GroupAddress ga = groupAddresses[i + lodLevelInfo.groupOffset];
        if (ga.srvIndex == shaderio::kStreamingInvalidSrvIndex)
            continue;
        ByteAddressBuffer groupData =
            ResourceDescriptorHeap[NonUniformResourceIndex(ga.srvIndex)];
        waveClusterCount += groupData.Load<shaderio::Group>(ga.byteOffset).clusterCount;
    }

    // WaveActiveSum sits after the loop, in convergent control flow, so the
    // per-wave total is well defined however the strided loop dropped lanes.
    uint waveClustersOffset = 0u;
    if (WaveIsFirstLane())
        InterlockedAdd(s_clusterOffset, WaveActiveSum(waveClusterCount), waveClustersOffset);
    waveClustersOffset = WaveReadLaneFirst(waveClustersOffset);

    // ---- Pass 2: dense pack inside the wave's block ----------------------
    // waveClustersOffset is a running cursor, uniform across the wave's active
    // lanes and advanced by the wave total each iteration.
    for (uint i = localThreadID; i < lodLevelInfo.groupCount; i += shaderio::kBlasCachingSetupBuildThreads)
    {
        shaderio::GroupAddress ga = groupAddresses[i + lodLevelInfo.groupOffset];

        uint clusterCount = 0u;
        uint clusterID    = 0u;
        if (ga.srvIndex != shaderio::kStreamingInvalidSrvIndex)
        {
            ByteAddressBuffer groupData =
                ResourceDescriptorHeap[NonUniformResourceIndex(ga.srvIndex)];
            shaderio::Group grp = groupData.Load<shaderio::Group>(ga.byteOffset);
            clusterCount = grp.clusterCount;
            clusterID    = grp.clusterResidentID;
        }

        uint dst = waveClustersOffset + WavePrefixSum(clusterCount);  // exclusive prefix
        for (uint c = 0; c < clusterCount; c++)
        {
            u_BlasClasAddrs[dst + c] = t_ResidentClasAddrs[clusterID + c];
        }
        waveClustersOffset += WaveActiveSum(clusterCount);            // advance cursor
    }
}
