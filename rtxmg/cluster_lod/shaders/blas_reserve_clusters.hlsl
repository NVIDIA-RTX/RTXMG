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


  This compute shader sets up the per BLAS cluster references list start
  pointer.  It does so by simply adding up the per-blas references counts
  that were filled during `traversal_run`.  These count values are also
  reset, so that the `blas_insert_clusters` kernel can increment them
  again when filling the lists.

  A single thread represents one BLAS.

  rtxmg port notes (D3D12, no GL_EXT_buffer_reference):
  -----------------------------------------------------

  * SceneBuilding state is discretely bound at fixed register slots matching
    m_setupLayout in [blas_pass.cpp]; the blasClasCounter / blasBuildCounter
    atomics live in SceneBuildingCounters[0].
  * The per-BLAS reference list is an nvrhi::rt::cluster::IndirectArgs entry
    whose `clusterAddresses` is a precomputed VA into the shared
    `u_BlasClasAddrs` pool; the implicit 8-byte stride is
    sizeof(GpuVirtualAddress).
  * `InstanceBuildInfo.blasClasOffset = referencesOffset` is also written,
    because blas_insert_clusters needs the index and not just the VA to update
    the per-instance CLAS address slot.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "rtxmg/cluster_lod/shader_registers.h"
#include "rtxmg/cluster_lod/blas_build_params.h"

// kBlasSetupInsertionThreads comes from shaderio.h (shared with C++).

////////////////////////////////////////////

ConstantBuffer<BlasBuildParams>                              g_Params         : register(b0);
RWStructuredBuffer<shaderio::SceneBuildingCounters>          u_Counters       : CLOD_UAV(CLOD_U_COUNTERS);
RWStructuredBuffer<shaderio::InstanceBuildInfo>              u_InstanceBuildInfos : CLOD_UAV(CLOD_U_INSTANCE_BUILD_INFOS);
RWStructuredBuffer<nvrhi::rt::cluster::IndirectArgs>         u_BlasArgs       : CLOD_UAV(CLOD_U_BLAS_ARGS);

////////////////////////////////////////////

[numthreads(shaderio::kBlasSetupInsertionThreads, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint instanceID = gid.x;

    if (instanceID < g_Params.numRenderInstances)
    {
        uint referencesCount = u_InstanceBuildInfos[instanceID].clusterReferencesCount;
        if (referencesCount > 0)
        {
            uint referencesOffset;
            InterlockedAdd(u_Counters[0].blasClasCounter,  referencesCount, referencesOffset);
            uint buildOffset;
            InterlockedAdd(u_Counters[0].blasBuildCounter, 1u,              buildOffset);

            // reset cluster count for the insertion pass to InterlockedAdd into,
            // and seed the indirect-args slot with this BLAS's slice of the
            // shared CLAS-address pool.
            u_BlasArgs[buildOffset].clusterCount     = 0;
            u_BlasArgs[buildOffset].clusterAddressStride = 8;
            u_BlasArgs[buildOffset].clusterAddresses =
                g_Params.blasClasAddressesBaseVA +
                uint64_t(referencesOffset) * uint64_t(8);

            // Hand the per-BLAS build slot index and the per-BLAS CLAS-pool
            // offset back to the per-instance record so blas_insert_clusters
            // can locate both its IndirectArgs slot and its CLAS-address
            // sub-range without recomputing.
            u_InstanceBuildInfos[instanceID].blasBuildIndex  = buildOffset;
            u_InstanceBuildInfos[instanceID].blasClasOffset  = referencesOffset;
        }
    }
}
