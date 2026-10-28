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
#pragma pack_matrix(row_major)

#include <nvrhi/nvrhiHLSL.h>
#include "rtxmg/cluster_tess/fill_blas_from_clas_args_params.h"
#include "rtxmg/cluster_tess/copy_cluster_offset_params.h"

ConstantBuffer<FillBlasFromClasArgsParams> g_Params : register(b0);

StructuredBuffer<uint2> t_ClusterOffsetCounts : register(t0);
RWStructuredBuffer<nvrhi::rt::cluster::IndirectArgs> u_BlasFromClasArgs : register(u0);

[numthreads(kFillBlasFromClasArgsThreads, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint32_t instanceIndex = threadIdx.x;
    if (instanceIndex >= g_Params.numInstances)
        return;

    uint2 offsetCount = t_ClusterOffsetCounts[instanceIndex * ClusterTessDispatchType::NumTypes + ClusterTessDispatchType::All];

    nvrhi::rt::cluster::IndirectArgs args = (nvrhi::rt::cluster::IndirectArgs)0;
    args.clusterCount = offsetCount.y;
    args.clusterAddresses = g_Params.clasAddressesBaseAddress + sizeof(nvrhi::GpuVirtualAddress) * offsetCount.x;
    args.clusterAddressStride = 8;
    u_BlasFromClasArgs[instanceIndex] = args;
}