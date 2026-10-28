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
#include <donut/shaders/cluster_args.hlsli>
#include "rtxmg/cluster_tess/fill_instantiate_template_args_params.h"

ConstantBuffer<FillInstantiateTemplateArgsParams> g_Params : register(b0);

StructuredBuffer<nvrhi::GpuVirtualAddress> t_TemplateAddresses : register(t0);
RWStructuredBuffer<donut::cluster::IndirectInstantiateTemplateArgs> u_InstantiateTemplateArgs : register(u0);

[numthreads(kFillInstantiateTemplateArgsThreads, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint templateIndex = threadIdx.x;
    if (templateIndex >= g_Params.numTemplates)
        return;

    donut::cluster::IndirectInstantiateTemplateArgs args = (donut::cluster::IndirectInstantiateTemplateArgs)0;
    args.clusterTemplate = t_TemplateAddresses[templateIndex];
    args.vertexBuffer.startAddress = 0; // not providing vertex positions returns the worst case m_size 
    args.vertexBuffer.strideInBytes = 0;
    u_InstantiateTemplateArgs[templateIndex] = args;
}