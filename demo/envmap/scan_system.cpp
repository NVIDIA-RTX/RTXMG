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

#include "scan_system.h"

#include <nvrhi/utils.h>
#include <donut/core/log.h>
#include <donut/core/math/math.h>

#include "rtxmg/utils/buffer.h"

using namespace donut;
using namespace donut::math;

void ScanSystem::Init(std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, nvrhi::IDevice* device)
{
    m_prefixScan = shaderFactory->CreateShader("envmap/prefix_scan.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);
    m_prefixScanPSO.Reset();
    if (!m_prefixScan)
    {
        log::fatal("Failed to create prefix scan shader");
    }
    m_prefixScanParams = device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(PrefixScanParams), "prefixScanParams",
        engine::c_MaxRenderPassConstantBufferVersions));
}

void ScanSystem::PrefixScan(nvrhi::IBuffer* inputBuffer, nvrhi::IBuffer* outputBuffer, int inputWidth, int inputHeight, nvrhi::ICommandList* commandList)
{
    auto device = commandList->getDevice();

    PrefixScanParams params = {};
    params.elementCountX = inputWidth;
    params.elementCountY = inputHeight;
    params.outputWidth = inputWidth + 2;
    commandList->writeBuffer(m_prefixScanParams, &params, sizeof(params));

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::TypedBuffer_SRV(0, inputBuffer))
        .addItem(nvrhi::BindingSetItem::TypedBuffer_UAV(0, outputBuffer))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_prefixScanParams));

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_prefixScanBSL, bindingSet))
    {
        log::fatal("Failed to create binding set and layout for prefix scan shader");
    }

    if (!m_prefixScanPSO)
    {
        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(m_prefixScan)
            .addBindingLayout(m_prefixScanBSL);

        m_prefixScanPSO = device->createComputePipeline(computePipelineDesc);
    }
    
    auto state = nvrhi::ComputeState()
        .setPipeline(m_prefixScanPSO)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);

    const uint32_t launchDimY = div_ceil(inputHeight, 16);
    commandList->dispatch(1, launchDimY);
}