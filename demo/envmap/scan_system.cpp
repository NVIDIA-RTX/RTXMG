/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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