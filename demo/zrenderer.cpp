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
//


#include "zrender_params.h"
#include "zrenderer.h"


#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/core/math/math.h>
#include <nvrhi/utils.h>

#include <filesystem>
#include <string>
#include <sstream>

using namespace donut;
using namespace donut::math;

#include "rtxmg/utils/buffer.h"
#include "rtxmg/profiler/statistics.h"
#include "rtxmg/scene/camera.h"

ZRenderer::ZRenderer(std::shared_ptr<engine::ShaderFactory> shaderFactory)
    : m_shaderFactory(shaderFactory)
{

}

ZRenderer::~ZRenderer() {}

void ZRenderer::BuildPipeline(nvrhi::IDevice* device)
{
    nvrhi::BindingLayoutDesc globalBindingLayoutDesc;
    globalBindingLayoutDesc.visibility = nvrhi::ShaderType::All;
    globalBindingLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0), // z render constants
        nvrhi::BindingLayoutItem::RayTracingAccelStruct(0),  // TLAS
        nvrhi::BindingLayoutItem::Texture_UAV(0),          // output
    };
    m_bindingLayout = device->createBindingLayout(globalBindingLayoutDesc);

    m_params =
        device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
            sizeof(ZRenderParams), "Z Render Params",
            engine::c_MaxRenderPassConstantBufferVersions));

    m_shaderLibrary =
        m_shaderFactory->CreateShaderLibrary("rtxmg_demo/zrender.hlsl", nullptr);

    if (!m_shaderLibrary)
    {
        log::fatal("Failed to create z render shader library");
    }

    nvrhi::rt::PipelineDesc pipelineDesc;
    pipelineDesc.globalBindingLayouts = { m_bindingLayout };
    pipelineDesc.shaders =
    {
        {"",m_shaderLibrary->getShader("RayGen", nvrhi::ShaderType::RayGeneration),nullptr},
        {"", m_shaderLibrary->getShader("Miss", nvrhi::ShaderType::Miss),nullptr},
    };

    pipelineDesc.hitGroups = { {
        "ZHitGroup",
        m_shaderLibrary->getShader("ClosestHit", nvrhi::ShaderType::ClosestHit),
        nullptr, // anyHit
        nullptr, // intersectionShader
        nullptr, // bindingLayout
        false    // isProceduralPrimitive
    } };

    pipelineDesc.maxPayloadSize = sizeof(ZRayPayload);
    pipelineDesc.maxRecursionDepth = 2;
    pipelineDesc.allowClusters = true;

    m_rayPipeline = device->createRayTracingPipeline(pipelineDesc);

    if (!m_rayPipeline)
    {
        log::fatal("Failed to create Z ray tracing pipeline");
    }

    m_shaderTable = m_rayPipeline->createShaderTable();

    if (!m_shaderTable)
    {
        log::fatal("Failed to create Z shader table");
    }

    m_shaderTable->setRayGenerationShader("RayGen");
    // The shared TLAS produces hit-group indices 0..3 (subd hit/shadow = 0/1,
    // cluster-LOD hit/shadow = 2/3); every index must have a record.  Depth-only
    // shading is geometry-agnostic, so all four map to the same ZHitGroup.
    m_shaderTable->addHitGroup("ZHitGroup");
    m_shaderTable->addHitGroup("ZHitGroup");
    m_shaderTable->addHitGroup("ZHitGroup");
    m_shaderTable->addHitGroup("ZHitGroup");
    m_shaderTable->addMissShader("Miss");
}

void ZRenderer::Render(Camera& camera, nvrhi::rt::AccelStructHandle tlas, 
    nvrhi::ITexture* zbuffer, nvrhi::ICommandList* commandList)
{
    if (!m_rayPipeline)
    {
        BuildPipeline(commandList->getDevice());
    }

    nvrhi::utils::ScopedMarker marker(commandList, "Z Render Pass");
    
    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_params),
        nvrhi::BindingSetItem::RayTracingAccelStruct(0, tlas),
        nvrhi::BindingSetItem::Texture_UAV(0, zbuffer),
    };

    m_bindingSet =
        commandList->getDevice()->createBindingSet(bindingSetDesc, m_bindingLayout);

    ZRenderParams params;
    auto const& [u, v, w] = camera.GetBasis();

    params.eye = camera.GetEye();

    params.U = u;
    params.V = v;
    params.W = w;

    commandList->writeBuffer(m_params, &params, sizeof(params));

    nvrhi::rt::State state;
    state.shaderTable = m_shaderTable;
    state.bindings = { m_bindingSet };
    commandList->setRayTracingState(state);

    nvrhi::rt::DispatchRaysArguments args;
    args.width = zbuffer->getDesc().width;
    args.height = zbuffer->getDesc().height;

    stats::frameSamplers.zRenderPassTime.Start(commandList);
    commandList->dispatchRays(args);
    stats::frameSamplers.zRenderPassTime.Stop();
}

