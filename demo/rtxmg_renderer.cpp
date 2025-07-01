//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "rtxmg_renderer.h"
#include "rtxmg/utils/buffer.h"
#include "ray_payload.h"
#include "render_targets.h"

#include <donut/app/ApplicationBase.h>
#include <donut/app/StreamlineInterface.h>
#include <donut/core/math/math.h>
#include <donut/engine/CommonRenderPasses.h>
#include <nvrhi/utils.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <sstream>

using namespace donut;
using namespace donut::math;

#include "lighting_cb.h"
#include "gbuffer.h"

#include "rtxmg/utils/debug.h"
#include "rtxmg/scene/scene.h"
#include "rtxmg/scene/camera.h"
#include "rtxmg/cluster_builder/fill_instance_descs_params.h"
#include "rtxmg/profiler/statistics.h"
#include "rtxmg/subdivision/subdivision_surface.h"
#include "envmap/preprocess_envmap_params.h"

RTXMGRenderer::RTXMGRenderer(Options const& opts)
    : m_options(opts), m_params(opts.params)
{
    std::filesystem::path frameworkShaderPath =
        app::GetDirectoryWithExecutable() / "shaders/framework" /
        app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

    std::filesystem::path appShaderPath =
        app::GetDirectoryWithExecutable() / "shaders/rtxmg_demo" /
        app::GetShaderTypeName(GetDevice()->getGraphicsAPI()) / "shaders";

    auto fs = std::make_shared<vfs::RootFileSystem>();

    auto mount = [&fs, this](std::string const& dir, std::string const& alias = "")
        {
            std::filesystem::path shaderPath =
                app::GetDirectoryWithExecutable() / "shaders" / dir /
                app::GetShaderTypeName(GetDevice()->getGraphicsAPI()) / "shaders";

            std::string aliasStr = (alias.empty() ? dir : alias);
            fs->mount(std::format("/shaders/{}", aliasStr), shaderPath.string());
        };

    mount("rtxmg_demo");
    mount("cluster_builder");
    mount("envmap");
    mount("hiz");
    mount("subdivision");

    fs->mount("/shaders/donut", frameworkShaderPath);

    m_shaderFactory =
        std::make_shared<engine::ShaderFactory>(GetDevice(), fs, "/shaders");
    m_commonPasses = std::make_shared<engine::CommonRenderPasses>(
        GetDevice(), m_shaderFactory);

    m_bindingCache = std::make_unique<engine::BindingCache>(GetDevice());

    nvrhi::BindlessLayoutDesc bindlessLayoutDesc;
    bindlessLayoutDesc.visibility = nvrhi::ShaderType::All;
    bindlessLayoutDesc.firstSlot = 0;
    bindlessLayoutDesc.maxCapacity = 1024;
    bindlessLayoutDesc.registerSpaces = {
        nvrhi::BindingLayoutItem::RawBuffer_SRV(1),
        nvrhi::BindingLayoutItem::Texture_SRV(2) };
    m_bindlessLayout = GetDevice()->createBindlessLayout(bindlessLayoutDesc);

    nvrhi::BindingLayoutDesc globalBindingLayoutDesc;
    globalBindingLayoutDesc.visibility = nvrhi::ShaderType::All;
    globalBindingLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0), // lighting constants
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(1), // render parameters
        nvrhi::BindingLayoutItem::RayTracingAccelStruct(0),  // TLAS
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),   // instance buffer
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),   // geometry buffer
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3),   // material buffer
        nvrhi::BindingLayoutItem::Texture_SRV(4),            // env map
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(5),   // env map conditional CDF
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(6),   // env map marginal CDF
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(7),   // env map conditional Func
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(8),   // env map marginal Func
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(9),  // cluster shading data
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(10),  // cluster vertex positions
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(11),  // topology quality
        nvrhi::BindingLayoutItem::Sampler(0),                // linear wrap
        nvrhi::BindingLayoutItem::Texture_UAV(0),            // accum
        nvrhi::BindingLayoutItem::Texture_UAV(1),            // depth
        nvrhi::BindingLayoutItem::Texture_UAV(2),            // normal
        nvrhi::BindingLayoutItem::Texture_UAV(3),            // albedo
        nvrhi::BindingLayoutItem::Texture_UAV(4),            // specular
        nvrhi::BindingLayoutItem::Texture_UAV(5),            // specular hitT
        nvrhi::BindingLayoutItem::Texture_UAV(6),            // roughness
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(7),   // hit result

        // DEBUG
#if ENABLE_DUMP_FLOAT
        nvrhi::BindingLayoutItem::Texture_UAV(8),        // debug
        nvrhi::BindingLayoutItem::Texture_UAV(9),       // debug
        nvrhi::BindingLayoutItem::Texture_UAV(10),       // debug
        nvrhi::BindingLayoutItem::Texture_UAV(11),       // debug
#endif
#if ENABLE_PIXEL_DEBUG
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(12),  // pixel debug buffer
#endif
        nvrhi::BindingLayoutItem::TypedBuffer_UAV(13),       // Timeview buffer
        nvrhi::BindingLayoutItem::TypedBuffer_UAV(RTXMG_NVAPI_SHADER_EXT_SLOT), // for nvidia extensions
    };
    m_bindingLayout = GetDevice()->createBindingLayout(globalBindingLayoutDesc);

    m_descriptorTable = std::make_shared<engine::DescriptorTableManager>(
        GetDevice(), m_bindlessLayout);

    m_lightingConstantsBuffer =
        GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
            sizeof(LightingConstants), "LightingConstants",
            engine::c_MaxRenderPassConstantBufferVersions));

    m_renderParamsBuffer =
        GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
            sizeof(RenderParams), "RenderParams",
            engine::c_MaxRenderPassConstantBufferVersions));

    m_fillInstanceDescsParams =
        GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
            sizeof(FillInstanceDescsParams), "FillInstanceDescsParams",
            engine::c_MaxRenderPassConstantBufferVersions));

    m_timeViewBuffer = CreateBuffer(2, sizeof(uint32_t), "TimeViewBuffer", GetDevice(), nvrhi::Format::R32_UINT);

    auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
    m_textureCache = std::make_shared<engine::TextureCache>(GetDevice(), nativeFS,
        m_descriptorTable);

    m_blitParamsBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(BlitParams), "BlitParams",
        engine::c_MaxRenderPassConstantBufferVersions));

    m_preprocessEnvMapResources.m_params = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(PreprocessEnvMapParams), "PreprocessEnvMapParams", engine::c_MaxRenderPassConstantBufferVersions));

    m_scanSystem.Init(m_shaderFactory, GetDevice());
}

RTXMGRenderer::~RTXMGRenderer() {}

void RTXMGRenderer::ReloadShaders()
{
    m_shaderFactory->ClearCache();

    m_pipelinesNeedsUpdate = true;
    m_needsRebind = true;
    
    m_clusterAccelBuilder = std::make_unique<ClusterAccelBuilder>(*m_shaderFactory, m_commonPasses, GetDescriptorTable()->GetDescriptorTable(), GetDevice());
    m_sceneAccels = std::make_unique<ClusterAccels>();
    m_zbuffer.reset();

    m_scanSystem.Init(m_shaderFactory, GetDevice());
    if (m_envMap)
    {
        m_needsEnvMapUpdate = true;
    }

    for (auto& pso : m_motionVectorsPSO)
    {
        pso.Reset();
    }
    m_blitPipeline.Reset();
    m_preprocessEnvMapShaders.m_computeConditionalPSO.Reset();
    m_preprocessEnvMapShaders.m_computeMarginalPSO.Reset();
    m_fillInstanceDescsPSO.Reset();
}

void RTXMGRenderer::BuildOrUpdatePipelines()
{
    if (m_pipelinesNeedsUpdate)
    {
        if (!CreateRayTracingPipeline(*m_shaderFactory))
        {
            log::fatal("Failed to create ray tracing pipeline");
        }
        ResetSubframes();
        m_pipelinesNeedsUpdate = false;
    }
}

void RTXMGRenderer::ComputeMotionVectors(nvrhi::ICommandList* commandList)
{
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_renderParamsBuffer))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_outputTextures[uint32_t(OutputTexture::Depth)]))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_hitResultBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_subdInstancesBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_scene->GetInstanceBuffer()))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_scene->GetGeometryBuffer()))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(5, m_scene->GetMaterialBuffer()))
        .addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_outputTextures[uint32_t(OutputTexture::MotionVectors)]))
#if ENABLE_PIXEL_DEBUG
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_motionVectorsPixelDebugBuffer))
#endif
        .addItem(nvrhi::BindingSetItem::Sampler(0, m_commonPasses->m_LinearWrapSampler));


    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_motionVectorsBL, bindingSet))
    {
        log::fatal("Failed to create binding set and layout for motion_vectors.hlsl");
    }

    nvrhi::ComputePipelineHandle& motionVectorsPSO = m_motionVectorsPSO[uint32_t(m_mvecDisplacement)];
    if (!motionVectorsPSO)
    {
        std::vector<donut::engine::ShaderMacro> macros;
        macros.push_back(donut::engine::ShaderMacro("MVEC_DISPLACEMENT", 
            m_mvecDisplacement == MvecDisplacement::FromSubdEval ? "MVEC_DISPLACEMENT_FROM_SUBD_EVAL" : "MVEC_DISPLACEMENT_FROM_MATERIAL"));
        nvrhi::ShaderHandle shader = m_shaderFactory->CreateShader("rtxmg_demo/motion_vectors.hlsl", "main", &macros, nvrhi::ShaderType::Compute);

        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(shader)
            .addBindingLayout(m_motionVectorsBL)
            .addBindingLayout(m_bindlessLayout);
        
        motionVectorsPSO = GetDevice()->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(motionVectorsPSO)
        .addBindingSet(bindingSet)
        .addBindingSet(m_descriptorTable->GetDescriptorTable());

    commandList->setComputeState(state);
    commandList->dispatch(div_ceil(m_renderSize.x, kMotionVectorsNumThreadsY), div_ceil(m_renderSize.y, kMotionVectorsNumThreadsY), 1);
}

void RTXMGRenderer::DlssUpscale(nvrhi::ICommandList *commandList, uint32_t frameIndex)
{
#if DONUT_WITH_STREAMLINE
    stats::frameSamplers.gpuDenoiserTime.Start(commandList);
    using StreamlineInterface = donut::app::StreamlineInterface;

    if (m_params.denoiserMode == DenoiserMode::None)
        return;

    StreamlineInterface& streamline = donut::app::DeviceManager::GetStreamline();
    const uint32_t kViewportId = 0;
    streamline.SetViewport(kViewportId);

    // SET STREAMLINE CONSTANTS
    {
        // This section of code updates the streamline constants every frame. Regardless of whether we are utilising the streamline plugins, as long as streamline is in use, we must set its constants.
        affine3 viewReprojection = m_view.GetInverseViewMatrix() * m_viewPrevious.GetViewMatrix();
        float4x4 reprojectionMatrix = m_view.GetInverseProjectionMatrix(false) * affineToHomogeneous(viewReprojection) * m_viewPrevious.GetProjectionMatrix(false);
        float aspectRatio = float(m_renderSize.x) / float(m_renderSize.y);
        
        float2 jitterOffset = m_view.GetPixelOffset();

        StreamlineInterface::Constants slConstants = {};
        slConstants.cameraAspectRatio = aspectRatio;
        slConstants.cameraFOV = dm::radians(m_camera.GetFovY());
        slConstants.cameraFar = m_camera.GetZFar();
        slConstants.cameraMotionIncluded = true;
        slConstants.cameraNear = m_camera.GetZNear();
        slConstants.cameraPinholeOffset = { 0.f, 0.f };
        slConstants.cameraPos = m_view.GetInverseViewMatrix().m_translation;
        slConstants.cameraFwd = m_view.GetInverseViewMatrix().m_linear[2];
        slConstants.cameraUp = m_view.GetInverseViewMatrix().m_linear[1];
        slConstants.cameraRight = m_view.GetInverseViewMatrix().m_linear[0];
        slConstants.cameraViewToClip = m_view.GetProjectionMatrix(false);
        slConstants.clipToCameraView = m_view.GetInverseProjectionMatrix(false);
        slConstants.clipToPrevClip = reprojectionMatrix;
        slConstants.prevClipToClip = inverse(reprojectionMatrix);
        slConstants.depthInverted = m_view.IsReverseDepth();
        slConstants.jitterOffset = -jitterOffset; // Jitter application to primary rays is negated relative to DLSS expectations.
        slConstants.mvecScale = { 1.0f / m_renderSize.x , 1.0f / m_renderSize.y }; // This are scale factors used to normalize mvec (to -1,1) and donut has mvec in pixel space
        slConstants.reset = m_resetDenoiser;
        slConstants.motionVectors3D = false;
        slConstants.motionVectorsInvalidValue = FLT_MIN;

        streamline.SetConstants(slConstants);
    }

    streamline.TagResourcesGeneral(commandList,
       m_view.GetChildView(ViewType::PLANAR, 0),
       m_outputTextures[uint32_t(OutputTexture::MotionVectors)],
       m_outputTextures[uint32_t(OutputTexture::Depth)],
       m_outputTextures[uint32_t(OutputTexture::Accumulation)]);

    if (m_params.denoiserMode == DenoiserMode::DlssSr)
    {
        streamline.TagResourcesDLSSNIS(commandList,
            m_view.GetChildView(ViewType::PLANAR, 0),
            m_outputTextures[uint32_t(OutputTexture::DlssOutputColor)],
            m_outputTextures[uint32_t(OutputTexture::Accumulation)]);

        streamline.EvaluateDLSS(commandList);
    } 
    else if (m_params.denoiserMode == DenoiserMode::DlssRr)
    {
        streamline.TagResourcesDLSSRR(commandList,
            m_view.GetChildView(ViewType::PLANAR, 0),
            m_renderSize,
            m_displaySize,
            m_outputTextures[uint32_t(OutputTexture::Accumulation)],
            m_outputTextures[uint32_t(OutputTexture::Albedo)],
            m_outputTextures[uint32_t(OutputTexture::Specular)],
            m_outputTextures[uint32_t(OutputTexture::Normals)],
            m_outputTextures[uint32_t(OutputTexture::Roughness)],
            m_outputTextures[uint32_t(OutputTexture::SpecularHitT)],
            nullptr,
            m_outputTextures[uint32_t(OutputTexture::DlssOutputColor)]);

        streamline.EvaluateDLSSRR(commandList);
    }
    stats::frameSamplers.gpuDenoiserTime.Stop();
#endif

    m_resetDenoiser = false;
}

void RTXMGRenderer::CreateOutputs(nvrhi::ICommandList *commandList)
{ 
    auto UpdateTexture = [this](nvrhi::TextureHandle& handle, int2 size, nvrhi::Format format, const char* debugName)
        {
            if (handle && handle->getDesc().width == size.x && handle->getDesc().height == size.y)
                return;

            nvrhi::TextureDesc desc;
            desc.width = size.x;
            desc.height = size.y;
            desc.isUAV = true;
            desc.keepInitialState = true;
            desc.format = format;
            desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            desc.debugName = debugName;
            handle = GetDevice()->createTexture(desc);

            m_bindingSet = nullptr;
        };

    auto UpdateRenderTexture = [&UpdateTexture, this](nvrhi::TextureHandle& handle, nvrhi::Format format, const char* debugName)
        {
            UpdateTexture(handle, m_renderSize, format, debugName);
        };
    auto UpdateDisplayTexture = [&UpdateTexture, this](nvrhi::TextureHandle& handle, nvrhi::Format format, const char* debugName)
        {
            UpdateTexture(handle, m_displaySize, format, debugName);
        };

    auto UpdateRenderBuffer = [this](nvrhi::BufferHandle& handle, size_t elementSize, nvrhi::Format format, const char* debugName)
        {
            size_t newSize = m_renderSize.x * m_renderSize.y * elementSize;
            if (handle && handle->getDesc().byteSize == newSize)
                return;

            nvrhi::BufferDesc bufferDesc =
                nvrhi::BufferDesc()
                .setByteSize(m_renderSize.x * m_renderSize.y * elementSize)
                .setCanHaveTypedViews(true)
                .setCanHaveUAVs(true)
                .setFormat(format)
                .setStructStride(uint32_t(elementSize))
                .setDebugName(debugName)
                .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
                .setKeepInitialState(true);

            handle = GetDevice()->createBuffer(bufferDesc);

            m_bindingSet = nullptr;
        };

    UpdateDisplayTexture(m_outputTextures[uint32_t(OutputTexture::DlssOutputColor)], nvrhi::Format::RGBA16_FLOAT, "DlssOutputColor");    
    UpdateRenderTexture(m_outputTextures[uint32_t(OutputTexture::Accumulation)], nvrhi::Format::RGBA32_FLOAT, "Accum");
    UpdateRenderTexture(m_outputTextures[uint32_t(OutputTexture::Depth)], nvrhi::Format::R32_FLOAT, "Depth");
    UpdateRenderTexture(m_outputTextures[uint32_t(OutputTexture::Normals)], nvrhi::Format::RGBA16_FLOAT, "Normals");
    UpdateRenderTexture(m_outputTextures[uint32_t(OutputTexture::Albedo)], nvrhi::Format::RGBA8_UNORM, "Albedo");
    UpdateRenderTexture(m_outputTextures[uint32_t(OutputTexture::Specular)], nvrhi::Format::RGBA8_UNORM, "Specular");
    UpdateRenderTexture(m_outputTextures[uint32_t(OutputTexture::SpecularHitT)], nvrhi::Format::R32_FLOAT, "SpecularHitT");
    UpdateRenderTexture(m_outputTextures[uint32_t(OutputTexture::Roughness)], nvrhi::Format::R8_UNORM, "Roughness");
    UpdateRenderTexture(m_outputTextures[uint32_t(OutputTexture::MotionVectors)], nvrhi::Format::RG16_FLOAT, "MotionVectors");
    UpdateRenderBuffer(m_hitResultBuffer, sizeof(HitResult), nvrhi::Format::UNKNOWN, "HitResult");

    UpdateDisplayTexture(m_displayTexture, nvrhi::Format::RGBA8_UNORM, "Display");

#if ENABLE_PIXEL_DEBUG    
    if (!m_pixelDebugBuffer.GetBuffer())
    {
        m_pixelDebugBuffer.Create(64, "PixelDebugBuffer", GetDevice());
        m_motionVectorsPixelDebugBuffer.Create(64, "MotionVectorPixelDebugBuffer", GetDevice());
    }
#endif

#if ENABLE_DUMP_FLOAT
    if (!m_outputTextures[index(OutputTexture::Debug1)])
    {
        static_assert(index(OutputTexture::Debug1) < index(OutputTexture::Debug4));
        for (size_t i = index(OutputTexture::Debug1); i <= index(OutputTexture::Debug4); i++)
        {
            std::string debugName = "Debug Texture " + std::to_string(i - index(OutputTexture::Debug1) + 1);
            UpdateRenderTexture(m_outputTextures[i], nvrhi::Format::RGBA16_FLOAT, debugName.c_str());
        }
    }
#endif


    if (!m_dummyBuffer)
    {
        m_dummyBuffer = CreateAndUploadBuffer(std::vector<float>{0.f}, "DummyBuffer", commandList, nvrhi::Format::R32_FLOAT);
    }

    if (!m_zbuffer)
    {
        m_zbuffer = ZBuffer::Create(uint2(m_renderSize.x, m_renderSize.y), m_commonPasses, m_shaderFactory, commandList);
    }
}

void RTXMGRenderer::Launch(nvrhi::ICommandList* commandList,
    uint32_t frameIndex,
    std::shared_ptr<engine::Light> light)
{
    CreateOutputs(commandList);

    BuildOrUpdatePipelines();

    if (m_needsEnvMapUpdate)
    {
        UpdateEnvMapSampling(commandList);
        m_needsEnvMapUpdate = false;
    }

    if (!m_bindingSet || m_needsRebind)
    {
        nvrhi::BufferHandle conditionalCDF = m_envMap ?
            m_preprocessEnvMapResources.m_conditionalCdf :
            m_dummyBuffer;
        nvrhi::BufferHandle marginalCDF = m_envMap ?
            m_preprocessEnvMapResources.m_marginalCdf :
            m_dummyBuffer;
        nvrhi::BufferHandle conditionalFunc = m_envMap ?
            m_preprocessEnvMapResources.m_conditionalFunc :
            m_dummyBuffer;
        nvrhi::BufferHandle marginalFunc = m_envMap ?
            m_preprocessEnvMapResources.m_marginalFunc :
            m_dummyBuffer;

        m_needsRebind = false;
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_lightingConstantsBuffer),
            nvrhi::BindingSetItem::ConstantBuffer(1, m_renderParamsBuffer),
            nvrhi::BindingSetItem::RayTracingAccelStruct(0, m_topLevelAS),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_scene->GetInstanceBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_scene->GetGeometryBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_scene->GetMaterialBuffer()),
            nvrhi::BindingSetItem::Texture_SRV(4, m_envMap ? m_envMap->texture : m_commonPasses->m_BlackTexture),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(5, conditionalCDF),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(6, marginalCDF),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(7, conditionalFunc),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(8, marginalFunc),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(9, m_sceneAccels->clusterShadingDataBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(10, m_sceneAccels->clusterVertexPositionsBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(11, m_subdInstancesBuffer),
            nvrhi::BindingSetItem::Sampler(0, m_commonPasses->m_LinearWrapSampler),
            nvrhi::BindingSetItem::Texture_UAV(0, m_outputTextures[uint32_t(OutputTexture::Accumulation)]),
            nvrhi::BindingSetItem::Texture_UAV(1, m_outputTextures[uint32_t(OutputTexture::Depth)]),
            nvrhi::BindingSetItem::Texture_UAV(2, m_outputTextures[uint32_t(OutputTexture::Normals)]),
            nvrhi::BindingSetItem::Texture_UAV(3, m_outputTextures[uint32_t(OutputTexture::Albedo)]),
            nvrhi::BindingSetItem::Texture_UAV(4, m_outputTextures[uint32_t(OutputTexture::Specular)]),
            nvrhi::BindingSetItem::Texture_UAV(5, m_outputTextures[uint32_t(OutputTexture::SpecularHitT)]),
            nvrhi::BindingSetItem::Texture_UAV(6, m_outputTextures[uint32_t(OutputTexture::Roughness)]),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(7, m_hitResultBuffer),
        #if ENABLE_DUMP_FLOAT
            nvrhi::BindingSetItem::Texture_UAV(8, m_outputTextures[index(OutputTexture::Debug1)]),
            nvrhi::BindingSetItem::Texture_UAV(9, m_outputTextures[index(OutputTexture::Debug2)]),
            nvrhi::BindingSetItem::Texture_UAV(10, m_outputTextures[index(OutputTexture::Debug3)]),
            nvrhi::BindingSetItem::Texture_UAV(11, m_outputTextures[index(OutputTexture::Debug4)]),
        #endif
        #if ENABLE_PIXEL_DEBUG
            nvrhi::BindingSetItem::StructuredBuffer_UAV(12, m_pixelDebugBuffer),
        #endif
            nvrhi::BindingSetItem::TypedBuffer_UAV(13, m_timeViewBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(RTXMG_NVAPI_SHADER_EXT_SLOT, nullptr) // for nvidia extensions
        };

        m_bindingSet =
            GetDevice()->createBindingSet(bindingSetDesc, m_bindingLayout);
    }

    {
        nvrhi::utils::ScopedMarker marker(commandList, "Ray Tracing Pass");

        LightingConstants constants = {};
        constants.ambientColor = float4(0.05f);

        light->FillLightConstants(constants.light);
        commandList->writeBuffer(m_lightingConstantsBuffer, &constants, sizeof(constants));

        RenderParams params = m_params;
        // Override settings
        params.colorMode = m_showMicroTriangles ? ColorMode::COLOR_BY_MICROTRI_ID : m_colorMode;
        params.shadingMode = m_showMicroTriangles ? ShadingMode::PRIMARY_RAYS : m_shadingMode;
        commandList->writeBuffer(m_renderParamsBuffer, &params, sizeof(params));

        nvrhi::rt::State state;
        state.shaderTable = m_shaderTable;
        state.bindings = { m_bindingSet, m_descriptorTable->GetDescriptorTable() };
        commandList->setRayTracingState(state);

        nvrhi::rt::DispatchRaysArguments args;
        args.width = m_renderSize.x;
        args.height = m_renderSize.y;

        stats::frameSamplers.gpuRenderTime.Start(commandList);
        commandList->dispatchRays(args);
        stats::frameSamplers.gpuRenderTime.Stop();
    }

    // Motion vectors
    {
        nvrhi::utils::ScopedMarker marker(commandList, "Motion Vectors");
        stats::frameSamplers.computeMotionVectorsTimer.Start(commandList);
        ComputeMotionVectors(commandList);
        stats::frameSamplers.computeMotionVectorsTimer.Stop();
    }

    if (m_displayZBuffer && m_zbuffer)
    {
        m_zbuffer->Display(m_outputTextures[uint32_t(OutputTexture::Accumulation)], commandList);
    }

    ++m_params.subFrameIndex;
}

void RTXMGRenderer::BlitFramebuffer(nvrhi::ICommandList* commandList, nvrhi::IFramebuffer* framebuffer)
{
    nvrhi::utils::ScopedMarker marker(commandList, "Blit");

    BlitParams blitParams;
    blitParams.m_blitDecodeMode = BlitDecodeMode::None;
    blitParams.m_tonemapOperator = m_showMicroTriangles ? TonemapOperator::Linear : m_tonemapOperator;
    blitParams.m_exposure = m_showMicroTriangles ? 1.0f : m_exposure;
    blitParams.m_zNear = m_camera.GetZNear();
    blitParams.m_zFar = m_camera.GetZFar();
    blitParams.m_separator = m_params.denoiserMode != DenoiserMode::None ? m_denoiserSeparator : 1.0f;

    nvrhi::ITexture* outputTexture = nullptr;

    nvrhi::ITexture* denoisedOutput = m_outputTextures[uint32_t(OutputTexture::DlssOutputColor)];

    Output outputIndex = m_outputIndex;
    switch (outputIndex)
    {
    case Output::DlssOutputColor:
    case Output::Accumulation:
    case Output::Albedo:
    case Output::Specular:
    default:
        blitParams.m_blitDecodeMode = BlitDecodeMode::None;
        break;
    case Output::Depth:
    case Output::SpecularHitT:
        blitParams.m_blitDecodeMode = BlitDecodeMode::Depth;
        break;
    case Output::Normals:
        blitParams.m_blitDecodeMode = BlitDecodeMode::Normals;
        break;
    case Output::Roughness:
        blitParams.m_blitDecodeMode = BlitDecodeMode::SingleChannel;
        break;
    case Output::MotionVectors:
        blitParams.m_blitDecodeMode = BlitDecodeMode::MotionVectors;
        break;
    case Output::InstanceId:
        blitParams.m_blitDecodeMode = BlitDecodeMode::InstanceId;
        break;
    case Output::SurfaceIndex:
        blitParams.m_blitDecodeMode = BlitDecodeMode::SurfaceIndex;
        break;
    case Output::SurfaceUv:
        blitParams.m_blitDecodeMode = BlitDecodeMode::SurfaceUv;
        break;
    case Output::Texcoord:
        blitParams.m_blitDecodeMode = BlitDecodeMode::Texcoord;
        break;
    case Output::HiZ:
        outputIndex = Output::Accumulation;
        blitParams.m_blitDecodeMode = BlitDecodeMode::Depth;
        break;
    }

    OutputTexture outputTextureIndex = uint32_t(outputIndex) < uint32_t(OutputTexture::Count) ?
        OutputTexture(uint32_t(outputIndex)) :
        OutputTexture::Accumulation;

    outputTexture = m_outputTextures[uint32_t(outputTextureIndex)];
    
    commandList->writeBuffer(m_blitParamsBuffer, &blitParams, sizeof(blitParams));

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_blitParamsBuffer))
        .addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_displayTexture))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, outputTexture))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(1, denoisedOutput))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_hitResultBuffer))
        .addItem(nvrhi::BindingSetItem::Sampler(0, m_commonPasses->m_LinearClampSampler));

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_blitBL, bindingSet))
    {
        log::fatal("Failed to create binding set for blit");
    }

    if (!m_blitPipeline)
    {
        nvrhi::ShaderHandle shader = m_shaderFactory->CreateShader("rtxmg_demo/blit.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

        auto pipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(shader)
            .addBindingLayout(m_blitBL);

        m_blitPipeline = GetDevice()->createComputePipeline(pipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_blitPipeline)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    const int blockSize = 32;

    commandList->dispatch(div_ceil(m_displayTexture->getDesc().width, blockSize), div_ceil(m_displayTexture->getDesc().height, blockSize));

    donut::engine::BlitParameters params =
    {
        .targetFramebuffer = framebuffer,
        .sourceTexture = m_displayTexture,
    };
    GetCommonPasses()->BlitTexture(commandList, params, m_bindingCache.get());
}

void RTXMGRenderer::ResetSubframes() 
{ 
    if (m_params.enableTimeView != 0 || m_params.denoiserMode == DenoiserMode::None)
    {
        // The denoiser really doesn't like it when its noise is related between frames.
        // Since we don't feed the accumulation buffer to the denoiser, it won't matter
        // if we reset the subframe index or not.
        m_params.subFrameIndex = 0;
    }
}

static float VanDerCorput(size_t base, size_t index)
{
    float ret = 0.0f;
    float denominator = float(base);
    while (index > 0)
    {
        size_t multiplier = index % base;
        ret += float(multiplier) / denominator;
        index = index / base;
        denominator *= base;
    }
    return ret;
}

void RTXMGRenderer::SetRenderCamera(Camera& camera, bool isCameraCut)
{
    donut::math::float3 eye = camera.GetEye();
    donut::math::float3 direction = camera.GetDirection();
    if (!all(isfinite(eye)) || !all(isfinite(direction)))
    {
        donut::log::error("Camera contains NaNs!: (%f %f %f) -> (%f %f %f)", eye.x, eye.y, eye.z, direction.x, direction.y, direction.z);
        return;
    }

    const uint32_t kBasePhaseCount = 8;
    uint32_t phaseCount = uint32_t(std::ceil(kBasePhaseCount * powf(float(m_displaySize.y) / float(m_renderSize.y), 2.0f)));
    uint32_t index = (m_params.subFrameIndex % phaseCount) + 1;
    m_params.jitter = float2{ VanDerCorput(2, index), VanDerCorput(3, index) } - 0.5f;

    m_previousCamera = m_camera;
    m_viewPrevious = m_view;

    m_camera = camera;

    float4x4 viewMatrixRowMajor = transpose(m_camera.GetViewMatrix());
    float4x4 projectionRowMajor = transpose(m_camera.GetProjectionMatrix());

    m_view.SetViewport(nvrhi::Viewport(float(m_renderSize.x), float(m_renderSize.y)));
    m_view.SetMatrices(homogeneousToAffine(viewMatrixRowMajor), projectionRowMajor);
    m_view.SetPixelOffset(m_params.jitter);
    m_view.UpdateCache();
    
    if (isCameraCut)
    {
        m_previousCamera = m_camera;
        m_viewPrevious = m_view;
        ResetDenoiser();
    }

    // Assumes that prev camera has the same output resolution
    auto MakeCameraConstants = [this](const Camera& camera)
        {
            return CameraConstants{
                camera.GetViewMatrix(),
                inverse(camera.GetViewMatrix()),
                camera.GetProjectionMatrix(),
                inverse(camera.GetProjectionMatrix()),
                float2(float(m_renderSize.x), float(m_renderSize.y)),
                float2(1.0f / m_renderSize.x, 1.0f / m_renderSize.y) };
        };

    auto const& [u, v, w] = camera.GetBasis();

    if (any(eye != m_params.eye) ||
         any(u != m_params.U) ||
         any(v != m_params.V) ||
         any(w != m_params.W))
    {
        ResetSubframes();
    }

    m_params.camera = MakeCameraConstants(m_camera);
    m_params.prevCamera = MakeCameraConstants(m_previousCamera);
    m_params.zFar = m_camera.GetZFar();
    m_params.eye = eye;
    m_params.U = u;
    m_params.V = v;
    m_params.W = w;
    m_params.viewProjectionMatrix = camera.GetViewProjectionMatrix();
}

void RTXMGRenderer::SetRenderSize(int2 renderSize, int2 displaySize)
{
    bool renderSizeChanged = any(renderSize != m_renderSize);
    bool displaySizeChanged = any(displaySize != m_displaySize);

    m_renderSize = renderSize;
    m_displaySize = displaySize;
    
    if (renderSizeChanged || displaySizeChanged)
    {
        m_zbuffer = nullptr;
        m_bindingCache->Clear();
        ResetSubframes();
        ResetDenoiser();
        GetDevice()->waitForIdle(); // About to free render buffers
    }
}

void RTXMGRenderer::SetTimeView(bool timeView)
{
    m_params.enableTimeView = timeView;
    ResetSubframes();
    ResetDenoiser();
}

void RTXMGRenderer::SceneFinishedLoading(std::shared_ptr<RTXMGScene> scene)
{
    m_bindingSet = nullptr;
    m_bindingCache->Clear();

    m_scene = scene;
    CreateAccelStructs();
    ResetSubframes();
    ResetDenoiser();
}

bool RTXMGRenderer::CreateRayTracingPipeline(
    engine::ShaderFactory& shaderFactory)
{
    m_shaderLibrary =
        shaderFactory.CreateShaderLibrary("rtxmg_demo/rtxmg_demo_path_tracer.hlsl", nullptr);

    if (!m_shaderLibrary)
        return false;

    nvrhi::rt::PipelineDesc pipelineDesc;
    pipelineDesc.globalBindingLayouts = { m_bindingLayout, m_bindlessLayout };
    pipelineDesc.shaders =
    {
        {"",m_shaderLibrary->getShader("RayGen", nvrhi::ShaderType::RayGeneration),nullptr},
        {"", m_shaderLibrary->getShader("Miss", nvrhi::ShaderType::Miss),nullptr},
        {"", m_shaderLibrary->getShader("ShadowMiss", nvrhi::ShaderType::Miss), nullptr}
    };

    pipelineDesc.hitGroups =
    { {
        "HitGroup",
        m_shaderLibrary->getShader("ClosestHit", nvrhi::ShaderType::ClosestHit),
        nullptr, // m_ShaderLibrary->getShader("AnyHit", nvrhi::ShaderType::AnyHit),
        nullptr, // intersectionShader
        nullptr, // bindingLayout
        false    // isProceduralPrimitive
    },
    {
        "ShadowHitGroup",
        nullptr, // closestHitShader
        nullptr, // anyHitShader
        nullptr, // intersectionShader
        nullptr, // bindingLayout
        false    // isProceduralPrimitive
    } };

    pipelineDesc.maxPayloadSize = sizeof(RayPayload);
    pipelineDesc.maxRecursionDepth = m_params.ptMaxBounces + 1;
    pipelineDesc.hlslExtensionsUAV = int32_t(RTXMG_NVAPI_SHADER_EXT_SLOT);

    m_rayPipeline = GetDevice()->createRayTracingPipeline(pipelineDesc);

    if (!m_rayPipeline)
        return false;

    m_shaderTable = m_rayPipeline->createShaderTable();

    if (!m_shaderTable)
        return false;

    m_shaderTable->setRayGenerationShader("RayGen");
    m_shaderTable->addHitGroup("HitGroup");
    m_shaderTable->addHitGroup("ShadowHitGroup");
    m_shaderTable->addMissShader("Miss");
    m_shaderTable->addMissShader("ShadowMiss");

    return true;
}

void RTXMGRenderer::CreateAccelStructs()
{
    nvrhi::rt::AccelStructDesc tlasDesc;
    tlasDesc.isTopLevel = true;
    tlasDesc.topLevelMaxInstances =
        m_scene->GetSceneGraph()->GetMeshInstances().size();
    m_topLevelAS = GetDevice()->createAccelStruct(tlasDesc);

    m_clusterAccelBuilder = std::make_unique<ClusterAccelBuilder>(*m_shaderFactory, m_commonPasses, GetDescriptorTable()->GetDescriptorTable(), GetDevice());
    m_sceneAccels = std::make_unique<ClusterAccels>();
}

void RTXMGRenderer::UpdateEnvMapTransform()
{
    affine3 Ry = rotation(float3(0, 1, 0), m_environmentMapAzimuth);
    affine3 Rx = rotation(float3(1, 0, 0), m_environmentMapElevation);
    m_params.envmapRotation = affineToHomogeneous(Ry * Rx);
    m_params.envmapRotationInv = transpose(m_params.envmapRotation);
}

void RTXMGRenderer::SetEnvMap(const std::string& filePath, nvrhi::ICommandList* commandList)
{
    m_needsRebind = true;
    m_needsEnvMapUpdate = true;

    auto existing = m_textureCache->GetLoadedTexture(filePath);
    if (existing)
    {
        m_envMap = existing;
    }
    else
    {
        m_envMap = m_textureCache->LoadTextureFromFile(filePath, false, m_commonPasses.get(), commandList);
    }
}

void RTXMGRenderer::UpdateEnvMapSampling(nvrhi::ICommandList* commandList)
{
    // allocate importance sampling buffers

    bool dumpIntermediateResults = false;

    nvrhi::utils::ScopedMarker marker(commandList, "RTXMGScene::UpdateEnvMapSampling");

    const uint32_t inputWidth = m_envMap->texture->getDesc().width;
    const uint32_t inputHeight = m_envMap->texture->getDesc().height;

    // CDFs are two wider than their function, this gives room to store the integral in the final position
    m_preprocessEnvMapResources.m_conditionalFunc.Create(inputWidth * inputHeight, "Conditional Func", GetDevice(), nvrhi::Format::R32_FLOAT);
    m_preprocessEnvMapResources.m_conditionalCdf.Create((inputWidth + 2) * inputHeight, "Conditional CDF", GetDevice(), nvrhi::Format::R32_FLOAT);
    m_preprocessEnvMapResources.m_marginalFunc.Create(inputHeight, "Marginal Func", GetDevice(), nvrhi::Format::R32_FLOAT);
    m_preprocessEnvMapResources.m_marginalCdf.Create(inputHeight + 2, "Marginal CDF", GetDevice(), nvrhi::Format::R32_FLOAT);

    m_preprocessEnvMapResources.m_sampler = m_commonPasses->m_LinearClampSampler;

    PreprocessEnvMapParams params;
    params.envMapHeight = inputHeight;
    params.envMapWidth = inputWidth;

    commandList->writeBuffer(m_preprocessEnvMapResources.m_params, &params, sizeof(params));

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_envMap->texture))
        .addItem(nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_preprocessEnvMapResources.m_conditionalFunc))
        .addItem(nvrhi::BindingSetItem::TypedBuffer_UAV(1, m_preprocessEnvMapResources.m_conditionalCdf))
        .addItem(nvrhi::BindingSetItem::TypedBuffer_UAV(2, m_preprocessEnvMapResources.m_marginalFunc))
        .addItem(nvrhi::BindingSetItem::TypedBuffer_UAV(3, m_preprocessEnvMapResources.m_marginalCdf))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_preprocessEnvMapResources.m_params))
        .addItem(nvrhi::BindingSetItem::Sampler(0, m_preprocessEnvMapResources.m_sampler));

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_preprocessEnvMapShaders.m_bindingLayout, bindingSet))
    {
        log::fatal("Failed to create binding set and layout for preprocess envmap shaders");
    }

    auto runShader = [&params, &dumpIntermediateResults, &bindingSet, &commandList, this](nvrhi::ComputePipelineHandle &computePipeline, 
        const char *shaderPath, const char *entryPointName, uint32_t x, uint32_t y)
        {
            if (!computePipeline)
            {
                auto shader = m_shaderFactory->CreateShader(shaderPath, entryPointName, nullptr, nvrhi::ShaderType::Compute);

                auto computePipelineDesc = nvrhi::ComputePipelineDesc()
                    .setComputeShader(shader)
                    .addBindingLayout(m_preprocessEnvMapShaders.m_bindingLayout);

                computePipeline = GetDevice()->createComputePipeline(computePipelineDesc);
            }

            // dumping the results will close and reopen the command list, so we need to re-write
            // the params buffer
            if (dumpIntermediateResults)
                commandList->writeBuffer(m_preprocessEnvMapResources.m_params, &params, sizeof(params));

            auto state = nvrhi::ComputeState()
                .setPipeline(computePipeline)
                .addBindingSet(bindingSet);

            commandList->setComputeState(state);
            commandList->dispatch(x, y);
        };

    // step 1: convert input image to luminance, store in conditionalFunc
    {
        nvrhi::utils::ScopedMarker marker(commandList, "Compute Conditional Func");
        runShader(m_preprocessEnvMapShaders.m_computeConditionalPSO, 
            "envmap/compute_conditional.hlsl", "main", div_ceil(inputWidth, 16), div_ceil(inputHeight, 16));
    }

    if (dumpIntermediateResults)
        WriteBufferToCSV(commandList, m_preprocessEnvMapResources.m_conditionalFunc, "01_conditional_func.csv", inputWidth, inputHeight);

    // step 2: compute the conditional CDF using a prefix scan
    {
        nvrhi::utils::ScopedMarker marker(commandList, "prefix scan conditional CDF");
        m_scanSystem.PrefixScan(m_preprocessEnvMapResources.m_conditionalFunc, m_preprocessEnvMapResources.m_conditionalCdf, inputWidth, inputHeight, commandList);
    }

    if (dumpIntermediateResults)
        WriteBufferToCSV(commandList, m_preprocessEnvMapResources.m_conditionalCdf, "02_conditional_cdf.csv", inputWidth + 2, inputHeight);
    else
        nvrhi::utils::BufferUavBarrier(commandList, m_preprocessEnvMapResources.m_conditionalCdf);

    // step 3: Copy the CDF integrals to the marginal func.
    {
        nvrhi::utils::ScopedMarker marker(commandList, "Compute  Marginal Func");
        runShader(m_preprocessEnvMapShaders.m_computeMarginalPSO,
            "envmap/compute_marginal.hlsl", "main", 1, div_ceil(inputHeight, 32));
    }

    if (dumpIntermediateResults)
        WriteBufferToCSV(commandList, m_preprocessEnvMapResources.m_marginalFunc, "03_marginal_func.csv", inputHeight, 1);

    // step 4: Compute the marginal CDF using a prefix scan
    {
        nvrhi::utils::ScopedMarker marker(commandList, "prefix scan marginal CDF");
        m_scanSystem.PrefixScan(m_preprocessEnvMapResources.m_marginalFunc, m_preprocessEnvMapResources.m_marginalCdf, inputHeight, 1, commandList);
    }

    if (dumpIntermediateResults)
        WriteBufferToCSV(commandList, m_preprocessEnvMapResources.m_marginalCdf, "04_marginal_cdf.csv", inputHeight + 2, 1);
}

void RTXMGRenderer::FillInstanceDescs(nvrhi::ICommandList* commandList, nvrhi::IBuffer* outInstanceDescs, nvrhi::IBuffer* blasAddresses, uint32_t numInstances)
{
    assert(outInstanceDescs->getDesc().byteSize / sizeof(nvrhi::rt::IndirectInstanceDesc) >= numInstances);
    assert(blasAddresses->getDesc().byteSize / sizeof(nvrhi::GpuVirtualAddress) >= numInstances);

    FillInstanceDescsParams params = {};
    params.numInstances = numInstances;
    commandList->writeBuffer(m_fillInstanceDescsParams, &params, sizeof(params));

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, blasAddresses))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, outInstanceDescs))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_fillInstanceDescsParams));

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_fillInstanceDescsBL, bindingSet))
    {
        log::fatal("Failed to create binding set and layout for fill_instance_descs.hlsl");
    }

    if (!m_fillInstanceDescsPSO)
    {
        nvrhi::ShaderHandle shader = m_shaderFactory->CreateShader("cluster_builder/fill_instance_descs.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(shader)
            .addBindingLayout(m_fillInstanceDescsBL);

        m_fillInstanceDescsPSO = GetDevice()->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_fillInstanceDescsPSO)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    commandList->dispatch(div_ceil(numInstances, kFillInstanceDescsThreads), 1, 1);
}

void RTXMGRenderer::UpdateAccelerationStructures(const TessellatorConfig &tessConfig,
    ClusterStatistics& buildStats,
    uint32_t frameIndex,
    nvrhi::ICommandList* commandList)
{
    if (!m_scene->GetSubdMeshes().empty())
    {
        m_clusterAccelBuilder->BuildAccel(*m_scene, tessConfig, *m_sceneAccels, buildStats, frameIndex, commandList);
        m_needsRebind = true;
    }
    
    m_scene->Refresh(commandList, frameIndex);

    std::vector<nvrhi::rt::InstanceDesc> instances;
    std::vector<SubdInstance> subdInstances;

    for (const auto& instance : m_scene->GetSubdMeshInstances())
    {
        auto donutInstance = instance.meshInstance;
        auto& mesh = m_scene->GetSubdMeshes()[instance.meshID];

        unsigned int writeDepthFlag = !(mesh->HasAnimation());

        nvrhi::rt::InstanceDesc instanceDesc;
        instanceDesc.blasDeviceAddress = 0; // will get filled out later by fill instance desc indirect arg
        instanceDesc.instanceMask = (writeDepthFlag << 1) | 0x1;
        instanceDesc.instanceID = donutInstance->GetInstanceIndex();
        instanceDesc.instanceContributionToHitGroupIndex = 0;

        auto node = donutInstance->GetNode();
        assert(node);
        dm::affineToColumnMajor(node->GetLocalToWorldTransformFloat(),
            instanceDesc.transform);

        instances.push_back(instanceDesc);

        auto getDescriptorHeapIndex = [](const DescriptorHandle& descriptor) -> uint32_t
            {
                return descriptor.IsValid() ? uint32_t(descriptor.GetIndexInHeap()) : kInvalidBindlessIndex;
            };

        SubdInstance subdInstance;
        subdInstance.plansBindlessIndex = getDescriptorHeapIndex(mesh->GetTopologyMap()->plansDescriptor);
        subdInstance.stencilMatrixBindlessIndex = getDescriptorHeapIndex(mesh->GetTopologyMap()->stencilMatrixDescriptor);
        subdInstance.subpatchTreesBindlessIndex = getDescriptorHeapIndex(mesh->GetTopologyMap()->subpatchTreesDescriptor);
        subdInstance.patchPointIndicesBindlessIndex = getDescriptorHeapIndex(mesh->GetTopologyMap()->patchPointIndicesDescriptor);

        subdInstance.vertexSurfaceDescriptorBindlessIndex = getDescriptorHeapIndex(mesh->m_vertexSurfaceDescriptorDescriptor);
        subdInstance.vertexControlPointIndicesBindlessIndex = getDescriptorHeapIndex(mesh->m_vertexControlPointIndicesDescriptor);
        subdInstance.positionsBindlessIndex = getDescriptorHeapIndex(mesh->m_positionsDescriptor);
        subdInstance.positionsPrevBindlessIndex = getDescriptorHeapIndex(mesh->m_positionsPrevDescriptor);
        subdInstance.surfaceToGeometryIndexBindlessIndex = getDescriptorHeapIndex(mesh->m_surfaceToGeometryIndexDescriptor);
        subdInstance.topologyQualityBindlessIndex = getDescriptorHeapIndex(mesh->m_topologyQualityDescriptor);
        subdInstance.isolationLevel = mesh->m_dynamicIsolationLevel;

        affineToColumnMajor(node->GetPrevLocalToWorldTransformFloat(), subdInstance.prevLocalToWorld);
        affineToColumnMajor(inverse(node->GetLocalToWorldTransformFloat()), subdInstance.worldToLocal);
        subdInstances.push_back(subdInstance);
    }

    uint32_t numInstances = uint32_t(instances.size());

    if (!m_instanceDescs.GetBuffer() || m_instanceDescs.GetNumElements() != numInstances)
    {
        nvrhi::BufferDesc instanceDescsDesc =
        {
            .byteSize = numInstances * sizeof(instances[0]),
            .structStride = sizeof(instances[0]),
            .debugName = "TLAS InstanceDescs",
            .canHaveUAVs = true,
            .isAccelStructBuildInput = true,
            .initialState = nvrhi::ResourceStates::AccelStructBuildInput,
            .keepInitialState = true,
        };

        m_instanceDescs.Create(instanceDescsDesc, GetDevice());
        m_subdInstancesBuffer.Create(numInstances, "SubdInstances", GetDevice());
    }

    if (m_subdInstancesBuffer.GetNumElements() > subdInstances.size())
    {
        // Clear existing elements with default entries
        subdInstances.resize(m_subdInstancesBuffer.GetNumElements());
    }

    m_instanceDescs.Upload(instances, commandList);
    m_subdInstancesBuffer.Upload(subdInstances, commandList);

    // Patch the instance desc buffer with BLAS pointers
    nvrhi::IBuffer* blasAddresses = m_sceneAccels->blasPtrsBuffer;
    if (blasAddresses != nullptr)
    {
        FillInstanceDescs(commandList, m_instanceDescs, blasAddresses, numInstances);
    }

    nvrhi::utils::ScopedMarker marker(commandList, "TLAS Update");
    commandList->buildTopLevelAccelStructFromBuffer(m_topLevelAS, m_instanceDescs, 0u, numInstances);
}

void RTXMGRenderer::DumpPixelDebugBuffers(nvrhi::ICommandList* commandList)
{
#if ENABLE_PIXEL_DEBUG
    auto logLambda = [](std::ostream& ss, const PixelDebugElement& e)
        {
            if (e.payloadType == PixelDebugElement::PayloadType_None)
                return false;

            ss << "[Line:" << std::dec << e.lineNumber << "] ";

            if (e.payloadType >= PixelDebugElement::PayloadType_Float &&
                e.payloadType <= PixelDebugElement::PayloadType_Float4)
            {
                uint32_t numVectorElements = (e.payloadType - uint32_t(PixelDebugElement::PayloadType_Float)) + 1;

                ss << std::setprecision(12) << e.floatData.data()[0];
                for (uint32_t i = 1; i < numVectorElements; i++)
                    ss << ", " << e.floatData.data()[i];
            }
            else if (e.payloadType >= PixelDebugElement::PayloadType_Int &&
                e.payloadType <= PixelDebugElement::PayloadType_Int4)
            {
                uint32_t numVectorElements = (e.payloadType - uint32_t(PixelDebugElement::PayloadType_Int)) + 1;
                ss << std::dec << static_cast<int>(e.uintData.data()[0]) << std::hex << "(0x" << e.uintData.data()[0] << ")";
                for (uint32_t i = 1; i < numVectorElements; i++)
                    ss << ", " << std::dec << static_cast<int>(e.uintData.data()[i]) << std::hex << "(0x" << e.uintData.data()[i] << ")";
            }
            else if (e.payloadType >= PixelDebugElement::PayloadType_Uint &&
                e.payloadType <= PixelDebugElement::PayloadType_Uint4)
            {
                uint32_t numVectorElements = (e.payloadType - uint32_t(PixelDebugElement::PayloadType_Uint)) + 1;
                ss << std::dec << e.uintData.data()[0] << std::hex << "(0x" << e.uintData.data()[0] << ")";
                for (uint32_t i = 1; i < numVectorElements; i++)
                    ss << ", " << std::dec << e.uintData.data()[i] << std::hex << "(0x" << e.uintData.data()[i] << ")";
            }

            return true;
        };


    log::info("Raytracing Pixel Debug: %d, %d", m_params.debugPixel.x, m_params.debugPixel.y);
    m_pixelDebugBuffer.Log(commandList, logLambda, { .wrap = false, .header = false, .elementIndex = false });

    log::info("Motion Vector Pixel Debug: %d, %d", m_params.debugPixel.x, m_params.debugPixel.y);
    m_motionVectorsPixelDebugBuffer.Log(commandList, logLambda, { .wrap = false, .header = false, .elementIndex = false });
#endif
}
