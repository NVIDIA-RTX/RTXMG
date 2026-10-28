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

#include "rtxmg_renderer.h"
#include "rtxmg/utils/buffer.h"
#include "ray_payload.h"
#include "render_targets.h"

#include <donut/app/ApplicationBase.h>
#include <donut/app/StreamlineInterface.h>
#include <donut/core/math/math.h>
#include <donut/engine/CommonRenderPasses.h>
#include <nvrhi/utils.h>

#include <filesystem>
#include <string>
#include <sstream>

using namespace donut;
using namespace donut::math;

#include "lighting_cb.h"
#include "gbuffer.h"

#include "rtxmg/utils/bindless_layout.h"
#include "rtxmg/utils/cluster_operation.h"
#include "rtxmg/utils/debug.h"
#include "rtxmg/utils/vram_bytes.h"
#include "rtxmg/scene/scene.h"
#include "rtxmg/scene/camera.h"
#include "rtxmg/cluster_lod/baker.h"
#include "rtxmg/cluster_lod/preloaded.h"
#include "rtxmg/cluster_lod/streaming.h"
#include "rtxmg/cluster_lod/streaming_utils.h"
#include "rtxmg/cluster_tess/fill_instance_descs_params.h"
#include "rtxmg/profiler/statistics.h"
#include "rtxmg/subdivision/subdivision_surface.h"
#include "envmap/preprocess_envmap_params.h"

RTXMGRenderer::RTXMGRenderer(Options const& opts)
    : m_options(opts), m_params(opts.params)
{
    m_hlslExtensionsSupported = GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12 &&
        GetDevice()->queryFeatureSupport(nvrhi::Feature::HlslExtensionUAV);

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
    mount("cluster_tess");
    mount("cluster_lod");
    mount("envmap");
    mount("hiz");
    mount("subdivision");

    fs->mount("/shaders/donut", frameworkShaderPath);

    m_shaderFactory =
        std::make_shared<engine::ShaderFactory>(GetDevice(), fs, "/shaders");
    m_commonPasses = std::make_shared<engine::CommonRenderPasses>(
        GetDevice(), m_shaderFactory);

    m_bindingCache = std::make_unique<engine::BindingCache>(GetDevice());

    // Every pass binding this table must use this same desc: on Vulkan they have
    // to be layout-compatible, and VK sizes the mutable set at maxCapacity up
    // front and can never grow it.
    m_bindlessLayout = GetDevice()->createBindlessLayout(rtxmg::MakeGlobalBindlessLayoutDesc());

    nvrhi::BindingLayoutDesc globalBindingLayoutDesc;
    globalBindingLayoutDesc.visibility = nvrhi::ShaderType::All;
    globalBindingLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0), // lighting constants
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(1), // render parameters
        nvrhi::BindingLayoutItem::RayTracingAccelStruct(0),  // TLAS
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),   // instance buffer
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),   // material buffer
        nvrhi::BindingLayoutItem::Texture_SRV(3),            // env map
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(4),   // env map conditional CDF
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(5),   // env map marginal CDF
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(6),   // env map conditional Func
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(7),   // env map marginal Func
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(8),   // cluster shading data
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(9),   // cluster vertex positions
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(10),  // cluster vertex normals
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(11),  // subd instances
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(12),  // cluster LOD render instances
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(13),  // cluster LOD geometries
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(14),  // cluster LOD resident clusters
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(15),  // cluster LOD local→global material indirection
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
#if ENABLE_SHADER_DEBUG
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(12),  // pixel debug buffer
#endif
        nvrhi::BindingLayoutItem::TypedBuffer_UAV(13),       // Timeview buffer
#if ENABLE_PIXEL_PICK
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(14),  // viewport pick result
#endif
    };

    if (GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
    {
        // for nvidia extensions
        globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::TypedBuffer_UAV(RTXMG_NVAPI_SHADER_EXT_SLOT)); 
    }
    
    m_bindingLayout = GetDevice()->createBindingLayout(globalBindingLayoutDesc);

    m_descriptorTable = std::make_shared<engine::DescriptorTableManager>(
        GetDevice(), m_bindlessLayout);

    // Reserve before ANY descriptor is created: donut grows the table on demand
    // and a grow relocates it in the heap, staling the absolute heap indices the
    // cluster-LOD path bakes into GPU buffers at scene init.
    m_descriptorTable->ReserveCapacity(rtxmg::kGlobalBindlessCapacity);
    // Tripwire baseline: a later capacity change means the table relocated —
    // checked once per frame in UpdateAccelerationStructures.
    m_reservedDescriptorCapacity =
        m_descriptorTable->GetDescriptorTable()->getCapacity();

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
    // The per-texture "Loaded W x H" message fires from inside every async decode
    // task and the log sink is mutex-serialized, so at Info it throttles decoding
    // on large scenes.
    m_textureCache->SetInfoLogSeverity(donut::log::Severity::Debug);

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

    m_needsRebind = true;
    
    // Clear all ray tracing permutations
    for (auto& pipeline : m_rayPipelines)
    {
        pipeline.Reset();
    }
    for (auto& shaderTable : m_shaderTables)
    {
        shaderTable.Reset();
    }
    
    m_clusterAccelBuilder = std::make_unique<ClusterTessellator>(*m_shaderFactory, m_commonPasses, GetDescriptorTable()->GetDescriptorTable(), GetDevice());
    m_sceneAccels = std::make_unique<ClusterTessAccels>();
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
    // Rebuilt lazily on next blit (picks up any donut tone-map shader edits).
    m_toneMappingPass.reset();
    m_preprocessEnvMapShaders.m_computeConditionalPSO.Reset();
    m_preprocessEnvMapShaders.m_computeMarginalPSO.Reset();
    m_fillInstanceDescsPSO.Reset();
}



void RTXMGRenderer::ComputeMotionVectors(nvrhi::ICommandList* commandList)
{
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_renderParamsBuffer))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_outputTextures[uint32_t(OutputTexture::Depth)]))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_hitResultBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_subdInstancesBuffer.GetBuffer()
            ? static_cast<nvrhi::IBuffer*>(m_subdInstancesBuffer) : m_dummyBuffer.Get()))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_scene->GetMaterialBuffer()))
        .addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_outputTextures[uint32_t(OutputTexture::MotionVectors)]))
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
    ScopedGPUTimer timer(stats::frameSamplers.gpuDenoiserTime, commandList);

#if DONUT_WITH_STREAMLINE
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

#if ENABLE_SHADER_DEBUG
    if (!m_pixelDebugBuffer.GetBuffer())
        m_pixelDebugBuffer.Create(64, "PixelDebugBuffer", GetDevice());
#endif
#if ENABLE_PIXEL_PICK
    if (!m_pixelPickBuffer.GetBuffer())
        m_pixelPickBuffer.Create(1, "PixelPickBuffer", GetDevice());
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

uint64_t RTXMGRenderer::GetRenderTargetBytes() const
{
    nvrhi::IDevice* device = GetDevice();
    uint64_t bytes = 0;
    for (const nvrhi::TextureHandle& tex : m_outputTextures)
        bytes += rtxmg::TextureVramBytes(device, tex);
    bytes += rtxmg::TextureVramBytes(device, m_displayTexture);

    bytes += rtxmg::BufferVramBytes(m_hitResultBuffer);
    bytes += rtxmg::BufferVramBytes(m_timeViewBuffer);

    if (m_zbuffer)
    {
        bytes += rtxmg::TextureVramBytes(device, m_zbuffer->GetCurrent());
        for (int level = 0; level < m_zbuffer->GetNumHiZLODs(); ++level)
            bytes += rtxmg::TextureVramBytes(device, m_zbuffer->GetHierarchyTexture(uint32_t(level)));
    }
    return bytes;
}

uint64_t RTXMGRenderer::GetEnvMapBytes() const
{
    if (!m_envMap || !m_envMap->texture)
        return 0;
    const PreprocessEnvMapResources& pre = m_preprocessEnvMapResources;
    return rtxmg::TextureVramBytes(GetDevice(), m_envMap->texture)
         + pre.m_conditionalFunc.GetVramBytes() + pre.m_conditionalCdf.GetVramBytes()
         + pre.m_marginalFunc.GetVramBytes() + pre.m_marginalCdf.GetVramBytes();
}

uint64_t RTXMGRenderer::GetTlasBytes() const
{
    uint64_t bytes = m_instanceDescs.GetVramBytes();
    if (m_topLevelAS)
        bytes += GetDevice()->getAccelStructMemoryRequirements(m_topLevelAS).size;
    return bytes;
}

uint64_t RTXMGRenderer::GetClusterTessWorkingBytes() const
{
    return m_clusterAccelBuilder && m_sceneAccels ? m_clusterAccelBuilder->GetWorkingBufferBytes(*m_sceneAccels) : 0;
}

void RTXMGRenderer::Launch(nvrhi::ICommandList* commandList,
    uint32_t frameIndex,
    std::shared_ptr<engine::Light> light)
{
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
            nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_scene->GetInstanceBuffer() ? m_scene->GetInstanceBuffer() : m_dummyBuffer.Get()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_scene->GetMaterialBuffer()),
            nvrhi::BindingSetItem::Texture_SRV(3, m_envMap ? m_envMap->texture : m_commonPasses->m_BlackTexture),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(4, conditionalCDF),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(5, marginalCDF),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(6, conditionalFunc),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(7, marginalFunc),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(8,  m_sceneAccels->clusterShadingDataBuffer    ? m_sceneAccels->clusterShadingDataBuffer    : m_dummyBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(9,  m_sceneAccels->clusterVertexPositionsBuffer ? m_sceneAccels->clusterVertexPositionsBuffer : m_dummyBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(10, m_sceneAccels->clusterVertexNormalsBuffer   ? m_sceneAccels->clusterVertexNormalsBuffer   : m_dummyBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(11, m_subdInstancesBuffer.GetBuffer()           ? m_subdInstancesBuffer.GetBuffer()           : m_dummyBuffer),
            // Cluster LOD hit shader bindings
            nvrhi::BindingSetItem::StructuredBuffer_SRV(12, m_clusterLodResources ? m_clusterLodResources->GetShaderRenderInstancesBuffer().GetBuffer().Get() : m_dummyBuffer.Get()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(13, m_clusterLodResources ? m_clusterLodResources->GetShaderGeometriesBuffer().GetBuffer().Get()       : m_dummyBuffer.Get()),
            // Scene-global resident cluster-address table.
            nvrhi::BindingSetItem::StructuredBuffer_SRV(14, m_clusterLodResources ? m_clusterLodResources->GetResidentClustersBuffer().GetBuffer().Get() : m_dummyBuffer.Get()),
            // cluster-LoD local→scene-global material
            // indirection. Empty/dummy buffer when there's no cluster-LoD scene.
            nvrhi::BindingSetItem::StructuredBuffer_SRV(15, m_scene->GetClusterLodLocalMaterialIDsBuffer() ? m_scene->GetClusterLodLocalMaterialIDsBuffer() : m_dummyBuffer.Get()),
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
        #if ENABLE_SHADER_DEBUG
            nvrhi::BindingSetItem::StructuredBuffer_UAV(12, m_pixelDebugBuffer),
        #endif
            nvrhi::BindingSetItem::TypedBuffer_UAV(13, m_timeViewBuffer),
        #if ENABLE_PIXEL_PICK
            nvrhi::BindingSetItem::StructuredBuffer_UAV(14, m_pixelPickBuffer),
        #endif
        };

        if (GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
        {
            bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::TypedBuffer_UAV(RTXMG_NVAPI_SHADER_EXT_SLOT, nullptr)); // for nvidia extensions
        }


        m_bindingSet =
            GetDevice()->createBindingSet(bindingSetDesc, m_bindingLayout);
    }

    {
#if ENABLE_SHADER_DEBUG
        commandList->clearBufferUInt(m_pixelDebugBuffer, 0);
#endif
#if ENABLE_PIXEL_PICK
        commandList->clearBufferUInt(m_pixelPickBuffer, 0);
#endif
        nvrhi::utils::ScopedMarker marker(commandList, "Ray Tracing Pass");

        LightingConstants constants = {};
        constants.ambientColor = float4(0.05f);

        light->FillLightConstants(constants.light);
        commandList->writeBuffer(m_lightingConstantsBuffer, &constants, sizeof(constants));

        RenderParams params = m_params;
        // Override settings
        params.colorMode = m_showMicroTriangles ? ColorMode::COLOR_BY_MICROTRI_ID : m_colorMode;
        params.shadingMode = m_showMicroTriangles ? ShadingMode::PRIMARY_RAYS : m_shadingMode;
        params.enableTimeView = GetTimeView();
        commandList->writeBuffer(m_renderParamsBuffer, &params, sizeof(params));

        // Get the appropriate pipeline and shader table for current permutation
        RayTracingPermutation permutation(m_enableVertexNormals, m_enableClusterLodVertexNormals,
                                          m_enableClusterLodNormalMaps);
        
        auto GetRayTracingShaderTable = [this](const RayTracingPermutation& rtPermutation) -> nvrhi::rt::IShaderTable*
            {
                uint32_t index = rtPermutation.index();
                if (!m_rayPipelines[index])
                {
                    // Create shader macros based on permutation
                    std::vector<engine::ShaderMacro> macros;
                    macros.push_back(engine::ShaderMacro("VERTEX_NORMALS", rtPermutation.isVertexNormalsEnabled() ? "1" : "0"));
                    macros.push_back(engine::ShaderMacro("CLUSTER_LOD_SHADING", std::to_string(rtPermutation.clusterLodShading())));

                    const std::string pathTracer = m_options.forceDxrClusterIntrinsics
                        ? "rtxmg_demo/rtxmg_demo_path_tracer_dxr.hlsl"
                        : GetClusterApiShaderPath(GetDevice(), "rtxmg_demo/rtxmg_demo_path_tracer.hlsl");
                    nvrhi::ShaderLibraryHandle shaderLibrary = m_shaderFactory->CreateShaderLibrary(pathTracer.c_str(), &macros);

                    if (!shaderLibrary)
                        return nullptr;

                    nvrhi::rt::PipelineDesc pipelineDesc;
                    pipelineDesc.globalBindingLayouts = { m_bindingLayout, m_bindlessLayout };
                    pipelineDesc.shaders =
                    {
                        {"", shaderLibrary->getShader("RayGen", nvrhi::ShaderType::RayGeneration), nullptr},
                        {"", shaderLibrary->getShader("Miss", nvrhi::ShaderType::Miss), nullptr},
                        {"", shaderLibrary->getShader("ShadowMiss", nvrhi::ShaderType::Miss), nullptr}
                    };

                    pipelineDesc.hitGroups =
                    { {   // index 0 — subd primary hit
                        "HitGroup",
                        shaderLibrary->getShader("ClosestHit", nvrhi::ShaderType::ClosestHit),
                        nullptr, nullptr, nullptr, false
                    },
                    {   // index 1 — subd shadow hit (null)
                        "ShadowHitGroup",
                        nullptr, nullptr, nullptr, nullptr, false
                    },
                    {   // index 2 — cluster LOD primary hit
                        "ClusterLodHitGroup",
                        shaderLibrary->getShader("ClusterLodClosestHit", nvrhi::ShaderType::ClosestHit),
                        // any-hit runs the alpha-mask cutoff / blend gather; the CLAS
                        // Opaque geometry flag keeps it off opaque triangles.
                        shaderLibrary->getShader("ClusterLodAnyHit", nvrhi::ShaderType::AnyHit),
                        nullptr, nullptr, false
                    },
                    {   // index 3 — cluster LOD shadow hit
                        "ClusterLodShadowHitGroup",
                        shaderLibrary->getShader("ClusterLodShadowClosestHit", nvrhi::ShaderType::ClosestHit),
                        // any-hit runs the alpha-mask cutoff and lets blend cards through
                        shaderLibrary->getShader("ClusterLodShadowAnyHit", nvrhi::ShaderType::AnyHit),
                        nullptr, nullptr, false
                    } };

                    pipelineDesc.maxPayloadSize = sizeof(RayPayload);
                    pipelineDesc.maxRecursionDepth = m_params.ptMaxBounces + 1;
                    pipelineDesc.allowClusters = true;
                    if (m_hlslExtensionsSupported)
                        pipelineDesc.hlslExtensionsUAV = int32_t(RTXMG_NVAPI_SHADER_EXT_SLOT);

                    m_rayPipelines[index] = GetDevice()->createRayTracingPipeline(pipelineDesc);

                    if (!m_rayPipelines[index])
                    {
                        if (UsesDxrClusterOperations(GetDevice()))
                            log::fatal("Failed to create the path tracing pipeline. Its DXR cluster shaders need "
                                       "NVIDIA driver 621.01 or later. Run with --cluster-api nvapi or -vk instead.");
                        return nullptr;
                    }

                    m_shaderTables[index] = m_rayPipelines[index]->createShaderTable();

                    if (!m_shaderTables[index])
                        return nullptr;

                    m_shaderTables[index]->setRayGenerationShader("RayGen");
                    m_shaderTables[index]->addHitGroup("HitGroup");
                    m_shaderTables[index]->addHitGroup("ShadowHitGroup");
                    m_shaderTables[index]->addHitGroup("ClusterLodHitGroup");
                    m_shaderTables[index]->addHitGroup("ClusterLodShadowHitGroup");
                    m_shaderTables[index]->addMissShader("Miss");
                    m_shaderTables[index]->addMissShader("ShadowMiss");
                }
                return m_shaderTables[index].Get();
            };

        nvrhi::rt::IShaderTable *shaderTable = GetRayTracingShaderTable(permutation);
        
        nvrhi::rt::State state;
        state.shaderTable = shaderTable;
        state.bindings = { m_bindingSet, m_descriptorTable->GetDescriptorTable() };
        commandList->setRayTracingState(state);

        nvrhi::rt::DispatchRaysArguments args;
        args.width = m_renderSize.x;
        args.height = m_renderSize.y;

        stats::frameSamplers.gpuRenderTime.Start(commandList);
        commandList->dispatchRays(args);
        stats::frameSamplers.gpuRenderTime.Stop();
    }

    // Motion vectors — run when the TLAS has been built (there is geometry to shade)
    if (m_topLevelAS)
    {
        nvrhi::utils::ScopedMarker marker(commandList, "Motion Vectors");
        ScopedGPUTimer timer(stats::frameSamplers.computeMotionVectorsTimer, commandList);
        ComputeMotionVectors(commandList);
    }

    if (m_displayZBuffer)
    {
        m_zbuffer->Display(m_outputTextures[uint32_t(OutputTexture::Accumulation)], commandList);
    }

    ++m_params.subFrameIndex;
}

void RTXMGRenderer::EnsureToneMappingPass(nvrhi::ICommandList* commandList)
{
    if (m_toneMappingPass)
        return;

    // Only the pass's histogram + exposure compute steps are used, but its ctor
    // still builds the (unused) tone-map render PSO and needs a framebuffer.
    m_toneMappingFbFactory = std::make_shared<FramebufferFactory>(GetDevice());
    m_toneMappingFbFactory->RenderTargets = { m_displayTexture };

    donut::render::ToneMappingPass::CreateParameters params;
    params.histogramBins = 256;
    m_toneMappingPass = std::make_unique<donut::render::ToneMappingPass>(
        GetDevice(), m_shaderFactory, m_commonPasses, m_toneMappingFbFactory,
        m_view, params);

    // Seed the adapted-luminance buffer with a mid value so the first auto-exposed
    // frame doesn't spike from an uninitialized readback.
    m_toneMappingPass->ResetExposure(commandList, 0.1f);
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

    // Auto-exposure is only meaningful for the tone-mapped HDR color outputs;
    // debug decode modes and the MicroTriangle overlay keep manual exposure.
    EnsureToneMappingPass(commandList);
    const bool autoExposureActive = m_autoExposure && !m_showMicroTriangles &&
        (outputIndex == Output::Accumulation || outputIndex == Output::DlssOutputColor);
    if (autoExposureActive)
    {
        // Derive adapted luminance from the render-res HDR accumulation (matches
        // m_view's dimensions and is valid whether or not DLSS is upscaling it).
        nvrhi::ITexture* hdrSource = m_outputTextures[uint32_t(OutputTexture::Accumulation)];
        m_toneMappingPass->AdvanceFrame(m_frameDeltaTime);
        m_toneMappingPass->ResetHistogram(commandList);
        m_toneMappingPass->AddFrameToHistogram(commandList, m_view, hdrSource);
        m_toneMappingPass->ComputeExposure(commandList, m_toneMappingParams);
    }
    blitParams.m_autoExposureEnabled = autoExposureActive ? 1u : 0u;
    // Match donut's exposure target: HDR is scaled by exp2(exposureBias)/adaptedLum.
    blitParams.m_autoExposureScale = ::exp2f(m_toneMappingParams.exposureBias);

    commandList->writeBuffer(m_blitParamsBuffer, &blitParams, sizeof(blitParams));

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_blitParamsBuffer))
        .addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_displayTexture))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, outputTexture))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(1, denoisedOutput))
        .addItem(nvrhi::BindingSetItem::TypedBuffer_SRV(2, m_toneMappingPass->GetExposureBuffer()))
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
    if (GetTimeView() || m_params.denoiserMode == DenoiserMode::None)
    {
        // The denoiser really doesn't like it when its noise is related between frames.
        // Since we don't feed the accumulation buffer to the denoiser, it won't matter
        // if we reset the subframe index or not.
        m_params.subFrameIndex = 0;
    }
}

// --shot-list only: with the denoiser on, ResetSubframes is a no-op, so the
// jitter phase at capture would depend on how many frames the scene load took.
// Pinning it makes a golden a function of the shot list alone.
void RTXMGRenderer::ForceResetSubframes()
{
    m_params.subFrameIndex = 0;
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

void RTXMGRenderer::SceneFinishedLoading(std::shared_ptr<RTXMGScene> scene,
                                          const ClusterLodBudgets& budgets,
                                          nvrhi::ICommandList* commandList)
{
    m_bindingSet = nullptr;
    m_bindingCache->Clear();
    m_scene = scene;
    m_clodBudgets = budgets;
    CreateAccelStructs(commandList);
    ResetSubframes();
    ResetDenoiser();
}

void RTXMGRenderer::ReinitAccelStructs(const ClusterLodBudgets& budgets, nvrhi::ICommandList* commandList)
{
    if (!m_scene)
        return;
    m_clodBudgets = budgets;
    // CreateAccelStructs rebuilds purely from m_scene: it releases the streaming
    // object (and its pools) and sizes a fresh one to the current StreamingConfig.
    m_bindingSet = nullptr;
    m_bindingCache->Clear();
    CreateAccelStructs(commandList);
    ResetSubframes();
    ResetDenoiser();
}

// Retire acceleration-structure memory that this frame is about to stop using.
//
// Must run at the top of the frame, before RenderHiZPrepass: the prepass traces
// the PREVIOUS frame's TLAS, which reaches the BLAS by raw GPU VA that nvrhi
// cannot track.  Freeing that memory after the prepass is recorded unmaps it out
// from under a trace already in the command list.  Anything retired here
// therefore invalidates the TLAS, which skips the prepass for one frame.
//
// tessConfig is null when no cluster_tess build follows this frame; its storage
// is then left alone, since the TLAS keeps pointing into it.
void RTXMGRenderer::RetireAccelResources(nvrhi::ICommandList* commandList,
                                         const TessellatorConfig* tessConfig)
{
    RetireClusterLodAccelResources(commandList);
    RetireClusterTessAccelResources(tessConfig);
}

void RTXMGRenderer::RetireClusterTessAccelResources(const TessellatorConfig* tessConfig)
{
    if (!tessConfig || !m_clusterAccelBuilder || !m_scene || !m_sceneAccels)
        return;

    if (m_clusterAccelBuilder->ResizeAccelStorage(*m_scene, *tessConfig, *m_sceneAccels))
        m_topLevelASBuilt = false;
}

void RTXMGRenderer::RetireClusterLodAccelResources(nvrhi::ICommandList* commandList)
{
    if (!m_clusterLodResources)
        return;

    // Caching ⊆ sharing, and it is incompatible with the --linearalloc compaction
    // allocator: compaction relocates resident CLAS on every unload defrag, which
    // leaves cached BLAS dangling at stale addresses (device-removed).
    if (m_useBlasCaching && m_useBlasSharing && m_useLinearClasAllocator)
    {
        if (!m_warnedLinearAllocCaching)
        {
            donut::log::warning("Cluster-LOD: BLAS caching disabled - incompatible with the "
                                "--linearalloc compaction allocator (it relocates CLAS, "
                                "invalidating cached BLAS).  Use the persistent allocator for caching.");
            m_warnedLinearAllocCaching = true;
        }
    }

    // On a toggle edge the cached-cluster budget reservation appears/disappears
    // but the cached BLAS persist, so traversal would overwrite clusters they
    // still reference.  Idle and reset.
    const bool useBlasCaching = EffectiveUseBlasCaching();
    if (useBlasCaching != m_clusterLodPrevUseBlasCaching)
    {
        GetDevice()->waitForIdle();
        if (auto* streaming = m_clusterLodResources->GetStreamingHooks())
            streaming->ResetCachedBlas(commandList);
        m_clusterLodPrevUseBlasCaching = useBlasCaching;
        m_topLevelASBuilt              = false;
    }
}

// Trace the PREVIOUS frame's TLAS from the CURRENT camera and reduce the
// pyramid BEFORE the accel update, so this frame culls against a current-view
// pyramid.  Tracing after the build would give stale previous-camera depth,
// which self-occlusion-flickers while the camera moves.  No-op when the TLAS is
// stale or its memory was just retired.
void RTXMGRenderer::RenderHiZPrepass(Camera& camera, nvrhi::ICommandList* commandList)
{
    if (!m_topLevelAS || !m_topLevelASBuilt || !m_zbuffer)
        return;

    if (!m_zRenderer)
        m_zRenderer = std::make_unique<ZRenderer>(GetShaderFactory());

    m_zRenderer->Render(camera, m_topLevelAS, m_zbuffer->GetCurrent(), commandList);
    m_zbuffer->ReduceHierarchy(commandList);
}

void RTXMGRenderer::CreateAccelStructs(nvrhi::ICommandList* commandList)
{
    uint32_t numSubd = static_cast<uint32_t>(m_scene->GetSubdMeshInstances().size());
    uint32_t numClusterLod = m_scene->HasClusterLod()
                        ? static_cast<uint32_t>(m_scene->GetClusterLodInstances().size())
                        : 0u;

    // Checked before cluster-LOD registration bakes absolute heap indices into
    // GPU buffers.  Six descriptors per geometry: 5 metadata handles + 1
    // low-detail group-data SRV; the constant slack covers subd and pool blocks.
    if (m_scene->HasClusterLod())
    {
        const auto [allocated, capacity] = m_descriptorTable->GetUsage();
        const uint32_t numClusterLodGeometries =
            static_cast<uint32_t>(m_scene->GetClusterLodGeometries().size());
        const uint32_t upcoming = numClusterLodGeometries * 6u + 256u;
        donut::log::info(
            "Descriptor budget: allocated=%u + upcoming(cluster-LOD)~%u of capacity=%u (%.0f%%)",
            allocated, upcoming, capacity,
            capacity ? 100.0 * double(allocated + upcoming) / double(capacity) : 0.0);
        if (allocated + upcoming > capacity)
        {
            donut::log::error(
                "Descriptor budget EXCEEDS the pinned reservation (%u + %u > %u): the "
                "table will grow and relocate mid-registration, staling every cached "
                "absolute heap index. Raise kGlobalBindlessCapacity above %u.",
                allocated, upcoming, capacity, allocated + upcoming);
        }
    }

    nvrhi::rt::AccelStructDesc tlasDesc;
    tlasDesc.isTopLevel = true;
    tlasDesc.topLevelMaxInstances = numSubd + numClusterLod;
    m_topLevelAS = GetDevice()->createAccelStruct(tlasDesc);
    m_topLevelASBuilt = false;  // fresh handle — HiZ prepass must not trace it yet

    m_clusterAccelBuilder = std::make_unique<ClusterTessellator>(*m_shaderFactory, m_commonPasses, GetDescriptorTable()->GetDescriptorTable(), GetDevice());
    m_sceneAccels = std::make_unique<ClusterTessAccels>();

    // Release any previous cluster LOD data, then rebuild for the new scene.
    m_clusterLodSystem.SetResources(nullptr);
    m_clusterLodResources = nullptr;

    if (!m_scene->HasClusterLod())
        return;

    // Sized unconditionally on the streaming path (0 on preloaded, which has no
    // caching) so BLAS caching stays a live toggle; the pool itself is lazy.
    uint64_t cachedBlasPoolBytes = 0;

    // --nomat lives on the scene, which is what actually built (or skipped) the
    // cluster-LoD materials this has to agree with.
    const bool enableMaterials = m_scene->GetEnableMaterials();

    if (m_useStreaming)
    {
        donut::log::info("Cluster-LOD: streaming residency path (default; --preload selects the preloaded path)");
        auto streaming = std::make_unique<ClusterLodStreaming>();
        streaming->SetDebugClusterLod(m_debugClusterLod);
        // The scene builds this metadata on the load thread (with loading-screen
        // progress); creating thousands of buffers here would stall the renderer.
        streaming->SetPrebuiltGeometryMetadata(&m_scene->GetClusterLodPrebuiltGeometryMetadata());
        streaming->SetClusterLodLocalMaterialsOffsets(&m_scene->GetClusterLodLocalMaterialsOffsets());

        // Map renderer CLI flags onto rtxmg::StreamingConfig.
        rtxmg::StreamingConfig streamingConfig;
        if (m_useLinearClasAllocator)
        {
            donut::log::info("Cluster-LOD: --linearalloc enabled (compaction CLAS allocator)");
            streamingConfig.usePersistentClasAllocator = false;
            if (m_clodBudgets.clasPoolOverrideMB != 0)
            {
                donut::log::info("Cluster-LOD: CLAS pool capped to %u MB (compaction budget test)", m_clodBudgets.clasPoolOverrideMB);
                streamingConfig.maxClasMegaBytes = m_clodBudgets.clasPoolOverrideMB;
            }
        }
        if (m_clodBudgets.maxFrameLoadRequests != 0)
        {
            donut::log::info("Cluster-LOD: maxPerFrameLoadRequests set to %u", m_clodBudgets.maxFrameLoadRequests);
            streamingConfig.maxPerFrameLoadRequests = m_clodBudgets.maxFrameLoadRequests;
        }
        // 0 keeps the StreamingConfig default.
        if (m_clodBudgets.maxResidentGroups != 0)
        {
            donut::log::info("Cluster-LOD: maxGroups (resident) set to %u", m_clodBudgets.maxResidentGroups);
            streamingConfig.maxGroups = m_clodBudgets.maxResidentGroups;
        }
        if (m_clodBudgets.maxGeometryMB != 0)
            streamingConfig.maxGeometryMegaBytes = m_clodBudgets.maxGeometryMB;
        if (m_clodBudgets.geometryBlockMB != 0)
            streamingConfig.geometryBlockMegaBytes = m_clodBudgets.geometryBlockMB;
        // Pool must be at least one block — clamp here so GetStreamingConfig()
        // (and thus the UI's "Geometry pool" minimum) reflects the effective value.
        streamingConfig.maxGeometryMegaBytes =
            std::max(streamingConfig.maxGeometryMegaBytes, streamingConfig.geometryBlockMegaBytes);
        if (m_clodBudgets.maxClasMB != 0)
            streamingConfig.maxClasMegaBytes = m_clodBudgets.maxClasMB;
        if (m_clodBudgets.maxBlasCachingMB != 0)
            streamingConfig.maxBlasCachingMegaBytes = m_clodBudgets.maxBlasCachingMB;
        streamingConfig.clasPositionTruncateBits = m_clasPositionTruncateBits;
        if (m_debugClusterLod)
        {
            donut::log::info("Cluster-LOD: --debug-clusterlod enabled (verbose streaming/traversal/BLAS diagnostics)");
            streamingConfig.debugClusterLod = true;
        }
        if (!enableMaterials)
        {
            donut::log::info("Cluster-LOD: --nomat enabled (forcing opaque/single-sided/material 0)");
            streamingConfig.enableMaterials = false;
        }
        // Under facet shading the hit shader never reads the packed normal words,
        // so don't keep them resident.  Flipping the toggle re-runs this init.
        streamingConfig.stripResidentNormals = !m_enableClusterLodVertexNormals;
        if (!m_stripResidentPositions)
        {
            // --nostrippos diagnostic: positions stay resident, normals still
            // follow the Vertex Normals toggle (the channels are independent).
            donut::log::info("Cluster-LOD: --nostrippos enabled (positions stay resident)");
            streamingConfig.stripResidentPositions = false;
        }
        if (!m_stripResidentData)
        {
            // --nostrip diagnostic: verbatim uploads (positions + normals stay
            // resident) — isolates the strip rewrite from bake/shader bugs.
            donut::log::info("Cluster-LOD: --nostrip enabled (verbatim resident blobs)");
            streamingConfig.stripResidentPositions = false;
            streamingConfig.stripResidentNormals   = false;
        }

        // The scene's own bake config: it feeds CLAS position truncation and the
        // cluster-per-group budget math, so a stub with default
        // compressionPosDropBits / clusterGroupSize would silently mis-size.
        const BakerConfig& bakerConfig = m_scene->GetClusterBakerConfig();
        // --nomat: pretend the scene has no alpha-mask materials, so the CLAS
        // build uses maxGeometryIndex=0 and skips the per-cluster geometry buffer.
        const bool hasAlphaMask = m_scene->HasClusterLodAlphaMask() && enableMaterials;
        streaming->SetEnableMaterials(enableMaterials);
        streaming->Init(
            m_scene->GetClusterLodGeometries(),
            m_scene->GetClusterLodInstances(),
            bakerConfig,
            m_scene->GetMaxClusterTriangles(),
            m_scene->GetMaxClusterVertices(),
            hasAlphaMask,                              // scene-wide alpha-mask / two-sided flag
            m_scene->GetClusterLodMaterialBaseID(),    // cluster-LoD material base ID
            streamingConfig,
            m_scene->GetDescriptorTable(),
            m_shaderFactory.get(),
            GetDevice(),
            commandList);

        // rtxmg is always ray-traced, so allocate the CLAS-side streaming
        // buffers unconditionally.
        streaming->UpdateClasRequired(true, commandList);

        // The BLAS pass's MOVE_OBJECTS scratch must match the streaming
        // cached-BLAS allocator's reservation.
        cachedBlasPoolBytes = uint64_t(streamingConfig.maxBlasCachingMegaBytes) * 1024ull * 1024ull;

        m_clusterLodResources = std::move(streaming);
    }
    else
    {
        auto preloaded = std::make_unique<ClusterLodPreloaded>();
        preloaded->SetDebugClusterLod(m_debugClusterLod);
        preloaded->SetEnableMaterials(enableMaterials);
        preloaded->SetPrebuiltGeometryMetadata(&m_scene->GetClusterLodPrebuiltGeometryMetadata());
        preloaded->SetClusterLodLocalMaterialsOffsets(&m_scene->GetClusterLodLocalMaterialsOffsets());
        if (!enableMaterials)
            donut::log::info("Cluster-LOD: --nomat enabled (forcing opaque/single-sided/material 0)");
        // --nomat: pretend the scene has no alpha-mask materials (see
        // streaming branch above for rationale).
        const bool hasAlphaMaskPreload = m_scene->HasClusterLodAlphaMask() && enableMaterials;
        preloaded->Init(
            m_scene->GetClusterLodGeometries(),
            m_scene->GetClusterLodInstances(),
            m_scene->GetClusterBakerConfig(),
            m_clasPositionTruncateBits,
            m_scene->GetClusterLodMaterialBaseID(),    // cluster-LoD material base ID
            hasAlphaMaskPreload,                       // alpha-mask preload
            m_scene->GetDescriptorTable(),
            GetDevice(),
            commandList);
        m_clusterLodResources = std::move(preloaded);
    }

    m_clusterLodSystem.SetResources(m_clusterLodResources.get());

    m_clusterLodSystem.GetPass().Init(*m_clusterLodResources,
                          m_descriptorTable->GetDescriptorTable(),
                          m_shaderFactory.get(),
                          GetDevice(),
                          m_debugClusterLod,
                          m_clodBudgets.renderClusterBits);

    // Reserved unconditionally on the streaming path (cheap: address arrays, not
    // the lazy pool) so caching stays a live sidebar toggle.
    const IClusterLodStreamingHooks* streamingHooks = m_clusterLodResources->GetStreamingHooks();
    const uint32_t maxCachedBlasBuilds =
        (m_useStreaming && streamingHooks) ? streamingHooks->GetMaxCachedBlasBuilds() : 0u;

    m_clusterLodSystem.GetBlasPass().Init(*m_clusterLodResources,
                               m_clusterLodSystem.GetPass(),
                               m_descriptorTable->GetDescriptorTable(),
                               m_shaderFactory.get(),
                               GetDevice(),
                               m_debugClusterLod,
                               maxCachedBlasBuilds,
                               cachedBlasPoolBytes);

    // m_instanceDescs is allocated per-frame in UpdateAccelerationStructures
    // (sized for numSubd + numClusterLod). Cluster-LOD entries are written there too.
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
        m_envMap = m_textureCache->LoadTextureFromFile(filePath, engine::TextureLoadOptions{ engine::SRGBMode::FromFile },
                                                       m_commonPasses.get(), commandList);
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

void RTXMGRenderer::FillInstanceDescs(nvrhi::ICommandList* commandList, nvrhi::IBuffer* outInstanceDescs, nvrhi::IBuffer* blasAddresses, uint32_t numInstances, uint32_t instanceOffset)
{
    FillInstanceDescsParams params = {};
    params.numInstances   = numInstances;
    params.instanceOffset = instanceOffset;
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
        nvrhi::ShaderHandle shader = m_shaderFactory->CreateShader("cluster_tess/fill_instance_descs.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

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

bool RTXMGRenderer::ShouldDebugClusterLodTlasFill()
{
    if (!m_debugClusterLod)
        return false;

    ++m_debugClusterLodTlasFillFrame;
    return m_debugClusterLodTlasFillFrame == 5;
}

void RTXMGRenderer::LogClusterLodTlasFillInputs(nvrhi::IBuffer* blasAddresses,
                                                uint32_t numInstances,
                                                uint32_t instanceOffset) const
{
    donut::log::info("ClusterLod TLAS fill input: blasAddressesBuffer=%p numClusterLodInstances=%u instanceOffset=%u",
                     (void*)blasAddresses,
                     numInstances,
                     instanceOffset);
}

void RTXMGRenderer::DebugReadbackClusterLodTlasFill(nvrhi::ICommandList* commandList,
                                                   uint32_t instanceOffset)
{
    const std::vector<nvrhi::rt::InstanceDesc> instanceDescs =
        m_instanceDescs.Download(commandList);
    if (instanceOffset >= instanceDescs.size())
        return;

    const nvrhi::rt::InstanceDesc& desc = instanceDescs[instanceOffset];
    donut::log::info("ClusterLod TLAS fill output instanceDescs[%u]: blasDeviceAddress=0x%llx instanceID=%u mask=0x%x",
                     instanceOffset,
                     (unsigned long long)desc.blasDeviceAddress,
                     desc.instanceID,
                     desc.instanceMask);
}

void RTXMGRenderer::UpdateAccelerationStructures(const TessellatorConfig &tessConfig,
    ClusterTessStatistics& buildStats,
    uint32_t frameIndex,
    nvrhi::ICommandList* commandList)
{
    m_enableVertexNormals = tessConfig.enableVertexNormals;
    m_enableClusterLodVertexNormals = tessConfig.enableClusterLodVertexNormals;

    // Descriptor-table relocation tripwire: a runtime grow stales every absolute
    // heap index baked into GPU buffers, and the downstream symptoms (black
    // geometry, phantom residency, CLAS corruption) never name the cause.
    {
        const uint32_t descCap = m_descriptorTable->GetDescriptorTable()->getCapacity();
        if (descCap != m_reservedDescriptorCapacity && !m_descriptorRelocationLogged)
        {
            m_descriptorRelocationLogged = true;
            donut::log::error(
                "Bindless descriptor table RESIZED underneath the renderer: %u -> %u. "
                "All cached absolute heap indices are stale; expect black geometry / "
                "streaming corruption. Raise kGlobalBindlessCapacity.",
                m_reservedDescriptorCapacity, descCap);
        }
    }

    // Drive show/hide of the AccelBuilder tab's Cluster Tess vs Cluster Lod
    // plot groups based on which paths actually run for this scene.
    stats::clusterAccelSamplers.hasClusterTess = !m_scene->GetSubdMeshes().empty();
    stats::clusterAccelSamplers.hasClusterLod  = m_scene->HasClusterLod();

    if (!m_scene->GetSubdMeshes().empty())
    {
        m_clusterAccelBuilder->BuildAccel(*m_scene, tessConfig, *m_sceneAccels, buildStats, frameIndex, commandList);
        m_needsRebind = true;
    }

    if (m_scene->HasClusterLod())
    {
        // "Update LoD Camera" can freeze this camera while the render camera
        // keeps moving — that lets LoD be inspected from a locked vantage.
        const Camera& lodCamera = tessConfig.camera ? *tessConfig.camera : m_camera;

        shaderio::SceneBuildingConstants lodConsts{};
        lodConsts.traversalViewMatrix       = lodCamera.GetViewMatrix();

        // Adaptive LoD error: the streaming budgets drive the effective pixel
        // error (see rtxmg::AdaptiveLodError).
        float lodPixelError = m_lodPixelError;
        const IClusterLodStreamingHooks* adaptiveHooks =
            (m_useStreaming && m_adaptiveLodError && m_clusterLodResources)
                ? m_clusterLodResources->GetStreamingHooks() : nullptr;
        if (adaptiveHooks)
        {
            const float loadFactor = adaptiveHooks->GetLoadFactor();
            m_adaptiveLodErrorController.Observe(loadFactor);

            if (m_debugClusterLod && (frameIndex % 30u) == 0u)
            {
                rtxmg::StreamingStats s;
                adaptiveHooks->GetStats(s);
                donut::log::info(
                    "adaptive-lpe frame=%u load=%.3f smoothed=%.3f lpe=%.3f "
                    "geo=%llu/%lluMB clas=%llu+%llu/%lluMB maxSizedLeft=%u/%u "
                    "residentGroups=%u/%u loads=%u uncompleted=%u unloads=%u",
                    frameIndex, loadFactor,
                    m_adaptiveLodErrorController.GetSmoothedLoadFactor(),
                    m_adaptiveLodErrorController.GetEffectiveError(),
                    (unsigned long long)(s.usedDataBytes >> 20),
                    (unsigned long long)(s.maxDataBytes >> 20),
                    (unsigned long long)(s.usedClasBytes >> 20),
                    (unsigned long long)(s.wastedClasBytes >> 20),
                    (unsigned long long)(s.reservedClasBytes >> 20),
                    s.maxSizedLeft, s.maxSizedReserved,
                    s.residentGroups, s.maxGroups,
                    s.loadCount, s.uncompletedLoadCount, s.unloadCount);
            }

            lodPixelError = m_adaptiveLodErrorController.Advance(m_lodPixelError);
        }
        else
        {
            m_adaptiveLodErrorController.Reset(m_lodPixelError);
        }

        // threshold = 2 * tan(fov/2) * pixelError / viewportHeight.  The viewport
        // spans 2*tan(fov/2) per unit distance, so the leading 2 makes
        // lodPixelError a whole pixel rather than a half-pixel radius.
        //
        // Against the DISPLAY height, not the render height: the error is what
        // the viewer sees, so it has to mean the same thing whatever DLSS is
        // upscaling from.  Measured against render height, DLAA held ~4x the
        // cluster density of Performance for the same number.
        const float fovYRad = dm::radians(lodCamera.GetFovY());
        const float viewportHeight = float(std::max(1, m_displaySize.y));
        lodConsts.errorOverDistanceThreshold = 2.0f * tanf(fovYRad * 0.5f) * lodPixelError / viewportHeight;
        lodConsts.nearPlane                  = lodCamera.GetZNear();
        // Camera world position for the BLAS-sharing min-sphere far-push
        // (instance_classify_lod transforms it into object space).
        lodConsts.viewPos                    = lodCamera.GetEye();
        // BLAS-sharing knobs (sharingEnabledLevels from --blassharing [levels]).
        lodConsts.sharingEnabledLevels       = m_blasSharingEnabledLevels;
        lodConsts.sharingTolerantLevels      = 7u;
        lodConsts.sharingPushCulled          = 1u;
        lodConsts.useCulling                 = m_useCulling ? 1u : 0u;
        lodConsts.useHardCull                = m_useHardCull ? 1u : 0u;
        lodConsts.hardCullForcesInvisible    = m_hardCullForcesInvisible ? 1u : 0u;
        lodConsts.culledErrorScale           = std::max(1.0f, m_culledErrorScale);
        {
            const float vw = float(std::max(1, m_renderSize.x));
            const float vh = float(std::max(1, m_renderSize.y));
            lodConsts.viewportf = dm::float4(vw, vh, 1.0f / vw, 1.0f / vh);
        }
        // The cull matrix matches the pyramid's viewpoint: the z-prepass traced it
        // this frame from the current camera.
        lodConsts.cullViewProjMatrix = lodCamera.GetViewProjectionMatrix();

        rtxmg::ClusterLodSystem::FrameParams lodParams;
        lodParams.constants            = lodConsts;
        lodParams.zbuffer              = m_zbuffer.get();
        lodParams.useHizOcclusion      = m_useHizOcclusion;
        lodParams.useStreaming         = m_useStreaming;
        lodParams.useBlasSharing       = m_useBlasSharing;
        lodParams.useBlasCaching       = EffectiveUseBlasCaching();
        // Merging requires sharing + streaming.
        lodParams.useBlasMerging       = m_useStreaming && m_useBlasSharing && m_useBlasMerging;
        lodParams.blasCacheMinLevel    = m_blasCachingEnabledLevels;
        lodParams.blasCacheMaxClusters = 1u << m_clodBudgets.renderClusterBits;
        lodParams.debugClusterLod      = m_debugClusterLod;
        m_clusterLodSystem.Update(GetDevice(), commandList, lodParams);
    }

    ScopedGPUTimer tlasTimer(stats::clusterAccelSamplers.tlasBuildTime, commandList);

    m_scene->Refresh(commandList, frameIndex);

    std::vector<nvrhi::rt::InstanceDesc> instances;
    std::vector<SubdInstance> subdInstances;

    uint32_t instanceIndex = 0;
    for (const auto& instance : m_scene->GetSubdMeshInstances())
    {
        auto& mesh = m_scene->GetSubdMeshes()[instance.meshID];

        unsigned int writeDepthFlag = !(mesh->HasAnimation());

        nvrhi::rt::InstanceDesc instanceDesc;
        instanceDesc.blasDeviceAddress = 0; // will get filled out later by fill instance desc indirect arg
        instanceDesc.instanceMask = (writeDepthFlag << 1) | 0x1;
        instanceDesc.instanceID = instanceIndex;
        instanceDesc.instanceContributionToHitGroupIndex = 0;

        assert(instance.node);
        dm::affineToColumnMajor(instance.node->localToWorld, instanceDesc.transform);

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

        subdInstance.meshID = instance.meshID;
        subdInstance._meshPad = 0;
        subdInstance.vertexSurfaceDescriptorBindlessIndex = getDescriptorHeapIndex(mesh->m_vertexSurfaceDescriptorDescriptor);
        subdInstance.vertexControlPointIndicesBindlessIndex = getDescriptorHeapIndex(mesh->m_vertexControlPointIndicesDescriptor);
        const SubdivisionSurface::Pose* pose = m_scene->GetSubdPose(instance);
        subdInstance.positionsBindlessIndex =
            getDescriptorHeapIndex(pose ? pose->positionsDescriptor : mesh->m_positionsDescriptor);
        subdInstance.positionsPrevBindlessIndex =
            pose ? getDescriptorHeapIndex(pose->positionsPrevDescriptor) : kInvalidBindlessIndex;
        subdInstance.surfaceToMaterialIndexBindlessIndex = getDescriptorHeapIndex(mesh->m_surfaceToMaterialIndexDescriptor);
        subdInstance.topologyQualityBindlessIndex = getDescriptorHeapIndex(mesh->m_topologyQualityDescriptor);

        affineToColumnMajor(instance.node->prevLocalToWorld, subdInstance.prevLocalToWorld);
        affineToColumnMajor(inverse(instance.node->localToWorld), subdInstance.worldToLocal);
        subdInstances.push_back(subdInstance);

        ++instanceIndex;
    }

    uint32_t numSubd = uint32_t(instances.size());
    uint32_t numClusterLod = m_clusterLodResources ? m_clusterLodResources->GetRenderInstanceCount() : 0u;
    uint32_t total   = numSubd + numClusterLod;

    if (total == 0)
        return;

    // Allocate m_instanceDescs for the full combined count (subd + cluster-LOD).
    // Both portions are written every frame below.
    if (!m_instanceDescs.GetBuffer() || m_instanceDescs.GetNumElements() != total)
    {
        nvrhi::BufferDesc instanceDescsDesc =
        {
            .byteSize            = total * sizeof(nvrhi::rt::InstanceDesc),
            .structStride        = sizeof(nvrhi::rt::InstanceDesc),
            .debugName           = "TLAS InstanceDescs",
            .canHaveUAVs         = true,
            .isAccelStructBuildInput = true,
            .initialState        = nvrhi::ResourceStates::AccelStructBuildInput,
            .keepInitialState    = true,
        };
        m_instanceDescs.Create(instanceDescsDesc, GetDevice());
        m_subdInstancesBuffer.Create(std::max(numSubd, 1u), "SubdInstances", GetDevice());
    }

    if (numSubd > 0)
    {
        if (m_subdInstancesBuffer.GetNumElements() > subdInstances.size())
            subdInstances.resize(m_subdInstancesBuffer.GetNumElements());

        commandList->writeBuffer(m_instanceDescs.GetBuffer(),
                                  instances.data(),
                                  numSubd * sizeof(nvrhi::rt::InstanceDesc));
        m_subdInstancesBuffer.Upload(subdInstances, commandList);

        // Patch subd BLAS addresses [0..numSubd).
        nvrhi::IBuffer* blasAddresses = m_sceneAccels->blasPtrsBuffer;
        if (blasAddresses != nullptr)
            FillInstanceDescs(commandList, m_instanceDescs, blasAddresses, numSubd, 0u);
    }

    // Write cluster-LOD instance descs [numSubd..total) with static fields,
    // then patch BLAS addresses via FillInstanceDescs.
    if (numClusterLod > 0)
    {
        const auto& clusterLodInsts = m_scene->GetClusterLodInstances();
        std::vector<nvrhi::rt::InstanceDesc> clusterLodDescs(numClusterLod);
        for (uint32_t i = 0; i < numClusterLod; ++i)
        {
            nvrhi::rt::InstanceDesc& d = clusterLodDescs[i];
            affineToColumnMajor(homogeneousToAffine(clusterLodInsts[i].transform), d.transform);
            d.instanceID                          = i;  // CLUSTER_LOD-local index; ClusterLodClosestHit indexes t_ClusterLodInstances[InstanceID()]
            d.instanceContributionToHitGroupIndex = 2u;  // ClusterLodHitGroup
            d.instanceMask                        = 0xFF;
            d.blasDeviceAddress                   = 0;   // patched below
        }
        commandList->writeBuffer(m_instanceDescs.GetBuffer(),
                                  clusterLodDescs.data(),
                                  numClusterLod * sizeof(nvrhi::rt::InstanceDesc),
                                  numSubd * sizeof(nvrhi::rt::InstanceDesc));

        // Patch cluster-LOD BLAS addresses (lowDetail for fast-path, dynamic for slow-path).
        if (m_clusterLodSystem.GetPass().GetInstanceBlasAddrsBuffer())
        {
            nvrhi::IBuffer* clusterLodBlas = m_clusterLodSystem.GetPass().GetInstanceBlasAddrsBuffer();
            const bool debugTlasFill = ShouldDebugClusterLodTlasFill();
            if (debugTlasFill)
                LogClusterLodTlasFillInputs(clusterLodBlas, numClusterLod, numSubd);

            FillInstanceDescs(commandList, m_instanceDescs, clusterLodBlas, numClusterLod, numSubd);

            if (debugTlasFill)
                DebugReadbackClusterLodTlasFill(commandList, numSubd);
        }
    }

    {
        nvrhi::utils::ScopedMarker marker(commandList, "TLAS Update");
        commandList->buildTopLevelAccelStructFromBuffer(m_topLevelAS, m_instanceDescs, 0u, total);
    }
    m_topLevelASBuilt = true;
    m_needsRebind = true;
}

void RTXMGRenderer::DumpPixelDebugBuffers(nvrhi::ICommandList* commandList)
{
#if ENABLE_SHADER_DEBUG
    log::info("Raytracing Pixel Debug: %d, %d", m_params.debugPixel.x, m_params.debugPixel.y);
    auto debugOutput = m_pixelDebugBuffer.Download(commandList);
    uint numElements = debugOutput.front().payloadType;
    vectorlog::Log(debugOutput, ShaderDebugElement::OutputLambda, vectorlog::FormatOptions{ .wrap = false, .header = false, .elementIndex = false, .startIndex = 1, .count = numElements });
#endif
}

#if ENABLE_PIXEL_PICK
void RTXMGRenderer::ReadPixelPick(nvrhi::ICommandList* commandList)
{
    // Silent: a right-click is a user-facing gesture, not a diagnostic dump.
    auto pick = m_pixelPickBuffer.Download(commandList).front();

    m_pixelPick = {};
    if (pick.claimed == 0)
        return;

    m_pixelPick.instanceID   = pick.instanceID;
    m_pixelPick.surfaceID    = pick.surfaceID;
    m_pixelPick.materialID   = pick.materialID;
    m_pixelPick.isClusterLod = (pick.tag & 1u) != 0;
    m_pixelPick.lodLevel     = pick.tag >> 8;
    if (m_scene)
    {
        const auto& mats = m_scene->GetMaterials();
        m_pixelPick.name = (m_pixelPick.materialID < mats.size() && mats[m_pixelPick.materialID])
            ? mats[m_pixelPick.materialID]->name : "?";
    }
    m_pixelPick.valid    = true;
    m_pixelPick.sequence = ++m_pixelPickSeq;
}
#endif
