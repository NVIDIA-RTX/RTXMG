
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

#include "rtxmg_demo_app.h"
#include "maya_logger.h"
#include "korgi.h"

#include <donut/app/ApplicationBase.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>

#include <filesystem>
#include <iostream>
#include <fstream>
#include <utility>

#include "rtxmg/profiler/statistics.h"
#include "rtxmg/scene/scene.h"
#include "rtxmg/subdivision/subdivision_surface.h"
#include "rtxmg/cluster_builder/tessellator_config.h"
#include "rtxmg/subdivision/shape.h"
#include "rtxmg/utils/buffer.h"
#include "rtxmg/utils/debug.h"

#include "lerp_keyframes_params.h"


#define GLFW_INCLUDE_NONE // Do not include any OpenGL headers
#include <GLFW/glfw3.h>

using namespace donut;

namespace fs = std::filesystem;

// Sample application, so use dummy app id
static constexpr int kStreamlineAppId = 1;

// search "up-ward" from the start path for a given directory name
static fs::path findDir(fs::path const& startPath, fs::path const& dirname,
    int maxDepth)
{
    std::filesystem::path searchPath = "";

    for (int depth = 0; depth < maxDepth; depth++)
    {
        fs::path currentPath = startPath / searchPath / dirname;

        if (fs::is_directory(currentPath))
            return currentPath.lexically_normal();

        searchPath = ".." / searchPath;
    }
    return {};
}

static fs::path findMediaFolder(fs::path const& startdir, char const* dirname,
    int maxdepth = 5)
{
    fs::path mediapath;
    try
    {
        fs::path start = fs::canonical(startdir).parent_path();
        mediapath = findDir(start, dirname, maxdepth);
    }
    catch (std::exception const& e)
    {
        fprintf(stderr, "%s\n", e.what());
    }
    return mediapath;
}

void RTXMGDemoApp::MessageCallback::message(nvrhi::MessageSeverity severity, const char* messageText)
{
    donut::log::Severity donutSeverity = donut::log::Severity::Info;
    switch (severity)
    {
    case nvrhi::MessageSeverity::Info:
        donutSeverity = donut::log::Severity::Info;
        break;
    case nvrhi::MessageSeverity::Warning:
        donutSeverity = donut::log::Severity::Warning;
        break;
    case nvrhi::MessageSeverity::Error:
        donutSeverity = donut::log::Severity::Error;
        break;
    case nvrhi::MessageSeverity::Fatal:
        donutSeverity = donut::log::Severity::Fatal;
        break;
    }

    // Framecount
    if (m_deviceManager)
    {
        donut::log::message(donutSeverity, "[%u] %s", m_deviceManager->GetFrameIndex(), messageText);
    }
}

void RTXMGDemoApp::UpdateParams()
{
    m_denoiserMode = m_args.enableDenoiser ? DenoiserMode::DlssRr : DenoiserMode::None;

    m_renderParams.colorMode = m_args.colorMode;
    m_renderParams.shadingMode = m_args.shadingMode;
    m_renderParams.spp = m_args.spp;
    m_renderParams.enableWireframe = m_args.enableWireframe;
    m_renderParams.wireframeThickness = m_args.wireframeThickness;
    m_renderParams.fireflyMaxIntensity = m_args.firefliesClamp;
    m_renderParams.roughnessOverride = m_args.roughnessOverride;
    m_renderParams.missColor = float3(m_args.missColor);
    m_renderParams.ptMaxBounces = m_args.ptMaxBounces;
    m_renderParams.denoiserMode = m_denoiserMode;
    m_renderParams.enableTimeView = m_args.enableTimeView;

    m_renderParams.globalDisplacementScale = m_args.dispScale;
    m_renderParams.clusterPattern = uint32_t(m_args.clusterPattern);

    m_renderParams.hasEnvironmentMap = 0;
    m_renderParams.enableEnvmapHeatmap = 0;
    m_renderParams.debugPixel = int2(0, 0);

    if (m_renderer)
    {
        GetRenderer().SetEnvMapAzimuth(0);
        GetRenderer().SetEnvMapElevation(0);
        GetRenderer().SetEnvMapIntensity(1);

        for (size_t i = 0; i < m_args.textures.size(); ++i)
        {
            if (!m_args.textures[i].empty())
            {
                if (TextureType(i) == TextureType::ENVMAP)
                    SetEnvmapTex(m_args.textures[i]);
            }
        }
        GetRenderer().ResetSubframes();
        if (GetRenderer().GetEnvMap())
            m_renderParams.hasEnvironmentMap = 1;

        GetRenderer().SetShadingMode(m_args.shadingMode);
        GetRenderer().SetColorMode(m_args.colorMode);
        GetRenderer().SetTonemapOperator(m_args.tonemapOperator);
        GetRenderer().SetExposure(m_args.exposure);
    }
}

bool RTXMGDemoApp::SetEnvmapTex(const std::string& filePath)
{
    if (m_scene)
    {
        // current environment map might be in use
        GetDevice()->waitForIdle();
        m_commandList->open();
        GetRenderer().SetEnvMap(filePath, m_commandList);
        GetRenderer().SetEnvMapAzimuth(m_args.envmapAzimuth);
        GetRenderer().SetEnvMapElevation(m_args.envmapElevation);
        GetRenderer().SetEnvMapIntensity(m_args.envmapIntensity);
        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
        m_renderParams.hasEnvironmentMap = 1;
        return true;
    }
    return false;
}

RTXMGDemoApp::RTXMGDemoApp(app::DeviceManager* deviceManager,
    int argc, const char** argv)
    : app::ApplicationBase(deviceManager)
    , m_messageCallback(deviceManager)
{
    m_argc = argc;
    m_argv = argv;

    m_binaryPath = app::GetDirectoryWithExecutable().lexically_normal();
    m_mediaPath = findMediaFolder(m_binaryPath, "assets");

    m_args.Parse(m_argc, m_argv);
    m_displaySize = int2(m_args.width, m_args.height);

    UpdateParams();

    app::DeviceCreationParameters deviceParams;
    deviceParams.enableHeapDirectlyIndexed = true; // Needed for bindless look up in motion_vectors.hlsl
    deviceParams.enableRayTracingExtensions = true;
    deviceParams.backBufferWidth = m_displaySize.x;
    deviceParams.backBufferHeight = m_displaySize.y;
    deviceParams.enablePerMonitorDPI = true;
    deviceParams.allowModeSwitch = false; // Make Alt+Enter not switch monitor resolutions
    deviceParams.startMaximized = m_args.startMaximized;
    deviceParams.swapChainFormat = nvrhi::Format::RGBA8_UNORM;
    deviceParams.messageCallback = &m_messageCallback;
    if (m_args.debug)
    {
        deviceParams.enableDebugRuntime = true;
        deviceParams.enableNvrhiValidationLayer = true;
    }

    if (m_args.gpuValidation)
    {
        deviceParams.enableGPUValidation = true;
    }

    if (m_args.aftermath)
    {
#if DONUT_WITH_AFTERMATH
        deviceParams.enableAftermath = true;
#endif
        deviceParams.logBufferLifetime = true;
    }

#if DONUT_WITH_STREAMLINE
    if (m_args.enableStreamlineLog)
    {
        deviceParams.enableStreamlineLog = true;
    }
    deviceParams.streamlineAppId = kStreamlineAppId;
#endif

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams,
        UIData::WindowTitle))
    {
        log::fatal(
            "Cannot initialize a graphics device with the requested parameters");
    }

    if (!deviceManager->GetDevice()->queryFeatureSupport(
        nvrhi::Feature::RayTracingPipeline))
    {
        log::fatal("The graphics device does not support Ray Tracing Pipelines");
    }

    if (!deviceManager->GetDevice()->queryFeatureSupport(
        nvrhi::Feature::RayTracingClusters))
    {
        log::fatal("The graphics device does not support Clusters");
    }

    if (!deviceManager->GetDevice()->queryFeatureSupport(
        nvrhi::Feature::HeapDirectlyIndexed))
    {
        log::fatal("The graphics device does not support directly indexing heaps (ResourceDescriptorHeap) (SamplerDescriptorHeap)");
    }

    korgi::Init();
}

RTXMGDemoApp::~RTXMGDemoApp()
{
    korgi::Shutdown();
}

RTXMGRenderer& RTXMGDemoApp::GetRenderer()
{
    if (!m_renderer)
    {
        RTXMGRenderer::Options rendererOptions{
            .params = m_renderParams,
            .device = GetDevice()
        };

        m_renderer = std::make_unique<RTXMGRenderer>(rendererOptions);
    }
    return *m_renderer;
}

static constexpr size_t const kMinVram = 10;
static constexpr size_t const kGigabyte = 1024 * 1024 * 1024;

bool RTXMGDemoApp::Init()
{
    // Check to see if we have enough vram to run the demo

    std::vector<app::AdapterInfo> adapters;

    if (!GetDeviceManager()->EnumerateAdapters(adapters))
    {
        donut::log::fatal("Failed to enumerate adapters for vram check");
    }

#ifdef _WIN32
    ID3D12Device* rawDevice = (ID3D12Device*)GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);

    for (const auto& adapter : adapters)
    {
        LUID rawLuid = rawDevice->GetAdapterLuid();

        if (!memcmp(&adapter.luid, &rawLuid, sizeof(LUID)))
        {
            size_t vram = adapter.dedicatedVideoMemory;
            if (vram < kMinVram * kGigabyte)
            {
                donut::log::error("GPU has %.2fGB of VRAM and is below the required %dGB.\n\n"
                    "Expect the following:\n"
                    "1. Performance degradation or out of memory crashes.\n"
                    "2. Flickering and missing surfaces after adjusting the memory budget down", float(vram) / kGigabyte, kMinVram);
            }
        }
    }
#endif
    std::filesystem::path sceneFileName;
    if (!m_args.meshInputFile.empty())
    {
        sceneFileName = app::GetDirectoryWithExecutable().parent_path() / "assets" /
            m_args.meshInputFile;

        if (!std::filesystem::exists(sceneFileName))
        {
            std::filesystem::path mediaSceneFileName = GetMediaPath() / m_args.meshInputFile;
            if (mediaSceneFileName != sceneFileName)
            {
                sceneFileName = mediaSceneFileName;

                if (!std::filesystem::exists(sceneFileName))
                {
                    donut::log::error("Could not find mesh input %s", sceneFileName);
                    sceneFileName = "";
                }
            }
            else
            {
                donut::log::error("Could not find mesh input %s", sceneFileName.c_str());
                sceneFileName = "";
            }
        }
    }

    RTXMGRenderer& renderer = GetRenderer();
    m_TextureCache = renderer.GetTextureCache();
    m_CommonPasses = renderer.GetCommonPasses();
    m_commandList = GetDevice()->createCommandList();

    m_lerpKeyFramesParamsBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(LerpKeyFramesParams), "LerpKeyFramesParams", engine::c_MaxRenderPassConstantBufferVersions));

    SetAsynchronousLoadingEnabled(false);
    HandleSceneLoad(sceneFileName.lexically_normal().generic_string(),
        GetMediaPath().generic_string());

    return true;
}

void RTXMGDemoApp::HandleSceneLoad(const std::string& sceneFileName,
    const std::string& mediaPath,
    int2 frameRange)
{
    auto nativeFS = std::make_shared<vfs::NativeFileSystem>();

    // don't have a way to pass this through to the scene loader
    m_loadFrameRange = frameRange;

    m_args.sceneArgs() = {};
    auto& renderer = GetRenderer();
    renderer.ClearEnvMap();

    BeginLoadingScene(nativeFS, sceneFileName);

    m_sunLight = std::make_shared<engine::DirectionalLight>();
    m_scene->GetSceneGraph()->AttachLeafNode(
        m_scene->GetSceneGraph()->GetRootNode(), m_sunLight);

    m_sunLight->angularSize = 0.53f;
    m_sunLight->irradiance = 3.f;

    m_scene->FinishedLoading(GetFrameIndex());

    m_args.Parse(m_argc, m_argv); // re-parse command line args in case they override anything in the scene

    UpdateParams();

    m_zRenderer = std::make_unique<ZRenderer>(renderer.GetShaderFactory(), renderer.GetDescriptorTable());

    m_commandList->open();
    {
        nvrhi::utils::ScopedMarker marker(m_commandList, "Scene Load");
        if (auto* zbuffer = renderer.GetZBuffer())
        {
            zbuffer->Clear(m_commandList);
        }
        m_scene->Refresh(m_commandList, GetFrameIndex());
    }
    m_commandList->close();
    GetDevice()->executeCommandList(m_commandList);

    ResetCamera();
    m_tesselationCamera = m_camera;
    m_accelBuilderNeedsUpdate = true;

    GetRenderer().SceneFinishedLoading(m_scene);
}

void RTXMGDemoApp::ResetCamera()
{
    m_cameraReset = true;
    if (!m_args.camString.empty())
    {
        m_camera.Set(m_args.camString);
    }
    else if (const View* view = m_scene->GetView())
    {
        m_camera.SetEye(view->position);
        m_camera.SetLookat(view->lookat);
        m_camera.SetUp(view->up);
        m_camera.SetFovY(view->fov);
    }
    else
    {
        box3 aabb = m_scene->GetSceneGraph()->GetRootNode()->GetGlobalBoundingBox();
        m_camera.Frame(aabb);
    }

    m_camera.SetAspectRatio(float(m_displaySize.x) / float(m_displaySize.y));

    m_trackBall.SetGimbalLock(true);
    m_trackBall.SetCamera(&m_camera);
    m_trackBall.SetMoveSpeed(m_scene->GetAttributes().averageInstanceScale);
    m_trackBall.SetReferenceFrame({ 1.f, 0.f, 0.f }, { 0.f, 0.f, 1.f },
        { 0.f, 1.f, 0.f });
}

bool RTXMGDemoApp::LoadScene(std::shared_ptr<vfs::IFileSystem> fs,
    const std::filesystem::path& sceneFileName)
{
    stats::evaluatorSamplers = {};
    stats::memUsageSamplers = {};

    auto shaderFactory = GetRenderer().GetShaderFactory();
    RTXMGScene* scene =
        new RTXMGScene(GetDevice(), GetMediaPath(),
            m_CommonPasses, *shaderFactory, fs, m_TextureCache,
            GetRenderer().GetDescriptorTable(), nullptr,
            m_loadFrameRange, m_args.isoLevelSharp, m_args.isoLevelSmooth);

    if (scene->Load(sceneFileName))
    {
        m_args.meshInputFile = scene->GetInputPath();
        const Json::Value& settings = scene->GetSceneSettings();
        m_args << settings;

        if (std::string& envarg = m_args.textures[TextureType::ENVMAP]; envarg.empty())
            if (const Json::Value& envmap = settings["envmap"]; envmap.isString())
                if (fs::path filepath = RTXMGScene::ResolveMediapath(envmap.asString(), m_mediaPath); !filepath.empty())
                    envarg = filepath.lexically_normal().generic_string();


        m_scene = std::unique_ptr<RTXMGScene>(scene);
        m_accelBuilderNeedsUpdate = true;
        UpdateParams();
        return true;
    }

    log::fatal("Failed to load scene from file: %s", sceneFileName.string().c_str());

    return false;
}

bool RTXMGDemoApp::KeyboardUpdate(int key, int scancode, int action,
    int mods)
{
    auto& renderer = GetRenderer();
    m_trackBall.KeyboardUpdate(key, scancode, action, mods);

    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(GetDeviceManager()->GetWindow(), true);
            break;
        case GLFW_KEY_1:
            NextShadingMode();
            break;
        case GLFW_KEY_2:
            IncrementColorMode(1);
            break;
        case GLFW_KEY_3:
            ToggleWireframe();
            break;
        case GLFW_KEY_4:
            IncrementColorMode(-1);
            break;
        case GLFW_KEY_5:
            NextTonemapper();
            break;
        case GLFW_KEY_C:
            m_camera.Print();
            break;
        case GLFW_KEY_F:
            ResetCamera();
            break;
        case GLFW_KEY_R:
            if (mods & GLFW_MOD_CONTROL)
                ReloadShaders();
            break;
        case GLFW_KEY_P:
            SaveScreenshot();
            break;
        case GLFW_KEY_T:
            ToggleTimeView();
            break;
        case GLFW_KEY_SLASH:
            ToggleUpdateTessellationCamera();
            break;
        case GLFW_KEY_RIGHT:
            IncrementMaxBounces(1);
            break;
        case GLFW_KEY_LEFT:
            IncrementMaxBounces(-1);
            break;
        }
    }
    return true;
}

bool RTXMGDemoApp::MousePosUpdate(double xpos, double ypos)
{
    int2 pos = { static_cast<int>(xpos), static_cast<int>(ypos) };
    int2 canvas = { m_displaySize.x, m_displaySize.y };
    m_trackBall.MouseTrackingUpdate(pos, canvas);
    return true;
}

bool RTXMGDemoApp::MouseButtonUpdate(int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    {
        double mousex = 0, mousey = 0;
        glfwGetCursorPos(GetDeviceManager()->GetWindow(), &mousex, &mousey);

        m_renderParams.debugPixel = int2(int(mousex), int(mousey));

        m_dumpPixelDebug = true;
    }

    m_trackBall.MouseButtonUpdate(button, action, mods);
    return true;
}

bool RTXMGDemoApp::MouseScrollUpdate(double xoffset, double yoffset)
{
    if (yoffset != 0)
    {
        m_trackBall.MouseWheelUpdate((int)yoffset);
    }
    return true;
}

void RTXMGDemoApp::BackBufferResizing() {}

void RTXMGDemoApp::BackBufferResized(const uint32_t width,
    const uint32_t height,
    const uint32_t sampleCount)
{
    m_cameraReset = true;
    m_camera.SetAspectRatio(float(width) / float(height));
    m_displaySize = int2(width, height);

    // We need to cache here since we can't query the window during application quit
    // Donut doesn't have a callback to the application for window pos, so we skip saving that
    GLFWwindow* window = GetDeviceManager()->GetWindow();
    m_windowState.isMaximized = glfwGetWindowAttrib(window, GLFW_MAXIMIZED) != 0;
    m_windowState.isFullscreen = glfwGetWindowMonitor(window) != nullptr;
    if (!m_windowState.isMaximized && !m_windowState.isFullscreen)
    {
        // Only save the non-maximized size
        glfwGetWindowSize(window, &m_windowState.windowSize.x, &m_windowState.windowSize.y);
    }
}

void RTXMGDemoApp::UpdateDLSSSettings()
{
#if DONUT_WITH_STREAMLINE
    using StreamlineInterface = donut::app::StreamlineInterface;

    StreamlineInterface& streamline = donut::app::DeviceManager::GetStreamline();

    const uint32_t kViewportId = 0;
    streamline.SetViewport(kViewportId);

    DenoiserMode denoiserMode = GetRenderer().GetShowMicroTriangles() ? DenoiserMode::None : m_denoiserMode;
    if (denoiserMode == DenoiserMode::DlssSr ||
        denoiserMode == DenoiserMode::DlssRr)
    {
        bool isDlssRr = denoiserMode == DenoiserMode::DlssRr;

        StreamlineInterface::DLSSOptions dlssOptions = {};
        dlssOptions.mode = m_ui.dlssMode;
        dlssOptions.outputWidth = m_displaySize.x;
        dlssOptions.outputHeight = m_displaySize.y;
        dlssOptions.colorBuffersHDR = true;
        dlssOptions.sharpness = m_RecommendedDLSSSettings.sharpness;

        dlssOptions.preset = m_ui.dlssPreset;
        dlssOptions.useAutoExposure = false;

        StreamlineInterface::DLSSRROptions dlssRROptions = {};
        dlssRROptions.mode = m_ui.dlssMode;
        dlssRROptions.outputWidth = m_displaySize.x;
        dlssRROptions.outputHeight = m_displaySize.y;
        dlssRROptions.sharpness = m_RecommendedDLSSRRSettings.sharpness;
        dlssRROptions.preExposure = 1.0f;
        dlssRROptions.exposureScale = 1.0f;
        dlssRROptions.colorBuffersHDR = true;
        dlssRROptions.normalRoughnessMode = StreamlineInterface::DLSSRRNormalRoughnessMode::eUnpacked;

        dlssRROptions.preset = m_ui.dlssRRPreset;

        float4x4 worldToViewRowMajor = transpose(m_camera.GetViewMatrix());
        dlssRROptions.worldToCameraView = worldToViewRowMajor;
        dlssRROptions.cameraViewToWorld = inverse(worldToViewRowMajor);

        // Changing presets requires a restart of DLSS
        // Current bug, should get fixed in new version of streamline past 2.7
        if (m_dlssLastPreset != m_ui.dlssPreset)
        {
            streamline.CleanupDLSS(true);
            m_dlssLastPreset = m_ui.dlssPreset;
        }

        if (isDlssRr)
            streamline.SetDLSSRROptions(dlssRROptions);
        else
            streamline.SetDLSSOptions(dlssOptions);

        // Check if we need to update the rendertarget size.
        bool DLSS_resizeRequired = (m_ui.dlssMode != m_dlssLastMode) || (m_displaySize.x != m_dlssLastDisplaySize.x) || (m_displaySize.y != m_dlssLastDisplaySize.y);
        if (DLSS_resizeRequired)
        {
            // Only quality, target width and height matter here
            streamline.QueryDLSSOptimalSettings(dlssOptions, m_RecommendedDLSSSettings);
            streamline.QueryDLSSRROptimalSettings(dlssRROptions, m_RecommendedDLSSRRSettings);

            int2& optimalRenderSize = isDlssRr ? m_RecommendedDLSSRRSettings.optimalRenderSize :
                m_RecommendedDLSSSettings.optimalRenderSize;

            if (optimalRenderSize.x <= 0 || optimalRenderSize.y <= 0)
            {
                donut::log::warning("DLSS Recommended Settings returned render size %d,%d", optimalRenderSize.x, optimalRenderSize.y);
                denoiserMode = DenoiserMode::None;
            }
            else
            {
                m_dlssLastMode = m_ui.dlssMode;
                m_dlssLastDisplaySize = m_displaySize;
                m_renderSize = optimalRenderSize;
            }
        }

        float texLodXDimension = (float)m_renderSize.x;

        // Use the formula of the DLSS programming guide for the Texture LOD Bias...
        float optimalLodBias = std::log2f(texLodXDimension / m_displaySize.x) - 1;
        float lodBias = m_ui.dlssUseLodBiasOverride ? m_ui.dlssLodBiasOverride : optimalLodBias;
        if (lodBias != m_lodBias)
        {
            m_lodBias = lodBias;

            GetDevice()->waitForIdle();
            {
                nvrhi::SamplerDesc samplerDescPoint = m_CommonPasses->m_PointClampSampler->getDesc();
                nvrhi::SamplerDesc samplerDescLinear = m_CommonPasses->m_LinearClampSampler->getDesc();
                nvrhi::SamplerDesc samplerDescLinearWrap = m_CommonPasses->m_LinearWrapSampler->getDesc();
                nvrhi::SamplerDesc samplerDescAniso = m_CommonPasses->m_AnisotropicWrapSampler->getDesc();
                samplerDescPoint.mipBias = lodBias;
                samplerDescLinear.mipBias = lodBias;
                samplerDescLinearWrap.mipBias = lodBias;
                samplerDescAniso.mipBias = lodBias;
                m_CommonPasses->m_PointClampSampler = GetDevice()->createSampler(samplerDescPoint);
                m_CommonPasses->m_LinearClampSampler = GetDevice()->createSampler(samplerDescLinear);
                m_CommonPasses->m_LinearWrapSampler = GetDevice()->createSampler(samplerDescLinearWrap);
                m_CommonPasses->m_AnisotropicWrapSampler = GetDevice()->createSampler(samplerDescAniso);
            }
        }
    }
#else
    DenoiserMode denoiserMode = DenoiserMode::None;
#endif

    // Off, or disabled due to invalid settings.
    if (denoiserMode == DenoiserMode::None)
    {
#if DONUT_WITH_STREAMLINE
        StreamlineInterface::DLSSOptions dlssOptions = {};
        dlssOptions.mode = StreamlineInterface::DLSSMode::eOff;
        streamline.SetDLSSOptions(dlssOptions);
        m_dlssLastMode = StreamlineInterface::DLSSMode::eOff;
#endif
        m_renderSize = m_displaySize;
        m_lodBias = 1.0f;
    }

    // Update effective denoiser mode
    m_renderParams.denoiserMode = denoiserMode;
}

void RTXMGDemoApp::Animate(float fElapsedTimeSeconds)
{
    korgi::Update();
    m_trackBall.Animate(GetCPUFrameTime() / 1000.f);

    const auto& animState = m_ui.timeLineEditorState;
    float animTime = animState.AnimationTime();
    float animRate = animState.frameRate;

    m_animationUpdated = m_animationTime != animTime;
    if (m_animationUpdated)
    {
        // animation looped so we need to reset
        if (animTime == 0)
        {
            GetRenderer().ResetDenoiser();
        }

        m_scene->Animate(animTime, animRate);

        GetRenderer().ResetSubframes();
        m_animationTime = animTime;
        m_accelBuilderNeedsUpdate = true;
    }
}

void RTXMGDemoApp::LerpVertices(
    nvrhi::IBuffer* outBuffer,
    nvrhi::IBuffer* keyFrame0Buffer,
    nvrhi::IBuffer* keyFrame1Buffer,
    unsigned int numVertices, float animTime)
{
    constexpr int blockSize = 32;
    const int numBlocks = (numVertices + blockSize - 1) / blockSize;

    LerpKeyFramesParams params;
    params.numVertices = numVertices;
    params.animTime = animTime;

    m_commandList->writeBuffer(m_lerpKeyFramesParamsBuffer, &params, sizeof(LerpKeyFramesParams));

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, keyFrame0Buffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, keyFrame1Buffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, outBuffer))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_lerpKeyFramesParamsBuffer));

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_lerpVerticesBL, bindingSet))
    {
        log::fatal("Failed to create binding set and layout for lerp_keyframes.hlsl");
    }

    if (!m_lerpVerticesPSO)
    {
        nvrhi::ShaderHandle shader = GetRenderer().GetShaderFactory()->CreateShader("rtxmg_demo/lerp_keyframes.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(shader)
            .addBindingLayout(m_lerpVerticesBL);

        m_lerpVerticesPSO = GetDevice()->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_lerpVerticesPSO)
        .addBindingSet(bindingSet);
    m_commandList->setComputeState(state);
    m_commandList->dispatch(numBlocks, 1, 1);
}

void RTXMGDemoApp::DispatchGPUAnimation()
{
    nvrhi::utils::ScopedMarker marker(m_commandList, "GPU Animation");

    auto& subdMeshes = m_scene->GetSubdMeshes();
    auto& instances = m_scene->GetInstances();
    for (auto& subd : subdMeshes)
    {
        if (!subd->HasAnimation())
            continue;

        // Cache to previous
        m_commandList->copyBuffer(subd->m_positionsPrevBuffer, 0, subd->m_positionsBuffer, 0, subd->m_positionsBuffer->getDesc().byteSize);

        LerpVertices(subd->m_positionsBuffer,
            subd->m_positionKeyframeBuffers[subd->m_f0],
            subd->m_positionKeyframeBuffers[subd->m_f1],
            subd->NumVertices(),
            subd->m_dt);
    }
}

void RTXMGDemoApp::Render(nvrhi::IFramebuffer* framebuffer)
{
    auto& profiler = Profiler::Get();

    auto& renderer = GetRenderer();
    if (m_reloadShaders)
    {
        GetDevice()->waitForIdle();

        renderer.ReloadShaders();

        m_lerpVerticesPSO.Reset();
        m_reloadShaders = false;
    }

    // Calculate DLSS settings
    UpdateDLSSSettings();

    renderer.SetRenderSize(m_renderSize, m_displaySize);
    renderer.SetRenderCamera(m_camera, m_cameraReset);
    m_sunLight->SetDirection(double3(m_camera.GetDirection()));

    m_prevFrameStart = GetFrameIndex() > 0 ? m_currFrameStart : std::chrono::steady_clock::now();
    m_currFrameStart = std::chrono::steady_clock::now();

    if (profiler.IsRecording())
        stats::frameSamplers.cpuFrameTime.PushBack(std::chrono::duration<float, std::milli>(m_currFrameStart - m_prevFrameStart).count());

    profiler.FrameStart(m_currFrameStart);
    m_commandList->open();
    {
        std::string frameMarker = "Frame Rendering " + std::to_string(GetFrameIndex());
        nvrhi::utils::ScopedMarker marker(m_commandList, frameMarker.c_str());

        DispatchGPUAnimation();

        stats::frameSamplers.gpuFrameTime.Start(m_commandList);
        if (m_accelBuilderNeedsUpdate || m_ui.forceRebuildAccelStruct)
        {
            const TessellatorConfig tessConfig =
            {
                .memorySettings = m_args.tessMemorySettings,
                .visMode = m_args.visMode,
                .tessMode = m_args.tessMode,
                .fineTessellationRate = m_args.fineTessellationRate,
                .coarseTessellationRate = m_args.coarseTessellationRate,
                .enableFrustumVisibility = m_args.enableFrustumVisibility,
                .enableHiZVisibility = m_args.enableHiZVisibility,
                .enableBackfaceVisibility = m_args.enableBackfaceVisibility,
                .enableLogging = m_args.enableAccelBuildLogging,
                .enableMonolithicClusterBuild = m_ui.enableMonolithicClusterBuild,
                .viewportSize = { (uint32_t)m_renderSize.x, (uint32_t)m_renderSize.y },
                .edgeSegments = m_args.edgeSegments,
                .quantNBits = m_args.quantNBits,
                .clusterPattern = (ClusterPattern)m_renderParams.clusterPattern,
                .displacementScale = m_renderParams.globalDisplacementScale,
                .camera = &m_tesselationCamera,
                .zbuffer = renderer.GetZBuffer()
            };

            renderer.UpdateAccelerationStructures(tessConfig, m_BuildStats, GetFrameIndex(), m_commandList);
            m_accelBuilderNeedsUpdate = false;
        }

        if (m_args.updateTessCamera)
        {
            ZBuffer* zbuffer = renderer.GetZBuffer();
            if (zbuffer)
            {
                m_zRenderer->Render(m_camera, renderer.GetTopLevelAS(), zbuffer->GetCurrent(), m_commandList);
                zbuffer->ReduceHierarchy(m_commandList);
            }
            m_tesselationCamera = m_camera;
        }

        renderer.Launch(m_commandList, GetFrameIndex(), m_sunLight);

        DenoiserMode effectiveDenoiserMode = GetEffectiveDenoiserMode();
        if (effectiveDenoiserMode == DenoiserMode::DlssSr ||
            effectiveDenoiserMode == DenoiserMode::DlssRr)
        {
            renderer.DlssUpscale(m_commandList, GetFrameIndex());
        }

        renderer.BlitFramebuffer(m_commandList, framebuffer);

        stats::frameSamplers.gpuFrameTime.Stop();

        if (m_dumpFineTess)
        {
            // dump cluster vertex positions for debugging
            DoDumpFineTess();
            m_dumpFineTess = false;
        }

        if (m_dumpDebugBuffer)
        {
            // dump debugging output
            DoDumpDebugBuffer();
            m_dumpDebugBuffer = false;
        }

        if (m_dumpPixelDebug)
        {
            GetRenderer().DumpPixelDebugBuffers(m_commandList);
            m_dumpPixelDebug = false;
        }
    }
    m_commandList->close();
    GetDevice()->executeCommandList(m_commandList);

    profiler.FrameEnd();

    if (profiler.IsRecording())
    {
        stats::clusterAccelSamplers.numClusters.PushBack(m_BuildStats.desired.m_numClusters);
        stats::clusterAccelSamplers.numClusters.max = m_BuildStats.allocated.m_numClusters;
        stats::clusterAccelSamplers.numTriangles.PushBack(m_BuildStats.desired.m_numTriangles);

        stats::clusterAccelSamplers.renderSize = m_renderSize;

        stats::memUsageSamplers.blasSize.PushBack(m_BuildStats.desired.m_blasSize);
        stats::memUsageSamplers.clasSize.PushBack(m_BuildStats.desired.m_clasSize);
        stats::memUsageSamplers.blasScratchSize.PushBack(m_BuildStats.desired.m_blasScratchSize);
        stats::memUsageSamplers.vertexBufferSize.PushBack(m_BuildStats.desired.m_vertexBufferSize);
        stats::memUsageSamplers.clusterShadingDataSize.PushBack(m_BuildStats.desired.m_clusterDataSize);

        stats::memUsageSamplers.blasSize.max = m_BuildStats.allocated.m_blasSize;
        stats::memUsageSamplers.clasSize.max = m_BuildStats.allocated.m_clasSize;
        stats::memUsageSamplers.blasScratchSize.max = m_BuildStats.allocated.m_blasScratchSize;
        stats::memUsageSamplers.vertexBufferSize.max = m_BuildStats.allocated.m_vertexBufferSize;
        stats::memUsageSamplers.clusterShadingDataSize.max = m_BuildStats.allocated.m_clusterDataSize;
    }

    if (m_screenshot)
    {
        nvrhi::ITexture* framebufferTexture = framebuffer->getDesc().colorAttachments[0].texture;
        DoSaveScreenshot(framebufferTexture, "");
        m_screenshot = false;
    }

    m_cameraReset = false;
}

void RTXMGDemoApp::SetTessMemSettings(const TessellatorConfig::MemorySettings& settings)
{
    if (settings != m_args.tessMemorySettings)
    {
        m_args.tessMemorySettings = settings;
        m_accelBuilderNeedsUpdate = true;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::SetFineTessellationRate(float rate)
{
    if (rate != m_args.fineTessellationRate)
    {
        m_args.fineTessellationRate = rate;
        m_accelBuilderNeedsUpdate = true;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::SetCoarseTessellationRate(float rate)
{
    if (rate != m_args.coarseTessellationRate)
    {
        m_args.coarseTessellationRate = rate;
        m_accelBuilderNeedsUpdate = true;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::SetTessellatorVisibilityMode(TessellatorConfig::VisibilityMode visMode)
{
    if (visMode != m_args.visMode)
    {
        m_args.visMode = visMode;
        m_accelBuilderNeedsUpdate = true;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::SetBackfaceVisibilityEnabled(bool enabled)
{
    if (enabled != m_args.enableBackfaceVisibility)
    {
        m_args.enableBackfaceVisibility = enabled;
        m_accelBuilderNeedsUpdate = true;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::SetQuantizationBits(int quantNBits)
{
    quantNBits = std::clamp(quantNBits, 0, 32);
    if (quantNBits != m_args.quantNBits)
    {
        m_args.quantNBits = quantNBits;
        m_accelBuilderNeedsUpdate = true;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::SetDisplacementScale(float scale)
{
    if (scale != m_renderParams.globalDisplacementScale)
    {
        m_renderParams.globalDisplacementScale = scale;
        m_accelBuilderNeedsUpdate = true;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::SetClusterTessellationPattern(ClusterPattern clusterPattern)
{
    m_renderParams.clusterPattern = uint32_t(clusterPattern);
    m_accelBuilderNeedsUpdate = true;
    GetRenderer().ResetSubframes();
    GetRenderer().ResetDenoiser();
}

void RTXMGDemoApp::SetAdaptiveTessellationMode(TessellatorConfig::AdaptiveTessellationMode mode)
{
    if (mode != m_args.tessMode)
    {
        m_args.tessMode = mode;
        m_accelBuilderNeedsUpdate = true;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::SetFrustumVisibilityEnabled(bool enabled)
{
    if (enabled != m_args.enableFrustumVisibility)
    {
        m_args.enableFrustumVisibility = enabled;
        m_accelBuilderNeedsUpdate = true;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::SetHiZVisibilityEnabled(bool enabled)
{
    if (enabled != m_args.enableHiZVisibility)
    {
        m_args.enableHiZVisibility = enabled;
        m_accelBuilderNeedsUpdate = true;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::SetUpdateTessellationCamera(bool update)
{
    if (update != m_args.updateTessCamera)
    {
        m_args.updateTessCamera = update;
        m_accelBuilderNeedsUpdate = true;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::SetDenoiserMode(DenoiserMode denoiserMode)
{
    if (denoiserMode != m_denoiserMode)
    {
        m_denoiserMode = denoiserMode;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::DumpFineTess()
{
    m_dumpFineTess = true;
    m_accelBuilderNeedsUpdate = true; // force a re-build
    GetRenderer().ResetSubframes(); // force a re-render
}

void RTXMGDemoApp::DoDumpFineTess(std::string const& filepath)
{
    if (filepath.empty())
    {
        static char const base_name[] = "rtxmg_fine_tess_";
        int index = GetUniqueFileIndex(base_name, ".ma");
        char buf[32];
        std::snprintf(buf, std::size(buf), "%s%04d.ma", base_name, index);
        DoDumpFineTess(buf);
    }
    else
    {
        auto logger = MayaLogger::Create(filepath.c_str());
        MayaLogger::ParticleDescriptor desc;

        auto& sceneAccels = GetRenderer().GetSceneAccels();
        desc.positions = sceneAccels->clusterVertexPositionsBuffer.Download(m_commandList);
        logger->CreateParticles(desc);
    }
}

void RTXMGDemoApp::SaveScreenshot()
{
    m_screenshot = true;
}

void RTXMGDemoApp::DoSaveScreenshot(nvrhi::ITexture* framebufferTexture, std::string const& filepath)
{
    if (filepath.empty())
    {
        static char const base_name[] = "rtxmg_screenshot_";
        int index = GetUniqueFileIndex(base_name, ".png");

        char buf[32];
        std::snprintf(buf, std::size(buf), "%s%04d.png", base_name, index);
        DoSaveScreenshot(framebufferTexture, buf);
    }
    else
    {
        SaveTextureToFile(GetDevice(), GetRenderer().GetCommonPasses().get(), framebufferTexture, nvrhi::ResourceStates::Unknown, filepath.c_str());
    }
}

void RTXMGDemoApp::DumpDebugBuffer()
{
    m_dumpDebugBuffer = true;
    m_accelBuilderNeedsUpdate = true; // force a re-build
    GetRenderer().ResetSubframes(); // force a re-render
}

void RTXMGDemoApp::DoDumpDebugBuffer(std::string const& filepath)
{
    if (filepath.empty())
    {
        static char const base_name[] = "rtxmg_debug_buffer_";
        int index = GetUniqueFileIndex(base_name, ".txt");
        char buf[32];
        std::snprintf(buf, std::size(buf), "%s%04d.txt", base_name, index);
        DoDumpDebugBuffer(buf);
    }
    else
    {
        std::vector<float4> debugContents = GetRenderer().GetAccelBuilder()->GetDebugBuffer().Download(m_commandList);

        log::info("accel builder debug contents: ");
        int idx = 0;
        for (const auto& v : debugContents)
        {
            log::info("  [%d]: %f %f %f %f", idx++, v.x, v.y, v.z, v.w);
            if (idx > 20) break;
        }
        std::ofstream debug_dump(filepath);
        idx = 0;
        for (const auto& v : debugContents)
        {
            char indexbuf[16];
            sprintf(indexbuf, "[%d]: ", idx++);
            debug_dump << std::setw(8) << std::right << indexbuf
                << std::setw(10) << std::left << v.x << " "
                << std::setw(10) << std::left << v.y << " "
                << std::setw(10) << std::left << v.z << " "
                << std::setw(10) << std::left << v.w << std::endl;
        }
    }
}

float RTXMGDemoApp::GetCPUFrameTime() const
{
    return std::chrono::duration<float, std::milli>(m_currFrameStart -
        m_prevFrameStart)
        .count();
}

void RTXMGDemoApp::SetWindowState(const WindowState &state) 
{
    GLFWwindow* window = GetDeviceManager()->GetWindow();
    
    // Prioritize commandline options and ignore restoring window state
    if (m_args.resolutionSetByCmdLine || m_args.startMaximized)
        return;

    if (all(state.windowSize > 0))
    {
        glfwSetWindowSize(window, state.windowSize.x, state.windowSize.y);
    }
    
    if (state.isMaximized)
    {
        glfwMaximizeWindow(window);
    }
    else if (state.isFullscreen)
    {
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    }
}
