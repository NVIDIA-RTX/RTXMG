
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

#include "rtxmg_demo_app.h"
#include "maya_logger.h"
#include "korgi.h"

#include "rtxmg/scene/texture_budget.h"

#include <donut/app/AftermathCrashDump.h>
#include <donut/app/ApplicationBase.h>
#include <donut/core/log.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/ThreadPool.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_set>

#if DONUT_WITH_DX12
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <nvrhi/d3d12.h>
#endif

#include <filesystem>
#include <iostream>
#include <fstream>
#include <utility>

#include "rtxmg/profiler/statistics.h"
#include "rtxmg/profiler/stats_dump.h"
#include "rtxmg/scene/scene.h"
#include "rtxmg/scene/json.h"   // readFile
#include "rtxmg/cluster_lod/baking/bake_progress.h"  // loading-bar progress for the metadata pre-build
#include "rtxmg/subdivision/subdivision_surface.h"
#include "rtxmg/cluster_tess/tessellator_config.h"
#include "rtxmg/utils/buffer.h"
#include "rtxmg/utils/cluster_operation.h"
#include "rtxmg/utils/debug.h"
#include "rtxmg/utils/system_memory.h"
#include "rtxmg/utils/texture_bytes.h"
#include "rtxmg/utils/vram_bytes.h"

#include "lerp_keyframes_params.h"

#define GLFW_INCLUDE_NONE // Do not include any OpenGL headers
#include <GLFW/glfw3.h>

#if DONUT_WITH_VULKAN
// Use donut's dispatcher; nothing links the Vulkan loader statically.
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif
#include <vulkan/vulkan.hpp>
#endif

#ifdef _WIN32
#include <Windows.h>
#include <DbgHelp.h>
#include <sstream>
#include <mutex>

// Capture a symbolic callstack string, skipping `framesToSkip` innermost frames.
static std::string CaptureCallstackString(int framesToSkip)
{
    static std::once_flag s_symInit;
    static std::mutex     s_symMutex;
    std::call_once(s_symInit, []()
    {
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
        SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    });

    void* frames[64];
    int captured = CaptureStackBackTrace(framesToSkip, 64, frames, nullptr);

    HANDLE hProcess = GetCurrentProcess();
    char symBuf[sizeof(SYMBOL_INFO) + MAX_SYM_NAME];
    SYMBOL_INFO* sym = reinterpret_cast<SYMBOL_INFO*>(symBuf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen   = MAX_SYM_NAME;

    IMAGEHLP_LINE64 lineInfo = {};
    lineInfo.SizeOfStruct = sizeof(IMAGEHLP_LINE64);

    std::ostringstream ss;
    std::lock_guard<std::mutex> lock(s_symMutex);
    for (int i = 0; i < captured; ++i)
    {
        DWORD64 addr = reinterpret_cast<DWORD64>(frames[i]);
        ss << "  [" << i << "] ";
        DWORD64 symDisp = 0;
        if (SymFromAddr(hProcess, addr, &symDisp, sym))
        {
            ss << sym->Name;
            DWORD lineDisp = 0;
            if (SymGetLineFromAddr64(hProcess, addr, &lineDisp, &lineInfo))
                ss << " (" << lineInfo.FileName << ":" << lineInfo.LineNumber << ")";
        }
        else
        {
            ss << "0x" << std::hex << addr;
        }
        ss << "\n";
    }
    return ss.str();
}
#endif // _WIN32

using namespace donut;

namespace fs = std::filesystem;

// Sample application, so use dummy app id
static constexpr int kStreamlineAppId = 1;

static constexpr char const* kDxrRequirements =
    "DXR clusters need an NVIDIA GPU with driver 621.01 or later and Windows Developer Mode enabled. "
    "Run with --cluster-api nvapi or -vk instead.";

static const char* GetClusterApiName(nvrhi::IDevice* device)
{
    switch (GetClusterApi(device))
    {
    case nvrhi::ClusterApi::Nv:
        return device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN ? "VK_NV_cluster_acceleration_structure" : "NVAPI";
    case nvrhi::ClusterApi::Dxr:
        return "DXR clusters";
    default:
        return "none";
    }
}

static void RequireDxrSupport(nvrhi::IDevice* device, Args::ClusterApiRequest request)
{
    const nvrhi::ClusterApi api = GetClusterApi(device);
    if (request == Args::ClusterApiRequest::Dxr && api != nvrhi::ClusterApi::Dxr)
    {
#if !NVRHI_D3D12_WITH_DXRCLUSTER
        log::fatal("--cluster-api dxr needs a build configured with -DNVRHI_D3D12_WITH_DXRCLUSTER=ON");
#endif
        log::fatal("The device does not support DXR clusters.\n\n%s", kDxrRequirements);
    }
    // The experimental SM 6.10 is only reported when Developer Mode let it be enabled.
    if (api == nvrhi::ClusterApi::Dxr && !device->queryFeatureSupport(nvrhi::Feature::RayTracingPositionFetch))
        log::fatal("Shader Model 6.10 is not available.\n\n%s", kDxrRequirements);
}

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

// Set once we see nvrhi's "Device Removed!" message. After that, all Error and
// Fatal log messages are dropped — both nvrhi's downstream cascade (validation
// failures, fence timeouts) and any direct donut::log::error/fatal calls from
// app code that's still trying to render against a dead device. Suppressing
// Fatal also prevents donut's DefaultCallback from calling abort(), which would
// kill Aftermath's crash-dump writer mid-write.
static std::atomic<bool> g_DeviceRemoved{false};
static std::atomic<bool> g_AftermathEnabled{false};

// Max wall-clock time to wait after device-removed before forcing process
// exit. The watchdog returns early once Aftermath signals dump-complete.
static constexpr uint32_t kDeviceRemovedExitTimeoutSeconds = 5;

void RTXMGDemoApp::InstallDeviceRemovedAwareLogCallback()
{
    static std::once_flag s_once;
    std::call_once(s_once, []() {
        donut::log::Callback prev = donut::log::GetCallback();
        donut::log::SetCallback([prev](donut::log::Severity sev, char const* msg) {
            if (g_DeviceRemoved.load(std::memory_order_relaxed) &&
                (sev == donut::log::Severity::Error || sev == donut::log::Severity::Fatal))
            {
                return;
            }
            prev(sev, msg);
        });
    });
}

// Spawn a one-shot detached watchdog that terminates the process once Aftermath
// has finished writing its crash dump, or after a short timeout if the dump
// never starts / isn't being collected. Calling code must guarantee single-shot
// (we use the false→true transition on g_DeviceRemoved for that).
static void StartDeviceRemovedWatchdog()
{
    std::thread([]() {
        using namespace std::chrono;
        const auto timeout = seconds(kDeviceRemovedExitTimeoutSeconds);

        if (g_AftermathEnabled.load(std::memory_order_relaxed))
        {
            // Polls GFSDK_Aftermath_GetCrashDumpStatus; returns as soon as the
            // dump is Finished, or after the timeout if the status never
            // advances (e.g. Aftermath didn't catch this particular hang).
            donut::app::AftermathCrashDump::WaitForCrashDump(
                static_cast<uint32_t>(timeout.count()));
        }
        else
        {
            std::this_thread::sleep_for(timeout);
        }

        // _Exit skips static destructors. Most of those would try to release
        // D3D12 / VK objects against a dead device, which is at best wasted
        // work and at worst a deadlock.
        std::_Exit(1);
    }).detach();
}

void RTXMGDemoApp::MessageCallback::message(nvrhi::MessageSeverity severity, const char* messageText)
{
    const bool isError = severity == nvrhi::MessageSeverity::Error ||
                         severity == nvrhi::MessageSeverity::Fatal;
    if (g_DeviceRemoved.load(std::memory_order_relaxed) && isError)
    {
        return;
    }

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

#ifdef _WIN32
    if (severity == nvrhi::MessageSeverity::Error || severity == nvrhi::MessageSeverity::Fatal)
    {
        std::string callstack = CaptureCallstackString(2); // skip CaptureCallstackString + this function
        donut::log::message(donutSeverity, "Callstack:\n%s", callstack.c_str());
    }
#endif

    // nvrhi emits "Device Removed!" from both d3d12-device and vulkan-queue
    // when GetDeviceRemovedReason / vkQueueSubmit fail. Match on the substring
    // to cover either backend.
    if (isError && messageText && std::strstr(messageText, "Device Removed"))
    {
        const bool firstTime = !g_DeviceRemoved.exchange(true, std::memory_order_relaxed);
        if (firstTime)
        {
            StartDeviceRemovedWatchdog();
        }
    }
}

void RTXMGDemoApp::UpdateParams()
{
    m_denoiserMode = m_args.enableDenoiser ? DenoiserMode::DlssRr : DenoiserMode::None;

    // --dlssMode override (e.g. DLAA = 1:1 render-to-display so -dp pixel
    // coords match the screenshot pixel grid exactly).
    if (m_args.dlssModeSet)
        m_ui.dlssMode = m_args.dlssMode;

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

    m_renderParams.isolationLevel = m_args.globalIsolationLevel;
    m_renderParams.clusterPattern = uint32_t(m_args.clusterPattern);
    m_renderParams.globalDisplacementScale = m_args.dispScale;

    m_renderParams.hasEnvironmentMap = 0;
    m_renderParams.enableEnvmapHeatmap = 0;
    // (-1,-1) is the "no pixel selected" sentinel: (0,0) is a valid
    // DispatchRaysIndex, so the probe would fire at the top-left pixel of
    // every frame.  -dp is applied at the top of Render() on the exit frame.
    m_renderParams.debugPixel = int2(-1, -1);
    m_renderParams.selectedClusterLodGeometry = -1;
    m_renderParams.selectedClusterLodInstance = -1;
    m_renderParams.selectedClusterLodLevel    = -1;
    m_renderParams.selectedSubdMesh           = -1;
    m_renderParams.debugSurfaceIndex = m_debugSurfaceClusterLaneIndex[0]; // Initialize from GUI variable

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
        GetRenderer().SetAutoExposure(m_args.autoExposure);
        GetRenderer().SetLodPixelError(m_args.lodPixelError);
        GetRenderer().SetAdaptiveLodError(m_args.adaptiveLodError);
        // Must be set BEFORE SceneFinishedLoading; CreateAccelStructs branches on it.
        GetRenderer().SetUseStreaming(!m_args.usePreload);
        GetRenderer().SetUseLinearClasAllocator(m_args.useLinearClasAllocator);
        GetRenderer().SetUseBlasSharing(m_args.useBlasSharing);
        GetRenderer().SetBlasSharingEnabledLevels(m_args.blasSharingEnabledLevels);
        GetRenderer().SetUseBlasCaching(m_args.useBlasCaching);
        GetRenderer().SetBlasCachingEnabledLevels(m_args.blasCachingEnabledLevels);
        GetRenderer().SetUseBlasMerging(m_args.useBlasMerging);
        GetRenderer().SetUseCulling(m_args.useCulling);
        GetRenderer().SetUseHardCull(m_args.useHardCull);
        GetRenderer().SetHardCullForcesInvisible(m_args.hardCullForcesInvisible);
        GetRenderer().SetUseHizOcclusion(m_args.useHizOcclusion);
        GetRenderer().SetClasPositionTruncateBits(m_args.clasPositionTruncateBits);
        GetRenderer().SetDebugClusterLod(m_args.debugClusterLod);
        // Synced before the first streaming init: decides whether resident
        // group blobs keep their normal words (-cvn / Vertex Normals toggle).
        GetRenderer().SetEnableClusterLodVertexNormals(m_args.enableClusterLodVertexNormals);
        // Shading needs the maps to have been loaded, so it is the AND of the two.
        GetRenderer().SetEnableClusterLodNormalMaps(m_args.enableNormalMaps &&
                                                    m_args.normalMapShading);
        GetRenderer().SetStripResidentData(m_args.stripResidentData);
        GetRenderer().SetStripResidentPositions(m_args.stripResidentPositions);
        GetRenderer().SetDebugSurfaceIndex(m_debugSurfaceClusterLaneIndex[0]); // Initialize debug surface highlighting
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
    std::string &windowTitle,
    int argc, const char** argv)
    : app::ApplicationBase(deviceManager)
    , m_messageCallback(deviceManager)
{
    InstallDeviceRemovedAwareLogCallback();

    m_argc = argc;
    m_argv = argv;

    m_binaryPath = app::GetDirectoryWithExecutable().lexically_normal();
    m_mediaPath = findMediaFolder(m_binaryPath, "assets");

    m_args.Parse(m_argc, m_argv);

    stats::frameLog.enabled = !m_args.dumpStatsFile.empty();

    if (!m_args.shotListFile.empty())
    {
        if (!LoadShotList(m_args.shotListFile, m_shotPositions, m_shotConfigs))
            std::exit(EXIT_FAILURE);  // a malformed list must not silently render nothing
        // Loading normal maps is a scene-reload setting, so a list that toggles
        // shading is inert without --normalmaps -- and an inert sweep renders N
        // identical configs that look like a real result.
        if (!m_args.enableNormalMaps)
            for (const ShotConfig& config : m_shotConfigs)
                if (config.normalMapShading > 0)
                {
                    log::warning("--shot-list '%s' sets normalMapShading, but --no-normalmaps "
                                 "left nothing to shade with.", config.name.c_str());
                    break;
                }
        if (m_args.shotOutDir.empty())
            m_args.shotOutDir =
                std::filesystem::path(m_args.shotListFile).parent_path().generic_string();
        // The published report layout: <dir>/images and <dir>/stats beside the
        // page, so a run writes straight into it with nothing to move after.
        std::filesystem::create_directories(ShotImageDir());
        std::filesystem::create_directories(ShotStatsDir());

        // The shot list owns the exit: -nf would otherwise cut the walk short
        // and leave a partial golden set that looks like a completed run.
        if (m_args.exitAfterFrame >= 0)
        {
            log::warning("--shot-list ignores -nf %d; the walk exits when the last "
                         "capture point is written.", m_args.exitAfterFrame);
            m_args.exitAfterFrame = -1;
        }
    }

    // Media-folder resolution (see Args::mediaPath):
    //   1. explicit --media <dir> wins;
    //   2. else an ABSOLUTE -mf implies its parent dir (so a scene anywhere on
    //      disk resolves its sibling assets without a separate flag).  Reduce
    //      meshInputFile to the bare filename so the
    //      "GetMediaPath()/meshInputFile" joins stay correct.
    //      Relative -mf is left alone so the bundled-asset + envmap workflow (all
    //      under assets/) keeps using the exe-adjacent assets folder.
    if (!m_args.mediaPath.empty())
    {
        m_mediaPath = std::filesystem::path(m_args.mediaPath).lexically_normal();
    }
    else if (!m_args.meshInputFile.empty())
    {
        std::filesystem::path mf(m_args.meshInputFile);
        if (mf.is_absolute() && std::filesystem::exists(mf))
        {
            m_mediaPath = mf.parent_path().lexically_normal();
            m_args.meshInputFile = mf.filename().generic_string();
        }
    }

    m_displaySize = int2(m_args.width, m_args.height);

    UpdateParams();

    app::DeviceCreationParameters deviceParams;
    deviceParams.enableHeapDirectlyIndexed = true; // Needed for bindless look up in motion_vectors.hlsl
    deviceParams.enableRayTracingExtensions = true;
    deviceParams.backBufferWidth = m_displaySize.x;
    deviceParams.backBufferHeight = m_displaySize.y;
    deviceParams.enablePerMonitorDPI = true;
    deviceParams.startMaximized = m_args.startMaximized;
    deviceParams.startFullscreen = m_args.startFullscreen;
    deviceParams.swapChainFormat = nvrhi::Format::RGBA8_UNORM;
    deviceParams.vsyncEnabled = m_args.vsync;
    deviceParams.messageCallback = &m_messageCallback;
#if DONUT_WITH_VULKAN
    // Optional, so a driver without it just hides the VRAM window's driver row.
    deviceParams.optionalVulkanDeviceExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
#endif
    if (m_args.debug)
    {
        deviceParams.enableDebugRuntime = true;
        deviceParams.enableNvrhiValidationLayer = true;
    }

    if (m_args.gpuValidation)
    {
        deviceParams.enableGPUValidation = true;
        deviceParams.enableRayTracingValidation = true;
    }

    deviceParams.enableDxrClusters = m_args.clusterApi != Args::ClusterApiRequest::Nvapi;
    deviceParams.enableNvapiClusters = m_args.clusterApi != Args::ClusterApiRequest::Dxr;

#if NVRHI_D3D12_WITH_DXRCLUSTER
    // NVAPI cluster intrinsics return invalid data under GPU-based validation, so the NVAPI path
    // uses the DXR cluster intrinsics instead, when Developer Mode allows them.
    if (deviceManager->GetGraphicsAPI() == nvrhi::GraphicsAPI::D3D12 && m_args.gpuValidation &&
        m_args.clusterApi == Args::ClusterApiRequest::Nvapi)
    {
        m_forceDxrClusterIntrinsics = nvrhi::d3d12::enableExperimentalFeatures(&m_messageCallback);
        if (!m_forceDxrClusterIntrinsics)
            log::warning("NVAPI cluster intrinsics are unreliable under GPU-based validation (-gd); "
                         "enable Windows Developer Mode to avoid corrupted rendering.");
    }
#endif

    if (m_args.aftermath)
    {
#if DONUT_WITH_AFTERMATH
        deviceParams.enableAftermath = true;
        g_AftermathEnabled.store(true, std::memory_order_relaxed);
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
        windowTitle.c_str()))
    {
#if NVRHI_D3D12_WITH_DXRCLUSTER
        if (deviceManager->GetGraphicsAPI() == nvrhi::GraphicsAPI::D3D12 && deviceParams.enableDxrClusters)
            log::fatal("Cannot initialize a graphics device.\n\n%s", kDxrRequirements);
#endif
        log::fatal(
            "Cannot initialize a graphics device with the requested parameters");
    }

#if DONUT_WITH_DX12
    // Disable D3D12 InfoQueue break-on-severity (donut's default Debug-build
    // wiring enables it, causing 0x087A exceptions on the first validation
    // error and killing the process before the message is logged).  Replace
    // with an ID3D12InfoQueue1::RegisterMessageCallback that routes every
    // validation message through donut::log::error so we can read what went
    // wrong.  Only do this in --debug; InfoQueue exists only when the debug
    // runtime is enabled.
    if (m_args.debug && deviceManager->GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
    {
        ID3D12Device* d3dDev = (ID3D12Device*)deviceManager->GetDevice()->getNativeObject(
            nvrhi::ObjectTypes::D3D12_Device);
        ID3D12InfoQueue* iq = nullptr;
        if (d3dDev && SUCCEEDED(d3dDev->QueryInterface(__uuidof(ID3D12InfoQueue), (void**)&iq)) && iq)
        {
            iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR,      FALSE);
            iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, FALSE);
            iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING,    FALSE);

            ID3D12InfoQueue1* iq1 = nullptr;
            if (SUCCEEDED(iq->QueryInterface(__uuidof(ID3D12InfoQueue1), (void**)&iq1)) && iq1)
            {
                static auto cb = [](D3D12_MESSAGE_CATEGORY,
                                    D3D12_MESSAGE_SEVERITY severity,
                                    D3D12_MESSAGE_ID,
                                    LPCSTR description,
                                    void*) {
                    const char* sev =
                        severity == D3D12_MESSAGE_SEVERITY_CORRUPTION ? "CORRUPTION" :
                        severity == D3D12_MESSAGE_SEVERITY_ERROR      ? "ERROR"      :
                        severity == D3D12_MESSAGE_SEVERITY_WARNING    ? "WARNING"    :
                        severity == D3D12_MESSAGE_SEVERITY_INFO       ? "INFO"       :
                                                                        "MSG";
                    donut::log::error("[D3D12 %s] %s", sev, description);
                };
                DWORD cookie = 0;
                iq1->RegisterMessageCallback(cb,
                                             D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                                             nullptr, &cookie);
                iq1->Release();
                donut::log::info("D3D12 InfoQueue1 message callback registered (break-on-severity disabled)");
            }
            else
            {
                donut::log::info("D3D12 InfoQueue (no Info1 interface - messages still go via OutputDebugString)");
            }
            iq->Release();
        }
    }
#endif

    if (deviceManager->GetGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
        RequireDxrSupport(deviceManager->GetDevice(), m_args.clusterApi);

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

    {
#if defined(_M_ARM64) || defined(__aarch64__)
        const char* arch = "arm64";
#else
        const char* arch = "x64";
#endif
        const char* clusterApi = GetClusterApiName(deviceManager->GetDevice());
        log::info("Cluster acceleration structures: %s", clusterApi);

        if (deviceManager->GetGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
            windowTitle += std::string(" (D3D12 ") + clusterApi + " " + arch + ")";
        else
            windowTitle += std::string(" (VULKAN ") + arch + ")";
        deviceManager->SetWindowTitle(windowTitle.c_str());
    }

    if (!deviceManager->GetDevice()->queryFeatureSupport(
        nvrhi::Feature::HeapDirectlyIndexed))
    {
        log::fatal("The graphics device does not support directly indexing heaps (ResourceDescriptorHeap) (SamplerDescriptorHeap)");
    }

    {
        // The wave-dependent compute shaders assume kWaveSize lanes and pin it with
        // [WaveSize] on D3D12, which needs the device to support that width. Vulkan
        // reports a single subgroup size, so there it must already be kWaveSize.
        nvrhi::WaveLaneCountMinMaxFeatureInfo waveLanes = {};
        deviceManager->GetDevice()->queryFeatureSupport(nvrhi::Feature::WaveLaneCountMinMax,
                                                        &waveLanes, sizeof(waveLanes));
        if (waveLanes.minWaveLaneCount > shaderio::kWaveSize || waveLanes.maxWaveLaneCount < shaderio::kWaveSize)
        {
            log::fatal("The graphics device reports a wave size of %u..%u; the cluster-LOD "
                       "shaders require %u",
                       waveLanes.minWaveLaneCount, waveLanes.maxWaveLaneCount, shaderio::kWaveSize);
        }
    }

    korgi::Init();
}

RTXMGDemoApp::~RTXMGDemoApp()
{
    // Closing the window mid-load leaves the load thread running, and it uses
    // members that reverse-declaration order would destroy first (m_textureLoadPool
    // above all).  Ask it to stop, then join before anything is torn down.
    m_loadCancelled.store(true, std::memory_order_relaxed);
    if (m_loadResult.valid())
        m_loadResult.wait();

    korgi::Shutdown();
}

RTXMGRenderer& RTXMGDemoApp::GetRenderer()
{
    if (!m_renderer)
    {
        RTXMGRenderer::Options rendererOptions{
            .params = m_renderParams,
            .device = GetDevice(),
            .forceDxrClusterIntrinsics = m_forceDxrClusterIntrinsics
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

    app::AdapterInfo::LUID luid = {};
    app::AdapterInfo::UUID uuid = {};

    const app::AdapterInfo* pAdapter = nullptr;

#if DONUT_WITH_DX12
    if (GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
    {
        ID3D12Device* rawDevice = (ID3D12Device*)GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        LUID dxLuid = rawDevice->GetAdapterLuid();
        static_assert(luid.size() == sizeof(dxLuid));
        memcpy(luid.data(), &dxLuid, luid.size());

        // The unique/total triangle stats need a 64-bit atomic on the
        // descriptor-table-bound counters buffer; without it they read garbage,
        // so gate them off rather than displaying it.
        D3D12_FEATURE_DATA_D3D12_OPTIONS11 options11 = {};
        if (SUCCEEDED(rawDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS11, &options11, sizeof(options11))))
        {
            const bool supported = options11.AtomicInt64OnDescriptorHeapResourceSupported;
            GetRenderer().SetAtomicInt64OnHeapSupported(supported);
            donut::log::info("D3D12 AtomicInt64OnDescriptorHeapResourceSupported = %s%s",
                             supported ? "true" : "false",
                             supported ? "" : " (render-stats triangle tallies disabled)");
        }

        for (const auto& adapter : adapters)
        {
            if (adapter.luid == luid)
            {
                pAdapter = &adapter;
                break;
            }
        }
    }
#endif
#if DONUT_WITH_VULKAN
    if (GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
    {
        vk::PhysicalDevice rawDevice = (VkPhysicalDevice)GetDevice()->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice).pointer;
        vk::PhysicalDeviceProperties2 properties2;
        vk::PhysicalDeviceIDProperties idProperties;
        properties2.pNext = &idProperties;
        rawDevice.getProperties2(&properties2);

        // Vulkan analogue of the D3D12 gate above: no descriptor-heap
        // restriction, just the 64-bit buffer atomics themselves.
        vk::PhysicalDeviceFeatures2 features2;
        vk::PhysicalDeviceVulkan12Features vk12Features;
        features2.pNext = &vk12Features;
        rawDevice.getFeatures2(&features2);
        const bool supported = vk12Features.shaderBufferInt64Atomics == VK_TRUE;
        GetRenderer().SetAtomicInt64OnHeapSupported(supported);
        donut::log::info("Vulkan shaderBufferInt64Atomics = %s%s",
                         supported ? "true" : "false",
                         supported ? "" : " (render-stats triangle tallies disabled)");

        // Device support stands in for "enabled": donut enables every optional
        // device extension the physical device reports, and this one is requested
        // optional at device creation.
        for (const vk::ExtensionProperties& ext : rawDevice.enumerateDeviceExtensionProperties())
        {
            if (strcmp(ext.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0)
            {
                m_vkMemoryBudgetAvailable = true;
                break;
            }
        }

        app::AdapterInfo::UUID uuid;
        static_assert(uuid.size() == idProperties.deviceUUID.size());
        memcpy(uuid.data(), idProperties.deviceUUID.data(), uuid.size());

        for (const auto& adapter : adapters)
        {
            if (adapter.uuid == uuid)
            {
                pAdapter = &adapter;
                break;
            }
        }
    }
#endif
    
    if (!pAdapter)
    {
        donut::log::fatal("Failed to find active adapter for vram check");
    }

    size_t vram = pAdapter->dedicatedVideoMemory;
    m_physicalVramBytes = vram;
    m_adapterName = pAdapter->name;
#if DONUT_WITH_DX12
    if (pAdapter->dxgiAdapter)
        pAdapter->dxgiAdapter->QueryInterface(IID_PPV_ARGS(&m_dxgiAdapter3));
#endif
    if (m_args.vramOverrideMB > 0)
        donut::log::info("VRAM budgeting simulated at %d MB (real: %llu MB); allocations are unaffected.",
                         m_args.vramOverrideMB, (unsigned long long)(vram >> 20));

    if (vram < kMinVram * kGigabyte)
    {
        donut::log::error("GPU has %.2fGB of VRAM and is below the required %dGB.\n\n"
            "Expect the following:\n"
            "1. Performance degradation or out of memory crashes.\n"
            "2. Flickering and missing surfaces after adjusting the memory budget down", float(vram) / kGigabyte, kMinVram);
    }
        
    std::filesystem::path sceneFileName;
    if (!m_args.meshInputFile.empty())
    {
        sceneFileName = app::GetDirectoryWithExecutable().parent_path() / "assets" /
            m_args.meshInputFile;

        if (!std::filesystem::exists(sceneFileName))
        {
            std::filesystem::path mediaSceneFileName = GetMediaPath() / m_args.meshInputFile;
            if (mediaSceneFileName != sceneFileName && std::filesystem::exists(mediaSceneFileName))
            {
                sceneFileName = mediaSceneFileName;
            }
            else
            {
                // Fail fast rather than falling back to a default scene, which
                // would mask a mistyped path / wrong CWD.
                donut::log::fatal("Could not find mesh input file for '-mf %s' (looked in '%s'%s%s)",
                                  m_args.meshInputFile.c_str(),
                                  sceneFileName.generic_string().c_str(),
                                  (mediaSceneFileName != sceneFileName) ? " and " : "",
                                  (mediaSceneFileName != sceneFileName)
                                      ? mediaSceneFileName.generic_string().c_str() : "");
            }
        }
    }

    RTXMGRenderer& renderer = GetRenderer();
    m_TextureCache = renderer.GetTextureCache();
    m_CommonPasses = renderer.GetCommonPasses();
    m_commandList = GetDevice()->createCommandList();

    // One loading path for interactive and headless (-nf) runs alike, so the
    // loading screen is always visible.  -nf frame counts stay deterministic
    // because the exit / screenshot / pixel-debug checks count from
    // m_postLoadBaseFrame, not frame 0.
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

    m_sunLight = std::make_shared<engine::DirectionalLight>();
    m_sunLight->angularSize = 0.53f;
    m_sunLight->irradiance = 3.f;
    m_sunLightNode = std::make_shared<engine::SceneGraphNode>();
    m_sunLightNode->SetLeaf(m_sunLight);

    // Capture UI-edited state BEFORE the load, because the scene file and the
    // command-line re-parse snap it back to launch values; ReconcileLoadedScene
    // restores it, and LoadScene reads the grid pair straight from here.  On the
    // FIRST load the renderer still holds its compile-time defaults (the CLI
    // values have not reached it yet), so the command line is only honoured by
    // capturing from m_args there.
    m_savedGridCopies    = m_args.lodGridCopies;
    m_savedGridGap       = m_args.lodGridGap;
    const bool firstLoad = (m_scene == nullptr);
    const bool isSameScene = !firstLoad && (sceneFileName.empty() || sceneFileName == m_args.meshInputFile);
    if (isSameScene)
    {
        m_savedCamEye    = m_camera.GetEye();
        m_savedCamLookat = m_camera.GetLookat();
        m_savedCamUp     = m_camera.GetUp();
        m_savedCamFovY   = m_camera.GetFovY();
        m_hasSavedCamera = true;
    }
    m_savedBlasSharing   = firstLoad ? m_args.useBlasSharing           : renderer.GetUseBlasSharing();
    m_savedBlasCaching   = firstLoad ? m_args.useBlasCaching           : renderer.GetUseBlasCaching();
    m_savedBlasMerging   = firstLoad ? m_args.useBlasMerging           : renderer.GetUseBlasMerging();
    m_savedSharingLevels = firstLoad ? m_args.blasSharingEnabledLevels : renderer.GetBlasSharingEnabledLevels();
    m_savedCachingLevels = firstLoad ? m_args.blasCachingEnabledLevels : renderer.GetBlasCachingEnabledLevels();

    // Ensure GetCurrentBakerConfig() resolves the correct JSON path for this
    // scene even before ReconcileLoadedScene updates meshInputFile from the scene object.
    if (!sceneFileName.empty())
        m_args.meshInputFile = sceneFileName;

    // ApplicationBase::BeginLoadingScene's prologue, done here so the pool drain
    // and the cache reset stay in the right order: a decode task still running
    // from a previous load would otherwise push into the just-cleared queue.
    if (m_textureLoadPool)
        m_textureLoadPool->WaitForTasks();
    if (m_TextureCache)
        m_TextureCache->Reset();
    GetDevice()->waitForIdle();
    GetDevice()->runGarbageCollection();

    // Force the renderer into existence before the load thread touches it:
    // GetRenderer() constructs on first use and is not thread-safe.
    GetRenderer();

    m_pendingSceneFs   = nativeFS;
    m_pendingSceneFile = sceneFileName;
    m_loadPhase        = LoadPhase::Import;
    m_postLoadBaseFrame = -1;  // rebase -nf counting once loading completes

    // Returns immediately: the CPU half of the load (incl. the cluster-LOD bake)
    // runs on this thread while AdvanceLoad drives the rest a step per frame.
    m_loadResult = std::async(std::launch::async,
        [this, nativeFS, sceneFileName]() { return LoadScene(nativeFS, sceneFileName); });
}

// ---------------------------------------------------------------------------
// Bake config JSON persistence
// ---------------------------------------------------------------------------

static Json::Value BakerConfigToJson(const BakerConfig& c)
{
    Json::Value v;
    v["clusterVertices"]           = c.clusterVertices;
    v["clusterTriangles"]          = c.clusterTriangles;
    v["clusterGroupSize"]          = c.clusterGroupSize;
    v["preferredNodeWidth"]        = c.preferredNodeWidth;
    v["lodErrorMergePrevious"]     = c.lodErrorMergePrevious;
    v["lodErrorMergeAdditive"]     = c.lodErrorMergeAdditive;
    v["lodErrorEdgeLimit"]         = c.lodErrorEdgeLimit;
    v["meshoptPreferRayTracing"]   = c.meshoptPreferRayTracing;
    v["meshoptFillWeight"]         = c.meshoptFillWeight;
    v["meshoptSplitFactor"]        = c.meshoptSplitFactor;
    v["useCompressedData"]         = c.useCompressedData;
    v["compressionPosDropBits"]    = c.compressionPosDropBits;
    v["compressionTexDropBits"]    = c.compressionTexDropBits;
    v["simplifyNormalWeight"]      = c.simplifyNormalWeight;
    v["simplifyTexCoordWeight"]    = c.simplifyTexCoordWeight;
    v["simplifyTangentWeight"]     = c.simplifyTangentWeight;
    v["simplifyTangentSignWeight"] = c.simplifyTangentSignWeight;
    v["simplifyMaterialWeight"]    = c.simplifyMaterialWeight;
    v["quantizeTexCoords"]         = c.quantizeTexCoords;
    return v;
}

static BakerConfig BakerConfigFromJson(const Json::Value& v)
{
    BakerConfig c;
    if (v.isMember("clusterVertices"))          c.clusterVertices          = v["clusterVertices"].asUInt();
    if (v.isMember("clusterTriangles"))         c.clusterTriangles         = v["clusterTriangles"].asUInt();
    if (v.isMember("clusterGroupSize"))         c.clusterGroupSize         = v["clusterGroupSize"].asUInt();
    if (v.isMember("preferredNodeWidth"))       c.preferredNodeWidth       = v["preferredNodeWidth"].asUInt();
    if (v.isMember("lodErrorMergePrevious"))    c.lodErrorMergePrevious    = v["lodErrorMergePrevious"].asFloat();
    if (v.isMember("lodErrorMergeAdditive"))    c.lodErrorMergeAdditive    = v["lodErrorMergeAdditive"].asFloat();
    if (v.isMember("lodErrorEdgeLimit"))        c.lodErrorEdgeLimit        = v["lodErrorEdgeLimit"].asFloat();
    if (v.isMember("meshoptPreferRayTracing"))  c.meshoptPreferRayTracing  = v["meshoptPreferRayTracing"].asBool();
    if (v.isMember("meshoptFillWeight"))        c.meshoptFillWeight        = v["meshoptFillWeight"].asFloat();
    if (v.isMember("meshoptSplitFactor"))       c.meshoptSplitFactor       = v["meshoptSplitFactor"].asFloat();
    if (v.isMember("useCompressedData"))        c.useCompressedData        = v["useCompressedData"].asBool();
    if (v.isMember("compressionPosDropBits"))   c.compressionPosDropBits   = v["compressionPosDropBits"].asUInt();
    if (v.isMember("compressionTexDropBits"))   c.compressionTexDropBits   = v["compressionTexDropBits"].asUInt();
    if (v.isMember("simplifyNormalWeight"))     c.simplifyNormalWeight     = v["simplifyNormalWeight"].asFloat();
    if (v.isMember("simplifyTexCoordWeight"))   c.simplifyTexCoordWeight   = v["simplifyTexCoordWeight"].asFloat();
    if (v.isMember("simplifyTangentWeight"))    c.simplifyTangentWeight    = v["simplifyTangentWeight"].asFloat();
    if (v.isMember("simplifyTangentSignWeight"))c.simplifyTangentSignWeight= v["simplifyTangentSignWeight"].asFloat();
    if (v.isMember("simplifyMaterialWeight"))   c.simplifyMaterialWeight   = v["simplifyMaterialWeight"].asFloat();
    if (v.isMember("quantizeTexCoords"))        c.quantizeTexCoords        = v["quantizeTexCoords"].asBool();
    return c;
}

fs::path RTXMGDemoApp::GetBakeConfigJsonPath() const
{
    fs::path cacheDir;
    if (!m_args.clusterCacheDir.empty())
        cacheDir = m_args.clusterCacheDir;
    else if (!m_args.meshInputFile.empty())
        cacheDir = fs::path(m_args.meshInputFile).parent_path() / "_nvsngeocache";
    else
        return {};
    return cacheDir / "bake_config.json";
}

// Layers: BakerConfig defaults < bake_config.json < explicitly-set CLI args.
BakerConfig RTXMGDemoApp::GetCurrentBakerConfig() const
{
    BakerConfig cfg{};

    const fs::path jsonPath = GetBakeConfigJsonPath();
    if (!jsonPath.empty() && fs::exists(jsonPath))
    {
        try   { cfg = BakerConfigFromJson(readFile(jsonPath)); }
        catch (const std::exception& e)
        {
            log::warning("Failed to read bake config '%s': %s",
                         jsonPath.string().c_str(), e.what());
        }
    }

    // CLI sentinels: 0 / negative = not explicitly set; bool defaults differ
    if (m_args.compressClusterData)            cfg.useCompressedData      = true;
    if (m_args.clusterTriangles)               cfg.clusterTriangles       = m_args.clusterTriangles;
    if (m_args.clusterVertices)                cfg.clusterVertices        = m_args.clusterVertices;
    if (m_args.simplifyNormalWeight   >= 0.f)  cfg.simplifyNormalWeight   = m_args.simplifyNormalWeight;
    if (m_args.simplifyTexCoordWeight >= 0.f)  cfg.simplifyTexCoordWeight = m_args.simplifyTexCoordWeight;
    if (m_args.lodErrorMergePrevious  >= 0.f)  cfg.lodErrorMergePrevious  = m_args.lodErrorMergePrevious;
    if (m_args.lodErrorMergeAdditive  >= 0.f)  cfg.lodErrorMergeAdditive  = m_args.lodErrorMergeAdditive;
    if (!m_args.quantizeTexCoords)             cfg.quantizeTexCoords      = false;  // --no-uvquant

    return cfg;
}

void RTXMGDemoApp::SaveBakerConfigJson(const BakerConfig& cfg)
{
    const fs::path jsonPath = GetBakeConfigJsonPath();
    if (jsonPath.empty())
        return;
    try
    {
        fs::create_directories(jsonPath.parent_path());
        Json::StreamWriterBuilder builder;
        builder["indentation"] = "  ";
        std::ofstream f(jsonPath);
        f << Json::writeString(builder, BakerConfigToJson(cfg));
        log::info("Saved bake config to '%s'", jsonPath.string().c_str());
    }
    catch (const std::exception& e)
    {
        log::warning("Failed to save bake config '%s': %s",
                     jsonPath.string().c_str(), e.what());
    }
}

void RTXMGDemoApp::ApplyBakerConfigAndReload(const BakerConfig& cfg)
{
    // Snapshot the current scene's bake stats as the "before" baseline.
    m_pendingBakeReport = BakeReport{};
    {
        const stats::BakeStats& s = stats::bakeStats;
        auto& b = m_pendingBakeReport.before;
        b.valid      = s.geometries > 0;
        b.geometries = s.geometries;
        b.groups     = s.groups;
        b.clusters   = s.clusters;
        b.triangles  = s.triangles;
        b.bakedBytes  = s.bakedBytes;
        b.deviceBytes = s.deviceBytes;
        b.compressed  = s.compressed;
        b.quantizedUv = s.quantizedUv;
        const BakerConfig beforeCfg = GetCurrentBakerConfig();
        b.compressionPosDropBits = beforeCfg.compressionPosDropBits;
        b.compressionTexDropBits = beforeCfg.compressionTexDropBits;
    }
    m_rebakeTriggered = true;
    SaveBakerConfigJson(cfg);
    ReloadCurrentScene();
}

// donut's Render() models loading as one blocking LoadScene() and decides "all
// textures finalized" by draining a queue our textures are not in yet, so we
// drive the pipeline ourselves; see LoadPhase.
void RTXMGDemoApp::Render(nvrhi::IFramebuffer* framebuffer)
{
    // Unconditionally, as ApplicationBase::Render did, so the queue is always
    // drained and finalize overlaps decode: the GPU uploads textures while the
    // load thread is still decoding them.
    const bool texturesProcessed =
        m_TextureCache && m_TextureCache->ProcessRenderingThreadCommands(*m_CommonPasses, 20.f);

    if (m_loadPhase != LoadPhase::None && m_loadPhase != LoadPhase::Done)
    {
        AdvanceLoad(framebuffer, texturesProcessed);
        return;
    }

    RenderScene(framebuffer);
}

// One bounded step per frame, so the window stays responsive and the progress
// bar keeps animating through every phase — including the accel build.
void RTXMGDemoApp::AdvanceLoad(nvrhi::IFramebuffer* framebuffer, bool texturesProcessed)
{
    switch (m_loadPhase)
    {
    case LoadPhase::Import:
        if (m_loadResult.valid() &&
            m_loadResult.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            // The load thread has exited; everything from here is main-thread.
            if (!m_loadResult.get())
            {
                log::fatal("Failed to load scene from file: %s",
                           m_pendingSceneFile.string().c_str());
                m_pendingUploadCLs.clear();
                m_loadingScene.reset();
                m_loadPhase = LoadPhase::Done;
                break;
            }
            m_loadPhase = LoadPhase::Submit;
        }
        break;

    case LoadPhase::Submit:
        // The only part of the load that can't leave the main thread: queue
        // submission races the Streamline-wrapped present.
        for (auto& cl : m_pendingUploadCLs)
            GetDevice()->executeCommandList(cl);
        m_pendingUploadCLs.clear();
        m_loadPhase = LoadPhase::Textures;
        m_textureLoadStart = std::chrono::steady_clock::now();
        if (m_TextureCache && m_TextureCache->GetNumberOfRequestedTextures() > 0)
            log::info("Texture load: uploading %u textures...",
                      m_TextureCache->GetNumberOfRequestedTextures());
        break;

    case LoadPhase::Textures:
        // Decode finished on the load thread before it exited, so every texture
        // is already queued and donut's own test is exact: done when it drains.
        if (texturesProcessed)
            break;

        if (m_TextureCache)
        {
            const uint32_t requested = m_TextureCache->GetNumberOfRequestedTextures();
            m_TextureCache->LoadingFinished();  // release the upload command list
            if (requested > 0)
                log::info("Texture load: complete - %u uploaded in %.1f s.", requested,
                          std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - m_textureLoadStart).count());
        }
        m_loadPhase = LoadPhase::Reconcile;
        break;

    case LoadPhase::Reconcile:
        ReconcileLoadedScene();
        m_loadPhase = LoadPhase::Build;
        break;

    case LoadPhase::Build:
        BuildLoadedScene();
        m_loadPhase = LoadPhase::Done;
        break;

    default:
        break;
    }

    // Clear behind the UI pass, which draws the progress bar on top.
    nvrhi::ITexture* tex = framebuffer->getDesc().colorAttachments[0].texture;
    m_commandList->open();
    m_commandList->clearTextureFloat(tex, nvrhi::AllSubresources, nvrhi::Color(0.06f, 0.06f, 0.08f, 1.f));
    m_commandList->close();
    GetDevice()->executeCommandList(m_commandList);
}

// Reconcile the scene file's settings with the command line and the UI-edited
// state, then push the result to the renderer.  Must precede BuildLoadedScene:
// the streaming init reads the budgets and the vertex-normal flag.
void RTXMGDemoApp::ReconcileLoadedScene()
{
    RTXMGScene* scene = m_loadingScene.get();

    m_args.meshInputFile        = scene->GetInputPath();
    SaveBakerConfigJson(GetCurrentBakerConfig());
    const Json::Value& settings = scene->GetSceneSettings();
    m_args << settings;
    if (std::string& envarg = m_args.textures[TextureType::ENVMAP]; envarg.empty())
        if (const Json::Value& envmap = settings["envmap"]; envmap.isString())
        {
            // Resolve like 'models' does: scene-file-relative first, then the media root.
            const fs::path rel = envmap.asString();
            fs::path filepath = rtxmg::ResolveMediapath(
                m_pendingSceneFile.parent_path() / rel, m_mediaPath);
            if (filepath.empty())
                filepath = rtxmg::ResolveMediapath(rel, m_mediaPath);
            if (filepath.empty())
                log::warning("Envmap '%s' not found relative to the scene file or the media path '%s'.",
                             envmap.asCString(), m_mediaPath.generic_string().c_str());
            else
                envarg = filepath.lexically_normal().generic_string();
        }

    m_scene                   = std::move(m_loadingScene);
    m_accelBuilderNeedsUpdate = true;
    // Only a cluster_tess build refills these, and a scene without subd never runs one.
    m_BuildStats              = {};

    // The command line still outranks the scene file, but only over what a scene
    // file can set — the SceneArgs slice.  Re-parsing argv into m_args itself
    // also reset every UI-edited setting outside that slice.  Parse assigns only
    // what argv actually contains, so running it over a copy of the
    // scene-applied state reproduces the old precedence exactly.
    {
        Args sceneThenCli = m_args;
        sceneThenCli.Parse(m_argc, m_argv);
        m_args.sceneArgs() = sceneThenCli.sceneArgs();
    }

    // Restore the UI-edited state captured in HandleSceneLoad.
    m_args.lodGridCopies = m_savedGridCopies;
    m_args.lodGridGap    = m_savedGridGap;
    m_args.useBlasSharing            = m_savedBlasSharing;
    m_args.useBlasCaching            = m_savedBlasCaching;
    m_args.useBlasMerging            = m_savedBlasMerging;
    m_args.blasSharingEnabledLevels  = m_savedSharingLevels;
    m_args.blasCachingEnabledLevels  = m_savedCachingLevels;

    // Once, after the restore: UpdateParams loads the envmap (waitForIdle +
    // submit), so running it before the restore too would do that work twice.
    UpdateParams();

    GetRenderer().SetDisplayZBuffer(m_args.showOcclusionDepth);

    // Scene-attribute-dependent UI setup (animation range + audio); it can only
    // run now that the scene exists, not in UserInterface::CustomInit.
    if (m_gui)
        m_gui->OnSceneLoaded();
    GetRenderer().ResetPerSceneDiagnostics();
    m_tessSweepBaseMaxClusters = 0;
}

// GPU scene build: material/instance buffers + cluster-LOD streaming init /
// initial CLAS build, then camera framing.  The LoadPhase::Build step, so the
// streaming init consumes the finished metadata pre-upload.
void RTXMGDemoApp::BuildLoadedScene()
{
    auto& renderer = GetRenderer();

    const auto accelBuildStart = std::chrono::steady_clock::now();

    m_commandList->open();
    {
        nvrhi::utils::ScopedMarker marker(m_commandList, "Scene Load");
        if (auto* zbuffer = renderer.GetZBuffer())
        {
            zbuffer->Clear(m_commandList);
        }
        m_scene->FinishedLoading(m_commandList);
        RecordBakeStats();
        if (m_rebakeTriggered)
        {
            m_rebakeTriggered = false;
            m_bakeReport = m_pendingBakeReport;
            m_bakeReport.valid = true;
            GetUIData().showBakeReportWindow = true;
            GetUIData().showBakeConfigWindow = true;
            GetUIData().focusBakeReport      = true;
            GetUIData().refreshBakeConfig    = true;

            const auto& b = m_bakeReport.before;
            const auto& a = m_bakeReport.after;
            const double s = m_bakeReport.bakeSeconds;
            char timeBuf[32];
            if (s >= 60.0)
                snprintf(timeBuf, sizeof(timeBuf), "%dm %.0fs", int(s) / 60, std::fmod(s, 60.0));
            else
                snprintf(timeBuf, sizeof(timeBuf), "%.1fs", s);
            log::info("Bake report: %s, %u workers, %+lld MiB RAM",
                      timeBuf, m_bakeReport.workerThreads, (long long)m_bakeReport.memDeltaMiB);

            auto logRow = [&](const char* label, uint64_t before, uint64_t after) {
                if (!b.valid)
                    log::info("  %-16s %llu", label, (unsigned long long)after);
                else {
                    const double pct = before > 0
                        ? 100.0 * (double(after) - double(before)) / double(before) : 0.0;
                    log::info("  %-16s %llu \xe2\x86\x92 %llu (%+.1f%%)", label,
                              (unsigned long long)before, (unsigned long long)after, pct);
                }
            };
            logRow("Groups:",    b.groups,    a.groups);
            logRow("Clusters:",  b.clusters,  a.clusters);
            logRow("Triangles:", b.triangles, a.triangles);
            logRow("Disk (B):",  b.bakedBytes,  a.bakedBytes);
            logRow("GPU  (B):",  b.deviceBytes, a.deviceBytes);
            log::info("  Compressed:    %s \xe2\x86\x92 %s   Quantized UVs: %s \xe2\x86\x92 %s",
                      b.compressed ? "Yes":"No",  a.compressed ? "Yes":"No",
                      b.quantizedUv ? "Yes":"No", a.quantizedUv ? "Yes":"No");
            log::info("  Pos drop bits: %u \xe2\x86\x92 %u   Tex drop bits: %u \xe2\x86\x92 %u",
                      b.compressionPosDropBits, a.compressionPosDropBits,
                      b.compressionTexDropBits, a.compressionTexDropBits);
        }
        // Must run with the command list open so CreateAccelStructs can upload
        // the cluster-LOD GPU buffers.
        renderer.SceneFinishedLoading(m_scene, m_args.clodBudgets, m_commandList);
    }
    m_commandList->close();
    GetDevice()->executeCommandList(m_commandList);
    // Wall-clock time the main thread is blocked here.  Streaming init syncs
    // internally, so this captures most of the GPU cost without an extra sync.
    log::info("Scene build: buffers + cluster-LOD accel structs in %.1f s.",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - accelBuildStart).count());

    ResetCamera();
    if (m_hasSavedCamera)
    {
        m_camera.SetEye(m_savedCamEye);
        m_camera.SetLookat(m_savedCamLookat);
        m_camera.SetUp(m_savedCamUp);
        m_camera.SetFovY(m_savedCamFovY);
        m_hasSavedCamera = false;
    }

    // Sync CLAS position truncation to the baked precision.
    {
        const BakerConfig bakedCfg = GetCurrentBakerConfig();
        renderer.SetClasPositionTruncateBits(
            bakedCfg.useCompressedData ? bakedCfg.compressionPosDropBits : 0u);
    }

    m_lodCamera = m_camera;
    m_accelBuilderNeedsUpdate = true;
}

bool RTXMGDemoApp::IsLoadingTextures() const
{
    // Covers decode (on the load thread) and the upload drain alike: once the
    // requests are in, decoded/requested is the meaningful bar.
    return IsSceneLoading() && m_TextureCache &&
           m_TextureCache->GetNumberOfRequestedTextures() > 0;
}

RTXMGDemoApp::TextureLoadProgress RTXMGDemoApp::GetTextureLoadProgress() const
{
    TextureLoadProgress p;
    if (m_TextureCache)
    {
        p.requested = m_TextureCache->GetNumberOfRequestedTextures();
        p.decoded   = m_TextureCache->GetNumberOfLoadedTextures();
    }
    return p;
}

// Case-insensitive suffix test (`ext` must be given lowercase).  Avoids building
// a std::filesystem::path just to read an extension off a texture's path string.
static bool HasExtension(std::string_view path, std::string_view ext)
{
    if (path.size() < ext.size())
        return false;
    return std::equal(ext.begin(), ext.end(), path.end() - ext.size(),
                      [](char a, char b) { return a == char(std::tolower((unsigned char)b)); });
}

// Refresh the Memory tab's resident-texture figures: sum the GPU bytes of the
// finalized textures, deduping the LoadedTexture handles materials share.  Gated
// on the finalized counter so it costs one compare per frame once loading ends.
void RTXMGDemoApp::UpdateTextureMemStats()
{
    if (!m_scene || !m_TextureCache)
        return;

    const uint32_t finalized = m_TextureCache->GetNumberOfFinalizedTextures();
    if (finalized == m_textureMemFinalizedMark)
        return;
    m_textureMemFinalizedMark = finalized;

    std::unordered_set<const donut::engine::LoadedTexture*> counted;
    uint64_t bytes = 0, otherBytes = 0, vramBytes = 0;
    auto account = [&](const std::shared_ptr<donut::engine::LoadedTexture>& tex)
    {
        if (!tex || !tex->texture || !counted.insert(tex.get()).second)
            return;
        const uint64_t b = rtxmg::TextureGpuBytes(tex->texture->getDesc());
        bytes += b;
        vramBytes += rtxmg::TextureVramBytes(GetDevice(), tex->texture);
        // Split out the formats the budget pre-pass couldn't size from their
        // headers (.jpg/.png/...): their resident size is also their full size,
        // so the Memory tab folds them into its full-resolution denominator.
        // The budgetable set must match texture_budget.cpp's, or the ones that
        // differ get counted both here and in budgetableFullBytes.
        if (!HasExtension(tex->path, ".ktx2") && !HasExtension(tex->path, ".dds"))
            otherBytes += b;
    };
    for (const auto& mat : m_scene->GetMaterials())
    {
        if (!mat)
            continue;
        // Every slot, deduped: the cluster-LOD path aliases roughness onto the
        // metalness handle, while the subd/obj path can bind them separately.
        account(mat->baseOrDiffuseTexture);
        account(mat->metalnessTexture);
        account(mat->roughnessTexture);
        account(mat->specularF0Texture);
        account(mat->emissiveTexture);
        account(mat->normalOrDisplacementTexture);
    }

    stats::TextureMemStats& tm = stats::memUsageSamplers.textures;
    tm.loadedBytes      = bytes;
    tm.loadedOtherBytes = otherBytes;
    tm.vramBytes        = vramBytes;
    tm.loadedCount      = uint32_t(counted.size());
}

void RTXMGDemoApp::LogTessellationLimit()
{
    const auto& want = m_BuildStats.desired;
    const auto& have = m_BuildStats.allocated;
    const bool exceeded = want.m_numClusters > have.m_numClusters ||
                          want.m_clasSize > have.m_clasSize ||
                          want.m_vertexBufferSize > have.m_vertexBufferSize ||
                          want.m_vertexNormalsBufferSize > have.m_vertexNormalsBufferSize;

    const auto now = std::chrono::steady_clock::now();
    if (!exceeded)
    {
        if (m_tessLimitLogged)
            log::info("Tessellation back within its budget.");
        m_tessLimitLogged = false;
        return;
    }
    if (m_tessLimitLogged && now - m_tessLimitLogTime < std::chrono::seconds(10))
        return;

    constexpr double kMiB = 1024.0 * 1024.0;
    log::warning("Tessellation budget exceeded, clusters will be dropped: clusters %u / %u, "
                 "CLAS %.0f / %.0f MiB, vertices %.0f / %.0f MiB, normals %.0f / %.0f MiB (desired / allocated)",
                 want.m_numClusters, have.m_numClusters,
                 double(want.m_clasSize) / kMiB, double(have.m_clasSize) / kMiB,
                 double(want.m_vertexBufferSize) / kMiB, double(have.m_vertexBufferSize) / kMiB,
                 double(want.m_vertexNormalsBufferSize) / kMiB, double(have.m_vertexNormalsBufferSize) / kMiB);
    m_tessLimitLogged  = true;
    m_tessLimitLogTime = now;
}

float RTXMGDemoApp::GetTessBudgetLoad() const
{
    const auto& want = m_BuildStats.desired;
    const auto& have = m_BuildStats.allocated;
    auto ratio = [](double desired, double allocated) { return allocated > 0.0 ? float(desired / allocated) : 0.f; };
    return std::max({ ratio(want.m_numClusters, have.m_numClusters),
                      ratio(double(want.m_clasSize), double(have.m_clasSize)),
                      ratio(double(want.m_vertexBufferSize), double(have.m_vertexBufferSize)),
                      ratio(double(want.m_vertexNormalsBufferSize), double(have.m_vertexNormalsBufferSize)) });
}

void RTXMGDemoApp::LogAdaptiveTessRate()
{
    // Log each 10% move, plus the return to the user's rates.
    const float scale  = m_BuildStats.tessRateScale;
    const float logged = m_adaptiveTessLoggedScale;
    if (scale > logged * 1.1f || scale < logged / 1.1f || (scale == 1.f && logged != 1.f))
    {
        const auto& s = m_BuildStats;
        log::info("adaptive-tess-rate frame=%u load=%.3f scale=%.3f fineRate=%.4f "
                  "(clusters %.3f, CLAS %.3f, vertices %.3f)",
                  GetFrameIndex(), GetTessBudgetLoad(), scale, m_args.fineTessellationRate * scale,
                  s.clusterRateScale, s.clasRateScale, s.vertexRateScale);
        m_adaptiveTessLoggedScale = scale;
    }
}

void RTXMGDemoApp::UpdateVramBreakdown()
{
    const RTXMGRenderer& renderer = GetRenderer();

    stats::VramBreakdown vb;
    vb.textures      = stats::memUsageSamplers.textures.vramBytes;
    vb.envmap        = renderer.GetEnvMapBytes();
    vb.renderTargets = renderer.GetRenderTargetBytes();

    // Straight off m_BuildStats rather than the Memory tab's samplers: those only
    // advance while the profiler is recording, and this must read right when it
    // is paused.  `allocated` is what occupies VRAM; `desired` is only demand.
    const auto& tess = m_BuildStats.allocated;
    vb.tessVertices    = tess.m_vertexBufferSize + tess.m_vertexNormalsBufferSize;
    vb.tessClas        = tess.m_clasSize;
    vb.tessClusterData = tess.m_clusterDataSize;
    vb.tessSurfaces    = (m_scene ? m_scene->GetSubdVramBytes() : 0) + renderer.GetClusterTessWorkingBytes();
    // `desired` is this frame's demand against those buffers, and can exceed them
    // — that overflow is what the budget fields flash red for.
    const auto& tessWant = m_BuildStats.desired;
    vb.tessVerticesUsed    = tessWant.m_vertexBufferSize + tessWant.m_vertexNormalsBufferSize;
    vb.tessClasUsed        = tessWant.m_clasSize;
    vb.tessClusterDataUsed = tessWant.m_clusterDataSize;
    // Cluster-LoD dynamic BLAS is a separate pool from cluster_tess's, and both
    // can be live in a mixed scene.
    const uint64_t tlas = renderer.GetTlasBytes();
    vb.blas            = tess.m_blasSize + renderer.GetClusterLodBlasBufferBytes() + tlas;
    vb.blasUsed        = std::min(tessWant.m_blasSize, tess.m_blasSize)
                       + renderer.GetClusterLodBlasActualBytes() + tlas;
    // Every AS build takes its scratch from here, including cluster_tess's.
    vb.asScratch       = m_commandList ? m_commandList->getInternalMemoryUsage().scratchBytes : 0;

#if DONUT_WITH_STREAMLINE
    {
        // Both features: switching denoiser modes does not free the other one.
        donut::app::StreamlineInterface& streamline = donut::app::DeviceManager::GetStreamline();
        uint64_t dlssBytes = 0, dlssRrBytes = 0;
        streamline.QueryDLSSState(dlssBytes);
        streamline.QueryDLSSRRState(dlssRrBytes);
        vb.dlss = dlssBytes + dlssRrBytes;
    }
#endif

    // Traversal + BLAS-build tables exist on the preload path too, so this is
    // outside the streaming query below.
    if (renderer.GetClusterLodResources())
        vb.clodMetadata = renderer.GetClusterLodMetadataBytes();

    rtxmg::StreamingStats ss;
    if (renderer.GetClusterLodStreamingStats(ss))
    {
        // VRAM actually committed, not what is sub-allocated inside it: the
        // geometry and cached-BLAS pools hand out slices of whole blocks, and a
        // block stays resident until it empties.  The CLAS pool is one buffer at
        // the full budget, so there the two are the same number.
        vb.clodGeometry         = ss.allocatedDataBytes;
        vb.clodClas             = ss.reservedClasBytes;
        vb.clodCachedBlas       = ss.allocatedCachedBlasBytes;
        vb.clodCachedBlasBudget = ss.maxCachedBlasBytes;
        // What the Memory tab's Geometry / CLAS rows plot inside those pools.
        vb.clodGeometryUsed     = ss.usedDataBytes;
        vb.clodClasUsed         = ss.usedClasBytes;
        // The pinned low-detail geometry and CLAS sit outside both pools, so
        // they belong with the tables rather than inflating a budgeted bucket.
        vb.clodMetadata        += ss.operationsBytes
                                + ss.persistentDataBytes + ss.persistentClasBytes;
    }

    vb.driverValid = QueryDriverVram(vb.driverUsage, vb.driverBudget);
    stats::vramBreakdown = vb;
}

void RTXMGDemoApp::ReloadCurrentScene()
{
    if (m_args.meshInputFile.empty())
        return;

    // Resolve the same way Init() does, so HandleSceneLoad gets an absolute path.
    std::filesystem::path scenePath = app::GetDirectoryWithExecutable().parent_path() / "assets" / m_args.meshInputFile;
    if (!std::filesystem::exists(scenePath))
    {
        std::filesystem::path mediaScenePath = GetMediaPath() / m_args.meshInputFile;
        if (std::filesystem::exists(mediaScenePath))
            scenePath = mediaScenePath;
        else
            scenePath = m_args.meshInputFile;  // last-resort: assume already absolute
    }

    HandleSceneLoad(scenePath.lexically_normal().generic_string(),
                    GetMediaPath().generic_string());
}

void RTXMGDemoApp::ApplyStreamingBudgets()
{
    if (!m_scene)
        return;
    // A budget change means reallocating the pools, but the scene stays loaded.
    // The current pools may still be in flight, hence the idle on both sides.
    GetDevice()->waitForIdle();
    m_commandList->open();
    GetRenderer().ReinitAccelStructs(m_args.clodBudgets, m_commandList);
    m_commandList->close();
    GetDevice()->executeCommandList(m_commandList);
    GetDevice()->waitForIdle();
}

void RTXMGDemoApp::ApplyTextureSettingsAndReload(int textureBudgetMB, bool loadNormalMaps)
{
    m_args.textureBudgetMB  = std::max(0, textureBudgetMB);
    m_args.enableNormalMaps = loadNormalMaps;
    GetRenderer().SetEnableClusterLodNormalMaps(loadNormalMaps && m_args.normalMapShading);
    ReloadCurrentScene();
}

bool RTXMGDemoApp::QueryDriverVram(uint64_t& outUsage, uint64_t& outBudget) const
{
#if DONUT_WITH_DX12
    if (m_dxgiAdapter3)
    {
        DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
        if (SUCCEEDED(m_dxgiAdapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
        {
            outUsage  = info.CurrentUsage;
            outBudget = info.Budget;
            return true;
        }
    }
#endif
#if DONUT_WITH_VULKAN
    if (m_vkMemoryBudgetAvailable && GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
    {
        vk::PhysicalDevice phys =
            (VkPhysicalDevice)GetDevice()->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice).pointer;
        vk::PhysicalDeviceMemoryProperties2         props2;
        vk::PhysicalDeviceMemoryBudgetPropertiesEXT budgetProps;
        props2.pNext = &budgetProps;
        phys.getMemoryProperties2(&props2);

        uint64_t usage = 0, budget = 0;
        for (uint32_t i = 0; i < props2.memoryProperties.memoryHeapCount; ++i)
        {
            if (props2.memoryProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
            {
                usage  += budgetProps.heapUsage[i];
                budget += budgetProps.heapBudget[i];
            }
        }
        outUsage  = usage;
        outBudget = budget;
        return true;
    }
#endif
    return false;
}

void RTXMGDemoApp::ResetCamera()
{
    // Reachable from the 'F' key during the load splash, before the scene exists.
    if (!m_scene)
        return;

    m_cameraReset = true;
    // Restart the dolly oscillation from the reset position.
    m_dollyTime = 0.f;
    const bool useGridFraming = m_args.lodGridCopies > 1u && m_scene->HasClusterLod();

    if (!m_args.camString.empty())
    {
        if (!m_camera.Set(m_args.camString))
            log::warning("Ignoring malformed camera '%s'; expected [eye][lookat][up]fovY.",
                         m_args.camString.c_str());
    }
    else if (!useGridFraming && m_scene->GetView())
    {
        const View* view = m_scene->GetView();
        m_camera.SetEye(view->position);
        m_camera.SetLookat(view->lookat);
        m_camera.SetUp(view->up);
        m_camera.SetFovY(view->fov);
    }
    else if (useGridFraming)
    {
        // Position the camera near the corner (original instance) but aim it
        // at the grid's center of mass, so the grid extends across the frame
        // toward the far corner — ideal for distance-based LoD testing.
        const box3& origAabb = m_scene->GetOriginalClusterLodBbox();
        const box3  gridAabb = m_scene->GetSceneGraph().GetGlobalBoundingBox();
        const float3 origCenter  = origAabb.center();
        const float  origRadius  = 0.5f * length(origAabb.diagonal());
        const float3 gridCenter  = gridAabb.isempty() ? origCenter : gridAabb.center();
        const float3 dir         = float3(1.0f, 0.75f, 1.0f);
        m_camera.SetFovY(35.0f);
        m_camera.SetLookat(gridCenter);
        m_camera.SetEye(origCenter + dir * origRadius * 2.0f);
        m_camera.SetUp({ 0.f, 1.f, 0.f });
    }
    else
    {
        box3 aabb = m_scene->GetSceneGraph().GetGlobalBoundingBox();
        m_camera.Frame(aabb);
    }

    m_camera.SetAspectRatio(float(m_displaySize.x) / float(m_displaySize.y));

    m_trackBall.SetGimbalLock(true);
    SyncTrackballToCamera();
    // Default move speed (world units/second) from the MEDIAN world-space
    // instance size: robust against a handful of huge background vista
    // instances, which inflate the average.
    float moveSpeed = m_scene->GetAttributes().medianInstanceScale;
    if (m_args.cameraSpeed > 0.f)
        moveSpeed = m_args.cameraSpeed;
    moveSpeed = std::clamp(moveSpeed, Trackball::kMinMoveSpeed, Trackball::kMaxMoveSpeed);
    m_trackBall.SetMoveSpeed(moveSpeed);
    donut::log::info("Camera move speed: %.3f u/s (median instance size; avg %.3f)",
                     moveSpeed, m_scene->GetAttributes().averageInstanceScale);
}

void RTXMGDemoApp::ResetView()
{
    if (!m_scene)
        return;
    ResetCamera();
    GetRenderer().SetExposure(m_args.exposure);
    GetRenderer().SetAutoExposure(m_args.autoExposure);
}

void RTXMGDemoApp::SyncTrackballToCamera()
{
    m_trackBall.SetCamera(&m_camera);
    m_trackBall.SetReferenceFrame({ 1.f, 0.f, 0.f }, { 0.f, 0.f, 1.f },
        { 0.f, 1.f, 0.f });
}

bool RTXMGDemoApp::LoadScene(std::shared_ptr<vfs::IFileSystem> fs,
    const std::filesystem::path& sceneFileName)
{
    // The whole CPU half of the load, on the async load thread.  Buffer creation
    // and descriptor registration are free-threaded and the uploads are only
    // RECORDED here; queue submission stays on the main thread (LoadPhase::Submit)
    // because it races the Streamline-wrapped present.
    m_pendingSceneFs   = fs;
    m_pendingSceneFile = sceneFileName;
    m_pendingUploadCLs.clear();

    // Leave a couple of cores for the present + GPU-finalize thread — saturating
    // every core starves it and the progress bar stops animating.
    if (!m_textureLoadPool)
    {
        const unsigned hw = std::thread::hardware_concurrency();
        const unsigned workers = hw > 4 ? hw - 2 : std::max(1u, hw);
        m_textureLoadPool = std::make_unique<donut::engine::ThreadPool>(workers);
    }

    // Capture available RAM and effective worker count for the bake report.
    uint64_t ramBeforeMiB = 0;
    {
        rtxmg::SystemMemory ms;
        if (rtxmg::QuerySystemMemory(ms))
        {
            ramBeforeMiB = ms.availableBytes >> 20;
            uint32_t w = m_pendingBakeReport.workerThreads;
            if (w == 0)
            {
                const uint32_t cores = std::max(1u, std::thread::hardware_concurrency());
                const uint64_t budget =
                    (ms.totalBytes - (8ull << 30)) / (2ull << 30);
                w = m_args.bakeWorkers > 0 ? m_args.bakeWorkers
                  : uint32_t(std::clamp<uint64_t>(budget, 1, cores));
            }
            m_pendingBakeReport.workerThreads = w;
        }
    }
    const auto bakeT0 = std::chrono::steady_clock::now();

    stats::evaluatorSamplers = {};
    stats::memUsageSamplers  = {};

    auto scene = std::make_unique<RTXMGScene>(RTXMGSceneParams{
        .device          = GetDevice(),
        .mediaPath       = &GetMediaPath(),
        .commonPasses    = m_CommonPasses,
        .fs              = m_pendingSceneFs,
        .textureCache    = m_TextureCache,
        .descriptorTable = GetRenderer().GetDescriptorTable(),
        .initialFrameRange  = m_loadFrameRange,
        .isoLevelSharp      = m_args.isoLevelSharp,
        .isoLevelSmooth     = m_args.isoLevelSmooth,
        .clusterBakerConfig       = GetCurrentBakerConfig(),
        .clusterCacheDir          = m_args.clusterCacheDir,
        .clusterBakeWorkers       = m_args.bakeWorkers,
        .cancelLoad               = &m_loadCancelled,
    });
    scene->SetEnableMaterials(m_args.enableMaterials);
    scene->SetEnableNormalMaps(m_args.enableNormalMaps);

    // KTX2 texture mip budget: read texture headers, drop high-res mips to fit the
    // budget, and hand the scene its per-texture base mips BEFORE textures load.
    // Runs even with the budget disabled (--texture-budget-mb 0): it then drops
    // nothing and only measures the scene's texture footprint for the Memory tab.
    rtxmg::TextureBudgetStats texBudget;
    if (m_TextureCache)
    {
        std::unordered_map<std::string, uint32_t> textureBaseMips;
        rtxmg::ApplyKtxTextureBudget(textureBaseMips, m_pendingSceneFile,
                                     m_args.textureBudgetMB > 0 ? uint64_t(m_args.textureBudgetMB) << 20 : 0,
                                     m_args.enableNormalMaps, m_args.verboseLogging, &texBudget);
        scene->SetTextureBaseMips(std::move(textureBaseMips));
    }
    stats::memUsageSamplers.textures = {
        .textureCount    = texBudget.textureCount,
        .budgetableCount = texBudget.budgetableCount,
        .droppedCount    = texBudget.droppedCount,
        .diskBytes       = texBudget.diskBytes,
        .budgetableFullBytes = texBudget.budgetableFullBytes,
        .keptBytes       = texBudget.keptBytes,
        .budgetBytes     = texBudget.budgetBytes,
    };
    m_textureMemFinalizedMark = UINT32_MAX;  // force a resident-bytes recount

    if (m_loadCancelled.load(std::memory_order_relaxed))
        return false;

    if (!scene->LoadWithThreadPool(m_pendingSceneFile, m_textureLoadPool.get(), &m_pendingUploadCLs))
        return false;

    {
        auto& a = m_pendingBakeReport.after;
        a = {};
        a.valid = true;
        a.geometries = uint32_t(scene->GetClusterLodGeometries().size());
        for (const auto& geo : scene->GetClusterLodGeometries())
            for (const auto& l : geo.lodStats)
            {
                a.groups      += l.totGroups;
                a.clusters    += l.totClusters;
                a.triangles   += l.totTris;
                a.bakedBytes  += l.totBytes;
                a.deviceBytes += l.totDeviceBytes;
                a.compressed   = a.compressed  || bool(l.compressed);
                a.quantizedUv  = a.quantizedUv || bool(l.quantUv);
            }
        const BakerConfig afterCfg = GetCurrentBakerConfig();
        a.compressionPosDropBits = afterCfg.compressionPosDropBits;
        a.compressionTexDropBits = afterCfg.compressionTexDropBits;
    }

    // Pure CPU, and every input is known before the load starts (HandleSceneLoad
    // captures the UI-edited values, and the restore in ReconcileLoadedScene
    // outranks anything the scene file or the CLI re-parse could set).  Must run
    // before FinishedLoading, so the GPU buffers see the expanded instance count.
    scene->ApplyClusterLodGrid(m_savedGridCopies, m_savedGridGap, m_args.lodGridBits);

    // Drain decode before returning, so every texture is on the finalize queue by
    // the time the main thread starts pumping it — that is what makes the
    // "done when the queue drains" test exact.
    if (m_TextureCache && m_TextureCache->GetNumberOfRequestedTextures() > 0)
    {
        const auto decodeStart = std::chrono::steady_clock::now();
        m_textureLoadPool->WaitForTasks();
        log::info("Texture load: %u decoded in %.1f s.",
                  m_TextureCache->GetNumberOfLoadedTextures(),
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - decodeStart).count());
    }

    m_loadingScene = std::move(scene);

    m_pendingBakeReport.bakeSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - bakeT0).count();
    {
        rtxmg::SystemMemory ms;
        if (rtxmg::QuerySystemMemory(ms))
            m_pendingBakeReport.memDeltaMiB =
                (int64_t)ramBeforeMiB - (int64_t)(ms.availableBytes >> 20);
    }
    return true;
}

bool RTXMGDemoApp::KeyboardUpdate(int key, int scancode, int action,
    int mods)
{
    // Headless runs (`-nf` / `--nframes`) drop all input so smoke tests are
    // deterministic — a stray keystroke or trackball drag in the foreground
    // window can't perturb camera state, lpe, etc.
    if (m_args.exitAfterFrame >= 0)
        return true;

    // ESC toggles the UI; it also works during the async-load splash, which
    // draws its own progress window rather than the main UI.  Alt+F4 quits.
    if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE)
    {
        if (m_gui)
            m_gui->ToggleUIVisible();
        return true;
    }

    // The trackball's camera is wired in ResetCamera, from BuildLoadedScene, so
    // nothing below is safe to drive until the deferred scene build has run.
    if (!m_scene || IsSceneLoading())
        return true;

    auto& renderer = GetRenderer();
    m_trackBall.KeyboardUpdate(key, scancode, action, mods);

    if (action == GLFW_PRESS)
    {
        switch (key)
        {
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
            ResetView();
            break;
        case GLFW_KEY_R:
            if (mods & GLFW_MOD_CONTROL)
                ReloadShaders();
            break;
        case GLFW_KEY_P:
            if (mods & GLFW_MOD_SHIFT)
                SaveScreenshotWithUI();
            else
                SaveScreenshot();
            break;
        case GLFW_KEY_T:
            ToggleTimeView();
            break;
        case GLFW_KEY_SLASH:
            ToggleUpdateLodCamera();
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
    // The trackball's camera isn't wired until ResetCamera, so driving it during
    // the load splash would dereference a null camera.
    if (m_args.exitAfterFrame >= 0 || !m_scene || IsSceneLoading())
        return true;
    int2 pos = { static_cast<int>(xpos), static_cast<int>(ypos) };
    int2 canvas = { m_displaySize.x, m_displaySize.y };
    m_trackBall.MouseTrackingUpdate(pos, canvas);
    return true;
}

bool RTXMGDemoApp::MouseButtonUpdate(int button, int action, int mods)
{
    if (m_args.exitAfterFrame >= 0 || !m_scene || IsSceneLoading())
        return true;
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    {
        double mousex = 0, mousey = 0;
        glfwGetCursorPos(GetDeviceManager()->GetWindow(), &mousex, &mousey);

        float2 renderScale = float2(m_renderSize) / float2(m_displaySize);
        float2 mousePos = float2(float(mousex), float(mousey));
        float2 debugPixel = mousePos * renderScale;

        m_renderParams.debugPixel = int2(debugPixel);

        m_readPixelPick = true;
    }

    m_trackBall.MouseButtonUpdate(button, action, mods);
    return true;
}

bool RTXMGDemoApp::MouseScrollUpdate(double xoffset, double yoffset)
{
    if (m_args.exitAfterFrame >= 0 || !m_scene || IsSceneLoading())
        return true;
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

    // Cache window state here (and on WindowPosUpdate) since we can't query the
    // window during application quit.
    CaptureWindowState();
}

void RTXMGDemoApp::WindowPosUpdate(int /*xpos*/, int /*ypos*/)
{
    // A position change with no size change (e.g. dragging a fullscreen window
    // between equal-resolution monitors) doesn't fire BackBufferResized, so
    // refresh the cached state here too — otherwise the monitor change is lost.
    CaptureWindowState();
}

void RTXMGDemoApp::CaptureWindowState()
{
    GLFWwindow* window = GetDeviceManager()->GetWindow();
    m_windowState.isMaximized = glfwGetWindowAttrib(window, GLFW_MAXIMIZED) != 0;
    m_windowState.isFullscreen = glfwGetWindowMonitor(window) != nullptr;
    // The window's live top-left in virtual-screen coords, which is what
    // MonitorContainingPoint() matches on restore.  Reading it (not the
    // GLFW-tracked fullscreen monitor) also catches OS-initiated relocations
    // such as Win+Shift+Arrow, which GLFW's monitor pointer doesn't follow.
    glfwGetWindowPos(window, &m_windowState.windowPos.x, &m_windowState.windowPos.y);
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
    if (m_renderParams.denoiserMode != denoiserMode)
    {
        m_renderParams.denoiserMode = denoiserMode;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::Animate(float fElapsedTimeSeconds)
{
    // During async scene load m_scene is still being built on the load thread;
    // skip per-frame animation/state until it's ready (the splash is showing).
    if (!m_scene || IsSceneLoading())
        return;

    // Forward the authoritative wall-clock frame delta so auto-exposure's
    // eye-adaptation advances at a frame-rate-independent rate.
    GetRenderer().SetFrameDeltaTime(fElapsedTimeSeconds);

    korgi::Update();
    // fElapsedTimeSeconds, not GetCPUFrameTime(): the latter reads frame-start
    // timestamps that RenderScene updates later, so at Animate() time it lags a
    // frame and reads 0 right after load, making WASD movement stutter.
    m_trackBall.Animate(fElapsedTimeSeconds);

    // Counted from the first post-load frame, like the other two sweeps: the
    // scene load takes a variable number of frames, so keying the step off the
    // raw frame index left the ladder phase-shifted run to run.
    if (m_args.lpeSweep && m_postLoadBaseFrame >= 0)
    {
        static const float kLpeLadder[] = { 1.f, 2.f, 4.f, 8.f, 16.f, 32.f, 16.f, 8.f, 4.f, 2.f, 1.f };
        const uint32_t stepFrames = std::max(1u, m_args.lpeSweepFrameInterval);
        constexpr uint32_t kNumSteps = uint32_t(sizeof(kLpeLadder) / sizeof(kLpeLadder[0]));
        const uint32_t frameIdx = uint32_t(std::max(0, int(GetFrameIndex()) - m_postLoadBaseFrame));
        const uint32_t stepIdx  = (frameIdx / stepFrames) % kNumSteps;
        const float    lpeNow   = kLpeLadder[stepIdx];

        if (lpeNow != m_lastSweptLpe)
        {
            donut::log::info("[lpe-sweep] frame=%u step=%u interval=%u lpe=%.1f → %.1f",
                             frameIdx, stepIdx, stepFrames, m_lastSweptLpe, lpeNow);
            GetRenderer().SetLodPixelError(lpeNow);
            m_lastSweptLpe = lpeNow;
        }
    }

    // Drive the BLAS Sharing / BLAS Caching checkboxes from a fixed ladder, but
    // only once the scene has settled — a toggle edge is only interesting against
    // a converged resident set and a populated cached-BLAS pool.  Merging is left
    // alone; the renderer already gates it on sharing.
    if (m_args.blasToggleSweep && m_postLoadBaseFrame >= 0)
    {
        const int settle        = int(m_args.blasToggleSweepSettleFrames);
        const int postLoadFrame = int(GetFrameIndex()) - m_postLoadBaseFrame;
        if (postLoadFrame >= settle)
        {
            // (sharing, caching).  Steps 0<->1 are the UI's "uncheck BLAS Sharing"
            // edge (it force-clears caching too); 2<->3 the caching-only edge.
            static const bool kBothLadder[][2] = { { true, true }, { false, false }, { true, true }, { true, false } };
            const uint32_t kNumSteps  = (m_args.blasToggleMode == Args::BlasToggleMode::Both)
                                            ? uint32_t(sizeof(kBothLadder) / sizeof(kBothLadder[0])) : 2u;
            const uint32_t stepFrames = std::max(1u, m_args.blasToggleSweepFrameInterval);
            const uint32_t stepIdx    = (uint32_t(postLoadFrame - settle) / stepFrames) % kNumSteps;

            auto& renderer = GetRenderer();
            bool  sharing  = kBothLadder[stepIdx][0];
            bool  caching  = kBothLadder[stepIdx][1];
            if (m_args.blasToggleMode == Args::BlasToggleMode::Sharing)
            {
                sharing = (stepIdx == 0);
                caching = m_args.useBlasCaching;  // held at the CLI value
            }
            else if (m_args.blasToggleMode == Args::BlasToggleMode::Caching)
            {
                sharing = true;  // caching is a subset of sharing
                caching = (stepIdx == 0);
            }
            if (sharing != renderer.GetUseBlasSharing() || caching != renderer.GetUseBlasCaching())
            {
                donut::log::info("[blas-toggle-sweep] frame=%d step=%u sharing=%d->%d caching=%d->%d",
                                 postLoadFrame, stepIdx,
                                 int(renderer.GetUseBlasSharing()), int(sharing),
                                 int(renderer.GetUseBlasCaching()), int(caching));
                // A redirected stdout is block-buffered, so a device-removed a few
                // frames later would otherwise discard the edge that caused it.
                fflush(stdout);
                renderer.SetUseBlasSharing(sharing);
                renderer.SetUseBlasCaching(caching);
            }
        }
    }

    // Normal maps are only read at scene load, so toggling them is the one UI
    // control that goes through a full reload.  Alternate capture and toggle
    // steps, settling between each; the capture steps stall the machine until
    // RenderScene has actually written the file, so a reload never lands on the
    // frame being captured.
    if (m_args.normalMapSweep && m_nmSweepStep < kNormalMapSweepSteps
        && m_nmSweepCapture.empty() && m_postLoadBaseFrame >= 0
        && int(GetFrameIndex()) - m_postLoadBaseFrame >= int(m_args.normalMapSweepSettleFrames))
    {
        const stats::TextureMemStats& tm = stats::memUsageSamplers.textures;
        auto logState = [&](const char* what) {
            donut::log::info("[normalmap-sweep] step=%d %s  normalMaps=%d  textures=%u/%u resident  %llu MB",
                             m_nmSweepStep, what, int(m_args.enableNormalMaps),
                             tm.loadedCount, tm.textureCount,
                             (unsigned long long)(tm.loadedBytes >> 20));
            fflush(stdout);
        };

        switch (m_nmSweepStep)
        {
        case 0: logState("capture (initial)");   m_nmSweepCapture = "nmsweep_0_initial.png";  break;
        case 2: logState("capture (maps on)");   m_nmSweepCapture = "nmsweep_2_on.png";       break;
        case 4: logState("capture (maps off)");  m_nmSweepCapture = "nmsweep_4_off_again.png"; break;
        case 1:
            logState("enable normal maps -> reload");
            ApplyTextureSettingsAndReload(m_args.textureBudgetMB, true);
            break;
        case 3:
            logState("disable normal maps -> reload");
            ApplyTextureSettingsAndReload(m_args.textureBudgetMB, false);
            break;
        }
        ++m_nmSweepStep;
    }

    // Drive the cluster_tess Max Clusters budget from a two-step ladder, which
    // makes the tessellator reallocate its BLAS/CLAS storage mid-run.
    if (m_args.tessBudgetSweep && m_postLoadBaseFrame >= 0)
    {
        const int settle        = int(m_args.tessBudgetSweepSettleFrames);
        const int postLoadFrame = int(GetFrameIndex()) - m_postLoadBaseFrame;
        if (postLoadFrame >= settle)
        {
            // Latch the ladder's base once per scene: the sweep writes
            // tessMemorySettings back, so re-reading it would halve the base
            // every step.
            if (m_tessSweepBaseMaxClusters == 0)
                m_tessSweepBaseMaxClusters = m_args.tessMemorySettings.maxClusters;
            const uint32_t baseMaxClusters = m_tessSweepBaseMaxClusters;
            const uint32_t stepFrames = std::max(1u, m_args.tessBudgetSweepFrameInterval);
            const uint32_t stepIdx    = (uint32_t(postLoadFrame - settle) / stepFrames) % 2u;

            TessellatorConfig::MemorySettings settings = m_args.tessMemorySettings;
            settings.maxClusters = (stepIdx == 0) ? baseMaxClusters : std::max(1u, baseMaxClusters / 2u);
            if (settings.maxClusters != m_args.tessMemorySettings.maxClusters)
            {
                donut::log::info("[tess-budget-sweep] frame=%d step=%u maxClusters=%u->%u",
                                 postLoadFrame, stepIdx,
                                 m_args.tessMemorySettings.maxClusters, settings.maxClusters);
                fflush(stdout);
                SetTessMemSettings(settings);
            }
        }
    }

    // Fixed 1 s out, 1 s back at the camera fly speed.  The period is time-based
    // rather than a scene-diagonal extent so the turnaround stays watchable on
    // city-scale scenes (whose diagonal is minutes across).  Translation is applied
    // as a per-frame delta of the triangle wave's integral, so a cycle nets zero
    // even when a frame straddles the turnaround or the speed slider moves.
    if (m_args.dolly && m_scene)
    {
        constexpr float kDollyHalfPeriod = 1.f;  // seconds per leg
        const float3    forward = m_camera.GetDirection();
        // --dolly-frames pins the step to a fixed 1/N of a leg, which makes the
        // camera path a function of the frame index instead of the frame rate --
        // the difference between a stress test and a comparable one.
        const float     dt      = m_args.dollyFrames > 0
                                      ? kDollyHalfPeriod / float(m_args.dollyFrames)
                                      : std::max(0.f, fElapsedTimeSeconds);
        if (all(isfinite(forward)))
        {
            auto tri = [](float t)  // triangle wave in [0, kDollyHalfPeriod]
            {
                const float p = std::fmod(t, 2.f * kDollyHalfPeriod);
                return (p <= kDollyHalfPeriod) ? p : (2.f * kDollyHalfPeriod - p);
            };
            const float delta = m_trackBall.MoveSpeed() * (tri(m_dollyTime + dt) - tri(m_dollyTime));
            m_dollyTime += dt;
            if (delta != 0.f)
            {
                m_camera.Translate(forward * delta);
                GetRenderer().ResetSubframes();
            }
        }
    }

    const auto& animState = m_ui.timeLineEditorState;
    float animTime = animState.AnimationTime();
    float animRate = animState.frameRate;

    m_animationUpdated = m_animationTime != animTime;
    if (m_animationUpdated)
    {
        m_scene->Animate(animTime, animRate);

        GetRenderer().ResetSubframes();
        m_animationTime = animTime;
        m_accelBuilderNeedsUpdate = true;
    }
}

void RTXMGDemoApp::LimitFrameRate()
{
    if (m_args.maxFps == 0 || m_currFrameStart == std::chrono::steady_clock::time_point{})
        return;

    const auto minFrameTime = std::chrono::duration<double>(1.0 / double(m_args.maxFps));
    const auto nextFrameStart = m_currFrameStart +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(minFrameTime);

    if (const auto now = std::chrono::steady_clock::now(); now < nextFrameStart)
        std::this_thread::sleep_until(nextFrameStart);
}

void RTXMGDemoApp::LerpPoses(const SubdivisionSurface& mesh, uint32_t firstPose, uint32_t numPoses)
{
    constexpr uint32_t blockSize = 64;
    const uint32_t numVertices = mesh.NumVertices();

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, mesh.m_positionKeyframes))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_lerpPosesBuffer, nvrhi::Format::UNKNOWN,
            nvrhi::BufferRange(firstPose * sizeof(LerpKeyFramesPose), numPoses * sizeof(LerpKeyFramesPose))))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, mesh.m_posePositions))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, mesh.m_posePositionsPrev))
        .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(LerpKeyFramesConstants)));

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

    const LerpKeyFramesConstants constants{ .numVertices = numVertices, .poseStride = mesh.m_poseStride };
    m_commandList->setPushConstants(&constants, sizeof(constants));
    m_commandList->dispatch((numVertices + blockSize - 1) / blockSize, numPoses, 1);
}

void RTXMGDemoApp::DispatchGPUAnimation()
{
    const auto poses = m_scene->GetSubdPoses();
    if (poses.empty())
        return;

    nvrhi::utils::ScopedMarker marker(m_commandList, "GPU Animation");

    std::vector<LerpKeyFramesPose> lerpPoses(poses.size());
    for (size_t i = 0; i < poses.size(); ++i)
        lerpPoses[i] = { .f0 = poses[i].f0, .f1 = poses[i].f1, .blend = poses[i].blend, ._pad = 0 };

    const size_t bytes = lerpPoses.size() * sizeof(LerpKeyFramesPose);
    if (!m_lerpPosesBuffer || m_lerpPosesBuffer->getDesc().byteSize < bytes)
    {
        m_lerpPosesBuffer = GetDevice()->createBuffer(nvrhi::BufferDesc()
            .setByteSize(bytes)
            .setStructStride(sizeof(LerpKeyFramesPose))
            .setDebugName("LerpKeyFramesPoses")
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true));
    }
    m_commandList->writeBuffer(m_lerpPosesBuffer, lerpPoses.data(), bytes);

    // The scene keeps each mesh's poses contiguous: one dispatch per mesh.
    auto& subdMeshes = m_scene->GetSubdMeshes();
    for (uint32_t first = 0, last; first < poses.size(); first = last)
    {
        const uint32_t meshID = poses[first].meshID;
        for (last = first + 1; last < poses.size() && poses[last].meshID == meshID; ++last) {}
        LerpPoses(*subdMeshes[meshID], first, last - first);
    }
}

// Clear the framebuffer while the load thread runs; the UI pass draws the
// progress bar on top (see UserInterface::buildUI).
// Reached only once the load pipeline has run to Done (see Render/AdvanceLoad).
void RTXMGDemoApp::RenderScene(nvrhi::IFramebuffer* framebuffer)
{
    // First fully-loaded frame: rebase -nf frame counting here so a headless run
    // renders exitAfterFrame frames of the LOADED scene regardless of how many
    // frames loading took (see the exit / screenshot / pixel-debug checks).
    if (m_postLoadBaseFrame < 0)
    {
        m_postLoadBaseFrame = (int)GetFrameIndex();

        if (m_args.testRebake)
        {
            m_args.testRebake = false;
            BakerConfig cfg = GetCurrentBakerConfig();
            cfg.useCompressedData      = true;
            cfg.compressionPosDropBits = 8;
            cfg.compressionTexDropBits = 8;
            ApplyBakerConfigAndReload(cfg);
            return;
        }
    }

    // Before SetRenderCamera below, so a shot-list jump is seen as a camera cut.
    UpdateShotList();

    LimitFrameRate();

    auto& profiler = Profiler::Get();

    auto& renderer = GetRenderer();

    // -dp <x> <y>: queue a pixel-debug readback on the exit frame of -nf, the
    // way -s saves a screenshot there.  CLI input is in display-window coords
    // (matching the screenshot), but DLSS downscales dispatchRaysIndex to
    // m_renderSize, so scale as the right-click viewport handler does.
    if (m_args.debugPixel.x >= 0 && m_args.debugPixel.y >= 0
        && m_args.exitAfterFrame >= 0 && m_postLoadBaseFrame >= 0
        && (int)GetFrameIndex() - m_postLoadBaseFrame >= m_args.exitAfterFrame)
    {
        float2 renderScale = float2(m_renderSize) / float2(m_displaySize);
        m_renderParams.debugPixel = int2(float2(m_args.debugPixel) * renderScale);
        m_readPixelPick = true;
        m_dumpPixelDebug = true;
        m_args.debugPixel = int2(-1, -1);  // one-shot
    }
    if (m_reloadShaders)
    {
        GetDevice()->waitForIdle();

        m_accelBuilderNeedsUpdate = true;
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
        renderer.CreateOutputs(m_commandList);

        std::string frameMarker = "Frame Rendering " + std::to_string(GetFrameIndex());
        nvrhi::utils::ScopedMarker marker(m_commandList, frameMarker.c_str());

        DispatchGPUAnimation();

        stats::frameSamplers.gpuFrameTime.Start(m_commandList);

        const bool updateAccel = m_accelBuilderNeedsUpdate || m_ui.forceRebuildAccelStruct;
        {
            // m_lodCamera is written below, but only the pointer is
            // captured here — UpdateAccelerationStructures reads it after.
            const TessellatorConfig tessConfig =
            {
                .memorySettings = m_args.tessMemorySettings,
                .visMode = m_args.visMode,
                .tessMode = m_args.tessMode,
                .fineTessellationRate = m_args.fineTessellationRate,
                .coarseTessellationRate = m_args.coarseTessellationRate,
                .enableAdaptiveTessRate = m_args.adaptiveTessRate,
                .enableFrustumVisibility = m_args.enableFrustumVisibility,
                .enableHiZVisibility = m_args.enableHiZVisibility,
                .enableBackfaceVisibility = m_args.enableBackfaceVisibility,
                .enableLogging = m_args.enableAccelBuildLogging,
                .enableMonolithicClusterBuild = m_ui.enableMonolithicClusterBuild,
                .enableVertexNormals = m_args.enableVertexNormals,
                .enableClusterLodVertexNormals = m_args.enableClusterLodVertexNormals,
                .viewportSize = { (uint32_t)m_renderSize.x, (uint32_t)m_renderSize.y },
                .edgeSegments = m_args.edgeSegments,
                .isolationLevel = m_renderParams.isolationLevel,
                .clusterPattern = (ClusterTessPattern)m_renderParams.clusterPattern,
                .quantNBits = m_args.quantNBits,
                .displacementScale = m_renderParams.globalDisplacementScale,
                .camera = &m_lodCamera,
                .zbuffer = renderer.GetZBuffer(),
                .debugSurfaceIndex = m_debugSurfaceClusterLaneIndex[0],
                .debugClusterIndex = m_debugSurfaceClusterLaneIndex[1],
                .debugLaneIndex = m_debugSurfaceClusterLaneIndex[2],
            };

            // Retire before the prepass: it traces the previous frame's TLAS, so
            // anything freed here has to invalidate that TLAS first.
            renderer.RetireAccelResources(m_commandList, updateAccel ? &tessConfig : nullptr);

            // The prepass shares this gate on purpose: it is what keeps
            // cullViewProjMatrix matching the HiZ pyramid's viewpoint.
            if (m_args.updateLodCamera)
            {
                m_lodCamera = m_camera;
                renderer.RenderHiZPrepass(m_camera, m_commandList);
            }

            if (updateAccel)
            {
                renderer.UpdateAccelerationStructures(tessConfig, m_BuildStats, GetFrameIndex(), m_commandList);
                // Keep building while the GPU is still moving the adaptive rate scale.
                m_accelBuilderNeedsUpdate = m_BuildStats.tessRateAdjusted;
            }
        }

        renderer.Launch(m_commandList, GetFrameIndex(), m_sunLight);

        renderer.DlssUpscale(m_commandList, GetFrameIndex());
        {
            ScopedGPUTimer timer(stats::frameSamplers.blitTime, m_commandList);
            renderer.BlitFramebuffer(m_commandList, framebuffer);
        }

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

#if ENABLE_PIXEL_PICK
        if (m_readPixelPick)
        {
            GetRenderer().ReadPixelPick(m_commandList);
            m_readPixelPick = false;
        }
#endif

        if (m_dumpPixelDebug)
        {
            GetRenderer().DumpPixelDebugBuffers(m_commandList);
            m_dumpPixelDebug = false;
        }
    }
    m_commandList->close();
    GetDevice()->executeCommandList(m_commandList);

#if RTXMG_DEV_FEATURES
    if (!m_args.animDumpDir.empty() && m_postLoadBaseFrame >= 0)
        DumpAnimFrame(uint32_t(int(GetFrameIndex()) - m_postLoadBaseFrame));
#endif

    profiler.FrameEnd();

    // Only the profiler's Overview graph samples the frame/trace/DLSS timers, so
    // a --shot-list run with that window closed dumps them empty.  Must follow
    // executeCommandList: Stop() only records a timestamp write, so resolving
    // before submission reads whatever the timer's query ring held last -- which
    // lands different timers on different frames (trace appearing to exceed
    // frame).  Gated on the walk so an open profiler is not double-sampling.
    if (ShotListActive())
        stats::ProfileFrameTimers();

    // Outside the IsRecording() gate: not a time series, and the Memory tab
    // should read correctly with profiling paused.
    UpdateTextureMemStats();
    UpdateVramBreakdown();
    LogTessellationLimit();
    LogAdaptiveTessRate();

    if (profiler.IsRecording())
    {
        stats::clusterAccelSamplers.numClusters.PushBack(m_BuildStats.desired.m_numClusters);
        stats::clusterAccelSamplers.numClusters.max = m_BuildStats.allocated.m_numClusters;
        stats::clusterAccelSamplers.numTriangles.PushBack(m_BuildStats.desired.m_numTriangles);

        stats::clusterAccelSamplers.renderSize = m_renderSize;

        // Cluster-LOD per-frame TLAS geometry (TRACK_RENDER_STATS); unique =
        // CLAS-deduped footprint, total = instanced.  uint64 totals are clamped
        // for the uint32 graph samplers (the table/HUD show the full values).
        if (GetRenderer().GetClusterLodResources() && GetRenderer().GetAtomicInt64OnHeapSupported())
        {
            const auto& clc = GetRenderer().GetClusterLodCounters();
            auto clampU32 = [](uint64_t v) { return uint32_t(std::min<uint64_t>(v, 0xFFFFFFFFull)); };
            stats::clusterAccelSamplers.clusterLodUniqueTriangles.PushBack(clampU32(clc.uniqueTriangles));
            stats::clusterAccelSamplers.clusterLodTotalTriangles.PushBack(clampU32(clc.totalTriangles));
            stats::clusterAccelSamplers.clusterLodUniqueClusters.PushBack(clc.uniqueClusters);
            stats::clusterAccelSamplers.clusterLodTotalClusters.PushBack(clc.totalClusters);
        }

        stats::memUsageSamplers.blasSize.PushBack(m_BuildStats.desired.m_blasSize);
        stats::memUsageSamplers.clasSize.PushBack(m_BuildStats.desired.m_clasSize);
        stats::memUsageSamplers.blasScratchSize.PushBack(m_BuildStats.desired.m_blasScratchSize);
        stats::memUsageSamplers.vertexBufferSize.PushBack(m_BuildStats.desired.m_vertexBufferSize);
        stats::memUsageSamplers.vertexNormalsBufferSize.PushBack(m_BuildStats.desired.m_vertexNormalsBufferSize);
        stats::memUsageSamplers.clusterShadingDataSize.PushBack(m_BuildStats.desired.m_clusterDataSize);

        stats::memUsageSamplers.blasSize.max = m_BuildStats.allocated.m_blasSize;
        stats::memUsageSamplers.clasSize.max = m_BuildStats.allocated.m_clasSize;
        stats::memUsageSamplers.blasScratchSize.max = m_BuildStats.allocated.m_blasScratchSize;
        stats::memUsageSamplers.vertexBufferSize.max = m_BuildStats.allocated.m_vertexBufferSize;
        stats::memUsageSamplers.vertexNormalsBufferSize.max = m_BuildStats.allocated.m_vertexNormalsBufferSize;
        stats::memUsageSamplers.clusterShadingDataSize.max = m_BuildStats.allocated.m_clusterDataSize;

        // Outside the streaming block below: the phase series must also be summed
        // on the --preload path, which reports no streaming stats.
        const float clasBuildMs = stats::clusterAccelSamplers.ProfileClusterLodPhases();

        stats::StreamingSamplers& ssr = stats::streamingSamplers;

        // Traversal + BLAS-effectiveness counters. Produced by the same kernels
        // on both paths and read back unconditionally, so they are gated on
        // cluster LoD being present rather than on streaming.
        if (GetRenderer().GetClusterLodResources())
        {
            ssr.latestCounters        = GetRenderer().GetClusterLodCounters();
            ssr.latestBlasActualBytes = GetRenderer().GetClusterLodBlasActualBytes();
            ssr.blasBuilds.PushBack(ssr.latestCounters.blasBuildCounter);
        }

        // Cluster-LOD streaming stats (no-op in --preload mode).
        rtxmg::StreamingStats ss;
        if (GetRenderer().GetClusterLodStreamingStats(ss))
        {
            constexpr float kMB = 1.0f / (1024.0f * 1024.0f);
            ssr.latest = ss;
            ssr.geometryMB.PushBack(static_cast<float>(ss.usedDataBytes) * kMB);
            ssr.clasMB.PushBack(static_cast<float>(ss.usedClasBytes) * kMB);
            ssr.residentGroups.PushBack(ss.residentGroups);
            ssr.residentClusters.PushBack(ss.residentClusters);

            // Rates come from the delta of the cumulative totals: the latched
            // transferBytes/loadCount snapshot doesn't fall back to 0 when idle.
            // A scene reload resets the totals, so cur < prev means "use cur".
            auto deltaSince = [](uint64_t cur, uint64_t& prev) -> uint64_t {
                uint64_t d = (cur >= prev) ? (cur - prev) : cur;
                prev = cur;
                return d;
            };
            // A scene reload recreates the streaming object and resets the
            // cumulative totals to 0; clear the streaming-impact peaks then so
            // they measure the current scene only.
            if (ss.totalTransferBytes < ssr.prevTotalTransferBytes)
            {
                ssr.maxStreamClasBuildMs   = 0.f;
                ssr.sumStreamClasBuildMs   = 0.0;
                ssr.streamFrameCount       = 0;
                ssr.maxStreamTransferBytes = 0;
            }
            const uint64_t dBytes   = deltaSince(ss.totalTransferBytes, ssr.prevTotalTransferBytes);
            const uint64_t dLoads   = deltaSince(ss.totalLoads,         ssr.prevTotalLoads);
            const uint64_t dUnloads = deltaSince(ss.totalUnloads,       ssr.prevTotalUnloads);
            const float    frameMs  = GetCPUFrameTime();
            const double   perSec   = frameMs > 0.0f ? 1000.0 / static_cast<double>(frameMs) : 0.0;
            ssr.transferRate.PushBack(static_cast<float>(static_cast<double>(dBytes)   * perSec));
            ssr.loadsPerSec.PushBack(static_cast<float>(static_cast<double>(dLoads)    * perSec));
            ssr.unloadsPerSec.PushBack(static_cast<float>(static_cast<double>(dUnloads) * perSec));

            // The MAX is tracked unconditionally because the async readback lands
            // a few frames after the build; idle frames cost ~0 so the peak still
            // reads true.  The AVERAGE only counts frames with streaming activity,
            // so a converged scene stops diluting it.
            ssr.maxStreamClasBuildMs = std::max(ssr.maxStreamClasBuildMs, clasBuildMs);
            const bool streamedThisFrame = (dBytes > 0 || dLoads > 0 || dUnloads > 0);
            if (streamedThisFrame)
            {
                ssr.sumStreamClasBuildMs += clasBuildMs;
                ssr.streamFrameCount++;
                ssr.maxStreamTransferBytes = std::max(ssr.maxStreamTransferBytes, dBytes);
            }
        }

        // Outside the streaming block: --preload reports no streaming stats, but
        // its ladder is what scenario 2.2 compares against the streaming one.
        // The counter readback lags its frame by a fixed number of frames, so the
        // series is still exactly reproducible.
        if (stats::frameLog.enabled && m_postLoadBaseFrame >= 0
            && stats::clusterAccelSamplers.hasClusterLod)
        {
            const shaderio::SceneBuildingCounters clc = GetRenderer().GetClusterLodCounters();
            stats::frameLog.Push({
                .frame            = uint32_t(int(GetFrameIndex()) - m_postLoadBaseFrame),
                .renderedClusters = clc.numRenderedClusters,
                .desiredClusters  = clc.desiredRenderClusters,
                .uniqueClusters   = clc.uniqueClusters,
                .totalClusters    = clc.totalClusters,
                .residentGroups   = ss.residentGroups,
                .residentClusters = ss.residentClusters,
                .blasBuilds       = clc.blasBuildCounter,
            });
        }
    }

    if (m_screenshot)
    {
        nvrhi::ITexture* framebufferTexture = framebuffer->getDesc().colorAttachments[0].texture;
        DoSaveScreenshot(framebufferTexture, "");
        m_screenshot = false;
    }

    if (m_shotPendingCapture)
        CaptureShot(framebuffer);

    if (!m_nmSweepCapture.empty())
    {
        nvrhi::ITexture* tex = framebuffer->getDesc().colorAttachments[0].texture;
        DoSaveScreenshot(tex, m_nmSweepCapture);
        m_nmSweepCapture.clear();
        if (m_nmSweepStep >= kNormalMapSweepSteps)
        {
            log::info("[normalmap-sweep] done.");
            glfwSetWindowShouldClose(GetDeviceManager()->GetWindow(), true);
        }
    }

    // Exit after N frames — counted from the frame the scene finished loading
    // (m_postLoadBaseFrame), so the load's variable frame count doesn't affect
    // the number of rendered scene frames.
    if (m_args.exitAfterFrame >= 0 && m_postLoadBaseFrame >= 0
        && (int)GetFrameIndex() - m_postLoadBaseFrame >= m_args.exitAfterFrame)
        glfwSetWindowShouldClose(GetDeviceManager()->GetWindow(), true);

    // Save screenshot (no UI) on exit
    if (!m_args.outfile.empty() && glfwWindowShouldClose(GetDeviceManager()->GetWindow()))
    {
        nvrhi::ITexture* tex = framebuffer->getDesc().colorAttachments[0].texture;
        DoSaveScreenshot(tex, m_args.outfile);
        m_args.outfile.clear();
    }

    if (!m_args.dumpStatsFile.empty() && glfwWindowShouldClose(GetDeviceManager()->GetWindow()))
    {
        DumpStats(m_args.dumpStatsFile);
        m_args.dumpStatsFile.clear();
    }

    m_cameraReset = false;
}

// Residency is called settled once the counters hold still this many frames.
// Streaming lands loads in batches, so a single flat frame is common mid-batch.
static constexpr uint32_t kShotFlatFrames = 8;

void RTXMGDemoApp::RunShotList(std::vector<ShotPosition> positions, std::vector<ShotConfig> configs)
{
    m_shotPositions    = std::move(positions);
    m_shotConfigs      = std::move(configs);
    m_shotConfigIndex   = 0;
    m_shotPositionIndex = 0;
    m_shotIndex          = 0;
    m_shotPhase           = ShotPhase::Jump;
    m_shotPhaseFrames     = 0;
    m_shotInteractiveRun = true;

    const RTXMGRenderer& renderer = GetRenderer();
    ShotRestoreState     state{
        .camera           = m_camera.ToString(),
        .shadingMode      = renderer.GetShadingMode(),
        .colorMode        = renderer.GetColorMode(),
        .wireframe        = renderer.GetWireframe(),
        .exposure         = renderer.GetExposure(),
        .autoExposure     = renderer.GetAutoExposure(),
        .dlssMode         = m_ui.dlssMode,
        .lodPixelError    = renderer.GetLodPixelError(),
        .adaptiveLodError = renderer.GetAdaptiveLodError(),
        .normalMapShading = GetNormalMapShading(),
        .monitor          = nullptr,
    };
    if (GLFWwindow* window = GetDeviceManager()->GetWindow())
    {
        state.monitor = glfwGetWindowMonitor(window);
        glfwGetWindowPos(window, &state.windowX, &state.windowY);
        glfwGetWindowSize(window, &state.windowW, &state.windowH);
        state.decorated = glfwGetWindowAttrib(window, GLFW_DECORATED) == GLFW_TRUE;
    }
    m_shotRestore = state;
}

void RTXMGDemoApp::AbortShotList()
{
    m_shotPositions.clear();
    m_shotConfigs.clear();
    m_shotPhase           = ShotPhase::Done;
    m_shotInteractiveRun = false;
    RestoreAfterShotList();
}

void RTXMGDemoApp::RestoreAfterShotList()
{
    if (!m_shotRestore)
        return;
    const ShotRestoreState state = *m_shotRestore;
    m_shotRestore.reset();

    SetCameraString(state.camera);

    RTXMGRenderer& renderer = GetRenderer();
    renderer.SetShadingMode(state.shadingMode);
    renderer.SetColorMode(state.colorMode);
    renderer.SetWireframe(state.wireframe);
    renderer.SetExposure(state.exposure);
    renderer.SetAutoExposure(state.autoExposure);
    renderer.SetLodPixelError(state.lodPixelError);
    renderer.SetAdaptiveLodError(state.adaptiveLodError);
    renderer.ResetAdaptiveLodError();
    SetNormalMapShading(state.normalMapShading);
    m_ui.dlssMode = state.dlssMode;

    GLFWwindow* window = GetDeviceManager()->GetWindow();
    if (!window)
        return;
    if (state.monitor)
    {
        const GLFWvidmode* mode = glfwGetVideoMode(state.monitor);
        glfwSetWindowMonitor(window, state.monitor, 0, 0, mode->width, mode->height,
                             mode->refreshRate);
        return;
    }
    glfwSetWindowAttrib(window, GLFW_DECORATED, state.decorated ? GLFW_TRUE : GLFW_FALSE);
    glfwSetWindowPos(window, state.windowX, state.windowY);
    glfwSetWindowSize(window, state.windowW, state.windowH);
}

void RTXMGDemoApp::UpdateShotList()
{
    if (!ShotListActive() || m_shotPhase == ShotPhase::Done)
        return;

    const ShotConfig&   config   = m_shotConfigs[m_shotConfigIndex];
    const ShotPosition& pos      = m_shotPositions[m_shotPositionIndex];
    RTXMGRenderer&       renderer = GetRenderer();

    switch (m_shotPhase)
    {
    case ShotPhase::Jump:
        JumpCameraToShot(pos);  // read by SetRenderCamera below as a camera cut
        // Reallocating settings apply once per config, at its first position:
        // before the settle, so residency converges under the settings the
        // shots are captured with, but not repeated (and not re-triggering a
        // resolution/DLSS re-init) at every position under the same config.
        if (m_shotPositionIndex == 0)
        {
            if (config.outputWidth > 0 && config.outputHeight > 0)
                SetShotOutputResolution(config);
            if (config.dlssModeSet)
                m_ui.dlssMode = config.dlssMode;
            if (config.lodPixelError > 0.f)
                renderer.SetLodPixelError(config.lodPixelError);
            if (config.adaptiveLodError >= 0)
                renderer.SetAdaptiveLodError(config.adaptiveLodError != 0);
            if (config.normalMapShading >= 0)
                SetNormalMapShading(config.normalMapShading != 0);
        }
        // Each capture point must converge on its own: the effective error is
        // sticky, so without this a position inherits the previous one's and
        // the walk measures position order as much as it measures the settings.
        renderer.ResetAdaptiveLodError();
        // Pinned, never inherited: auto-exposure is a temporal feedback loop and
        // would drift the golden on its own.  LoadShotList requires the value.
        renderer.SetAutoExposure(false);
        renderer.SetExposure(pos.exposure > 0.f ? pos.exposure : config.exposure);
        renderer.ResetSubframes();
        renderer.ResetDenoiser();

        m_shotIndex                = 0;
        m_shotRecordsAtEntryStart  = stats::shotRecords.size();
        m_shotPhaseFrames          = 0;
        m_shotFlatFrames           = 0;
        m_shotLastResidentGroups   = ~0u;
        m_shotLastResidentClusters = ~0u;
        m_shotLastEffectiveError   = -1.f;
        m_shotPhase                = ShotPhase::Settle;
        break;

    case ShotPhase::Settle:
    {
        ++m_shotPhaseFrames;

        bool settled = false;
        if (config.settleFrames >= 0)
        {
            settled = m_shotPhaseFrames >= uint32_t(config.settleFrames);
        }
        else if (rtxmg::StreamingStats ss; renderer.GetClusterLodStreamingStats(ss))
        {
            // The adaptive controller walks the effective error by a few tenths
            // of a percent per frame, so it is still drifting long after the
            // resident set stops moving -- and it, not the -lpe value, is what
            // the frame was traversed at.  Constant when adaptive error is off.
            const float effectiveError = renderer.GetEffectiveLodPixelError();
            const bool  flat = ss.residentGroups   == m_shotLastResidentGroups
                            && ss.residentClusters == m_shotLastResidentClusters
                            && std::abs(effectiveError - m_shotLastEffectiveError) < 1e-4f;
            m_shotFlatFrames           = flat ? m_shotFlatFrames + 1 : 0;
            m_shotLastResidentGroups   = ss.residentGroups;
            m_shotLastResidentClusters = ss.residentClusters;
            m_shotLastEffectiveError   = effectiveError;
            settled                    = m_shotFlatFrames >= kShotFlatFrames;
        }
        else
        {
            // --preload and subdivision-only scenes have no residency to converge.
            settled = true;
        }

        if (settled || m_shotPhaseFrames >= config.settleCap)
        {
            if (!settled)
                log::warning("--shot-list '%s'/'%s': residency still moving after %u frames "
                             "(settleCap); capturing anyway.",
                             config.name.c_str(), pos.label.c_str(), m_shotPhaseFrames);
            m_shotSettleFrames = m_shotPhaseFrames;
            // Time the settled configuration, not the streaming churn that got
            // here: the settle can outlast the sampler window on its own.
            stats::ResetTimingSamplers();
            if (config.shots.empty())
            {
                DumpPositionStats(pos, config);
                AdvancePosition();
            }
            else
            {
                BeginShot();
            }
        }
        break;
    }

    case ShotPhase::Accumulate:
        ++m_shotPhaseFrames;
        if (m_shotPhaseFrames >= ShotAccumFrames(config.shots[m_shotIndex]))
            m_shotPendingCapture = true;
        break;

    case ShotPhase::Done:
        break;
    }
}

void RTXMGDemoApp::BeginShot()
{
    const ShotConfig&   config = m_shotConfigs[m_shotConfigIndex];
    const ShotPosition& pos    = m_shotPositions[m_shotPositionIndex];
    const Shot&         shot   = config.shots[m_shotIndex];

    // Both setters reset accumulation and the denoiser, so the shot accumulates
    // from scratch rather than inheriting the previous mode's history.
    GetRenderer().SetShadingMode(shot.shadingMode);
    GetRenderer().SetColorMode(shot.colorMode);
    GetRenderer().SetWireframe(shot.wireframe);
    GetRenderer().SetExposure(ShotExposure(shot, pos, config));
    // ResetSubframes is a no-op with the denoiser on, which would leave the
    // jitter phase dependent on the scene-load frame count.
    GetRenderer().ForceResetSubframes();

    m_shotPhaseFrames = 0;
    m_shotPhase       = ShotPhase::Accumulate;
}

void RTXMGDemoApp::AdvanceShot()
{
    const ShotConfig& config = m_shotConfigs[m_shotConfigIndex];
    const ShotPosition& pos  = m_shotPositions[m_shotPositionIndex];

    if (++m_shotIndex < config.shots.size())
    {
        BeginShot();
        return;
    }

    DumpPositionStats(pos, config);
    AdvancePosition();
}

void RTXMGDemoApp::AdvancePosition()
{
    if (++m_shotPositionIndex < m_shotPositions.size())
    {
        m_shotPhase = ShotPhase::Jump;
        return;
    }

    if (++m_shotConfigIndex < m_shotConfigs.size())
    {
        m_shotPositionIndex = 0;
        m_shotPhase          = ShotPhase::Jump;
        return;
    }

    m_shotPhase = ShotPhase::Done;
    log::info("--shot-list: all capture points written.");
    if (m_shotInteractiveRun)
    {
        // Hand control back to the interactive session instead of exiting.
        m_shotPositions.clear();
        m_shotConfigs.clear();
        m_shotInteractiveRun = false;
        RestoreAfterShotList();
    }
    else
    {
        glfwSetWindowShouldClose(GetDeviceManager()->GetWindow(), true);
    }
}

// Resize the window to the config's output resolution.  DeviceManager reads the
// GLFW size every frame and resizes the swap chain itself, so setting it here is
// enough; the settle that follows absorbs the DLSS re-init and the re-converge.
void RTXMGDemoApp::SetShotOutputResolution(const ShotConfig& config)
{
    GLFWwindow* window = GetDeviceManager()->GetWindow();
    if (!window)
        return;

    // Drop out of fullscreen: its size comes from the display, and a list that
    // spans resolutions has to drive the size itself.
    if (glfwGetWindowMonitor(window))
        glfwSetWindowMonitor(window, nullptr, 0, 0, config.outputWidth, config.outputHeight,
                             GLFW_DONT_CARE);

    // Undecorated at the origin, so the client area is the full requested size:
    // Windows trims the *frame* to the work area, and a decorated 3840x2160
    // request came back as a 3840x2119 client.  Only the framebuffer is
    // captured, so it does not matter that the window runs off-screen.
    glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_FALSE);
    glfwSetWindowPos(window, 0, 0);
    glfwSetWindowSize(window, config.outputWidth, config.outputHeight);
}

// Video memory this process has resident on the local adapter, which is the
// only figure that includes what the pool counters cannot see: render targets,
// the G-buffer, DLSS internals, TLAS and build scratch, driver overhead.
// D3D12 only -- the Vulkan equivalent needs VK_EXT_memory_budget, and a wrong
// number is worse than none, so VK reports nothing.
bool RTXMGDemoApp::QueryVideoMemory(uint64_t& usage, uint64_t& budget) const
{
#if DONUT_WITH_DX12
    if (GetDevice()->getGraphicsAPI() != nvrhi::GraphicsAPI::D3D12)
        return false;

    auto* device = static_cast<ID3D12Device*>(
        GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device).pointer);
    if (!device)
        return false;

    nvrhi::RefCountPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
        return false;

    // By LUID rather than index: the device may not be on adapter 0.
    nvrhi::RefCountPtr<IDXGIAdapter3> adapter;
    if (FAILED(factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter))))
        return false;

    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    if (FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
        return false;

    usage  = info.CurrentUsage;
    budget = info.Budget;
    return true;
#else
    (void)usage; (void)budget;
    return false;
#endif
}

// The memory, streaming and timer blocks are sampled live, so they describe a
// capture point only if written at it -- the run-end dump would report every
// location as whatever the last one happened to leave behind.  The 60-sample
// timer window is entirely this location's by now (a pt shot accumulates 128
// frames here), though min/max still span the run.
void RTXMGDemoApp::DumpPositionStats(const ShotPosition& pos, const ShotConfig& config)
{
    if (m_args.dumpStatsFile.empty())
        return;

    // Carry only this position's shots, and leave the cumulative frame log to
    // the run-end dump rather than repeating it in every location's file.
    std::vector<stats::ShotRecord>  allShots  = std::move(stats::shotRecords);
    std::vector<stats::FrameRecord> allFrames = std::move(stats::frameLog.records);
    stats::shotRecords.assign(allShots.begin() + m_shotRecordsAtEntryStart, allShots.end());

    const std::string configName = config.name.empty() ? "default" : config.name;
    const std::filesystem::path statsDir = ShotStatsDir() / configName;
    std::filesystem::create_directories(statsDir);
    DumpStats((statsDir / (pos.label + ".stats.json")).generic_string());

    stats::shotRecords     = std::move(allShots);
    stats::frameLog.records = std::move(allFrames);
}

void RTXMGDemoApp::CaptureShot(nvrhi::IFramebuffer* framebuffer)
{
    const ShotConfig&   config = m_shotConfigs[m_shotConfigIndex];
    const ShotPosition& pos    = m_shotPositions[m_shotPositionIndex];
    const Shot&          shot   = config.shots[m_shotIndex];

    const std::string configName = config.name.empty() ? "default" : config.name;
    const std::filesystem::path imageDir = ShotImageDir() / configName;
    std::filesystem::create_directories(imageDir);
    const std::filesystem::path out = imageDir / (pos.label + "." + shot.Tag() + ".png");

    nvrhi::ITexture* tex = framebuffer->getDesc().colorAttachments[0].texture;
    DoSaveScreenshot(tex, out.generic_string());

    stats::ShotRecord rec;
    rec.label         = pos.label;
    rec.mode          = shot.Tag();
    rec.file          = out.filename().generic_string();
    rec.settleFrames  = m_shotSettleFrames;
    rec.accumFrames   = ShotAccumFrames(shot);
    rec.subframeIndex = GetRenderer().GetSubframeIndex();
    rec.exposure      = ShotExposure(shot, pos, config);
    auto percentOf = [](uint64_t used, uint64_t budget)
    { return budget ? float(double(used) / double(budget) * 100.0) : 0.f; };

    if (rtxmg::StreamingStats ss; GetRenderer().GetClusterLodStreamingStats(ss))
    {
        rec.residentGroups   = ss.residentGroups;
        rec.residentClusters = ss.residentClusters;
        // Same used/budget pairs the Memory tab bars them against.
        rec.geometryBytes       = ss.usedDataBytes;
        rec.geometryBudgetBytes = ss.maxDataBytes;
        rec.geometryPercent     = percentOf(ss.usedDataBytes, ss.maxDataBytes);
        rec.clasBytes           = ss.usedClasBytes;
        rec.clasBudgetBytes     = ss.reservedClasBytes;
        rec.clasPercent         = percentOf(ss.usedClasBytes, ss.reservedClasBytes);
    }
    const shaderio::SceneBuildingCounters clc = GetRenderer().GetClusterLodCounters();
    rec.uniqueClusters   = clc.uniqueClusters;
    rec.totalTriangles   = clc.totalTriangles;
    rec.renderedClusters = clc.numRenderedClusters;

#if DONUT_WITH_STREAMLINE
    rec.dlssMode              = DlssModeName(m_ui.dlssMode);
#else
    rec.dlssMode              = "off";
#endif
    rec.renderWidth           = stats::clusterAccelSamplers.renderSize.x;
    rec.renderHeight          = stats::clusterAccelSamplers.renderSize.y;
    rec.outputWidth           = m_displaySize.x;
    rec.outputHeight          = m_displaySize.y;
    rec.lodPixelError         = GetRenderer().GetLodPixelError();
    rec.adaptiveLodPixelError = GetRenderer().GetEffectiveLodPixelError();
    rec.normalMapShading      = GetRenderer().GetEnableClusterLodNormalMaps();
    QueryVideoMemory(rec.processVramBytes, rec.vramBudgetBytes);
    stats::shotRecords.push_back(std::move(rec));

    m_shotPendingCapture = false;
    AdvanceShot();
}

// Reduce the baked per-LoD totals the shards already carry.  --dump-stats has no
// other view of the bake, so a --compress whose flag never reached BakerConfig
// would be indistinguishable from a real one: both render correctly.
void RTXMGDemoApp::RecordBakeStats()
{
    stats::BakeStats& b = stats::bakeStats;
    b = {};
    for (const GeometryView& geo : m_scene->GetClusterLodGeometries())
    {
        b.geometries++;
        for (const LodStats& l : geo.lodStats)
        {
            b.bakedBytes  += l.totBytes;
            b.deviceBytes += l.totDeviceBytes;
            b.posBytes    += l.posBytes;
            b.nrmBytes    += l.nrmBytes;
            b.uvBytes     += l.uvBytes;
            b.triangles   += l.totTris;
            b.groups      += l.totGroups;
            b.clusters    += l.totClusters;
            b.compressed  = b.compressed  || l.compressed != 0;
            b.quantizedUv = b.quantizedUv || l.quantUv != 0;
        }
    }
}

void RTXMGDemoApp::DumpStats(const std::string& filepath) const
{
    stats::RunIdentity run;
    run.commandLine = m_args.commandLine;
#if defined(NDEBUG)
    run.buildConfig = "Release";
#else
    run.buildConfig = "Debug";
#endif
    run.graphicsApi = GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12 ? "D3D12" : "VULKAN";
    run.gpuName     = GetDeviceManager()->GetRendererString();
    run.sceneFile   = m_args.meshInputFile;
    run.renderedFrames = m_postLoadBaseFrame >= 0
                             ? uint32_t(int(GetFrameIndex()) - m_postLoadBaseFrame) : 0;
    run.renderWidth  = stats::clusterAccelSamplers.renderSize.x;
    run.renderHeight = stats::clusterAccelSamplers.renderSize.y;

    stats::DumpToJson(filepath, run);
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

void RTXMGDemoApp::SetGlobalIsolationLevel(uint32_t isolationLevel)
{
    isolationLevel = std::clamp(isolationLevel, TessellatorConfig::kMinIsolationLevel, TessellatorConfig::kMaxIsolationLevel);
    if (isolationLevel != m_renderParams.isolationLevel)
    {
        m_renderParams.isolationLevel = isolationLevel;
        m_accelBuilderNeedsUpdate = true;
        GetRenderer().ResetSubframes();
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

void RTXMGDemoApp::SetClusterTessellationPattern(ClusterTessPattern clusterPattern)
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

void RTXMGDemoApp::SetUpdateLodCamera(bool update)
{
    if (update != m_args.updateLodCamera)
    {
        m_args.updateLodCamera = update;
        m_accelBuilderNeedsUpdate = true;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::SetVertexNormalsEnabled(bool enabled)
{
    if (enabled != m_args.enableVertexNormals)
    {
        m_args.enableVertexNormals = enabled;
        m_accelBuilderNeedsUpdate = true;
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::SetClusterLodVertexNormalsEnabled(bool enabled)
{
    if (enabled != m_args.enableClusterLodVertexNormals)
    {
        m_args.enableClusterLodVertexNormals = enabled;
        m_accelBuilderNeedsUpdate = true;
        // Resident group blobs carry their normal words only while this is on,
        // so flipping it has to re-stream (ApplyStreamingBudgets) with the
        // renderer's flag already synced — streaming init reads it.
        GetRenderer().SetEnableClusterLodVertexNormals(enabled);
        ApplyStreamingBudgets();
        GetRenderer().ResetSubframes();
        GetRenderer().ResetDenoiser();
    }
}

void RTXMGDemoApp::SetDollyEnabled(bool e)
{
    m_args.dolly = e;
    // Reset the accumulator so the dolly restarts from the current position.
    m_dollyTime = 0.f;
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

void RTXMGDemoApp::CaptureScreenshotWithUI(nvrhi::IFramebuffer* framebuffer)
{
    bool exiting = glfwWindowShouldClose(GetDeviceManager()->GetWindow());

    // Save screenshot with UI on exit
    if (!m_args.outfileWithUI.empty() && exiting)
    {
        nvrhi::ITexture* tex = framebuffer->getDesc().colorAttachments[0].texture;
        DoSaveScreenshot(tex, m_args.outfileWithUI);
        m_args.outfileWithUI.clear();
    }

    // Interactive Shift+P
    if (!m_screenshotWithUI)
        return;
    m_screenshotWithUI = false;
    nvrhi::ITexture* tex = framebuffer->getDesc().colorAttachments[0].texture;
    static char const base_name[] = "rtxmg_screenshot_ui_";
    int index = GetUniqueFileIndex(base_name, ".png");
    char buf[40];
    std::snprintf(buf, std::size(buf), "%s%04d.png", base_name, index);
    DoSaveScreenshot(tex, buf);
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
        SaveTextureToFile(GetDevice(), GetRenderer().GetCommonPasses().get(), framebufferTexture, nvrhi::ResourceStates::Unknown, filepath.c_str(), false);
        donut::log::message(donut::log::Severity::Info, "Screenshot saved: %s", std::filesystem::absolute(filepath).string().c_str());
    }
}

#if RTXMG_DEV_FEATURES
// Writes a mapped H x W texture as an (H, W, channels) .npy array.
static void WriteTextureNpy(const fs::path& path, const char* descr, uint32_t channels, uint32_t bytesPerTexel,
                            const nvrhi::TextureDesc& desc, const uint8_t* data, size_t rowPitch)
{
    std::string header = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': (" +
                         std::to_string(desc.height) + ", " + std::to_string(desc.width) + ", " +
                         std::to_string(channels) + "), }";
    // Magic (6) + version (2) + length (2) + header must end on a 64-byte boundary.
    header.append(63 - (10 + header.size()) % 64, ' ');
    header.push_back('\n');

    std::ofstream f(path, std::ios::binary);
    const uint16_t headerLen = uint16_t(header.size());
    f.write("\x93NUMPY\x01\x00", 8);
    f.write(reinterpret_cast<const char*>(&headerLen), 2);
    f.write(header.data(), header.size());
    for (uint32_t y = 0; y < desc.height; ++y)
        f.write(reinterpret_cast<const char*>(data + y * rowPitch), size_t(desc.width) * bytesPerTexel);
}

void RTXMGDemoApp::DumpAnimFrame(uint32_t frame)
{
    struct Dump { RTXMGRenderer::Output output; const char* prefix; const char* descr; uint32_t channels, texelBytes; };
    const Dump dumps[] = {
        { RTXMGRenderer::Output::MotionVectors, "mv",  "<f2", 2, 4 },
        { RTXMGRenderer::Output::Albedo,        "alb", "|u1", 4, 4 },
    };

    std::error_code ec;
    fs::create_directories(m_args.animDumpDir, ec);

    for (const Dump& d : dumps)
    {
        nvrhi::ITexture* src = GetRenderer().GetOutputTexture(d.output);
        nvrhi::TextureDesc desc = src->getDesc();
        desc.isRenderTarget = false;
        desc.isUAV          = false;
        desc.initialState   = nvrhi::ResourceStates::CopyDest;
        desc.keepInitialState = true;
        nvrhi::StagingTextureHandle staging = GetDevice()->createStagingTexture(desc, nvrhi::CpuAccessMode::Read);

        m_commandList->open();
        m_commandList->copyTexture(staging, nvrhi::TextureSlice(), src, nvrhi::TextureSlice());
        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
        GetDevice()->waitForIdle();

        size_t rowPitch = 0;
        const auto* data = static_cast<const uint8_t*>(
            GetDevice()->mapStagingTexture(staging, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &rowPitch));
        char name[32];
        std::snprintf(name, std::size(name), "%s_%04u.npy", d.prefix, frame);
        WriteTextureNpy(fs::path(m_args.animDumpDir) / name, d.descr, d.channels, d.texelBytes, desc, data, rowPitch);
        GetDevice()->unmapStagingTexture(staging);
    }
}
#endif

void RTXMGDemoApp::DumpDebugBuffer()
{
    m_dumpDebugBuffer = true;
    m_accelBuilderNeedsUpdate = true; // force a re-build
    GetRenderer().ResetSubframes(); // force a re-render
}

void RTXMGDemoApp::DoDumpDebugBuffer(std::string const& filepath)
{
#if ENABLE_SHADER_DEBUG
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
        auto debugContents = GetRenderer().GetAccelBuilder()->GetDebugBuffer().Download(m_commandList);

        log::info("accel builder debug contents: ");
        
        vectorlog::OutputStream(debugContents, ShaderDebugElement::OutputLambda, nullptr, { .wrap = false, .header = false, .elementIndex = false, .startIndex = 1 });

        std::ofstream fileStream(filepath);
        vectorlog::OutputStream(debugContents, ShaderDebugElement::OutputLambda, &fileStream, { .wrap = false, .header = false, .elementIndex = false, .startIndex = 1 });
    }
#endif
}

float RTXMGDemoApp::GetCPUFrameTime() const
{
    return std::chrono::duration<float, std::milli>(m_currFrameStart -
        m_prevFrameStart)
        .count();
}

// Returns the monitor whose video-mode rectangle contains the given virtual-screen
// point, or nullptr if the point lies on no current monitor (the layout changed
// since the position was saved). Callers treat nullptr as "use the primary monitor".
static GLFWmonitor* MonitorContainingPoint(int px, int py)
{
    int count = 0;
    GLFWmonitor** monitors = glfwGetMonitors(&count);
    for (int i = 0; i < count; ++i)
    {
        int mx, my;
        glfwGetMonitorPos(monitors[i], &mx, &my);
        const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
        if (px >= mx && px < mx + mode->width &&
            py >= my && py < my + mode->height)
        {
            return monitors[i];
        }
    }
    return nullptr;
}

void RTXMGDemoApp::SetWindowState(const WindowState &state)
{
    GLFWwindow* window = GetDeviceManager()->GetWindow();
    
    // Prioritize commandline options and ignore restoring window state
    if (m_args.resolutionSetByCmdLine || m_args.startMaximized || m_args.startFullscreen)
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
        // Route through DeviceManager for the HWND_TOPMOST handling; null means
        // "leave fullscreen" there, so fall back to primary if the monitor is gone.
        GLFWmonitor* monitor = MonitorContainingPoint(state.windowPos.x, state.windowPos.y);
        GetDeviceManager()->SetFullscreen(monitor ? monitor : glfwGetPrimaryMonitor());
    }
}
