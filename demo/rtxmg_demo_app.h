#pragma once

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

#include <array>
#include <future>
#include <optional>
#include <donut/app/ApplicationBase.h>
#include <donut/app/Camera.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/math/quat.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/engine/Scene.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/View.h>

#if DONUT_WITH_DX12
#include <dxgi1_4.h>  // IDXGIAdapter3::QueryVideoMemoryInfo
#endif

#include "args.h"
#include "rtxmg_demo.h"
#include "rtxmg_renderer.h"
#include "gui.h"
#include "shot_list.h"
#include "zrenderer.h"
#include "trackball.h"

using namespace donut::math;

#include "render_params.h"

#include "rtxmg/scene/camera.h"
#include "rtxmg/scene/scene.h"

class UserInterface;
namespace donut::engine { class ThreadPool; }

class RTXMGDemoApp : public donut::app::ApplicationBase
{
public:
    struct WindowState
    {
        donut::math::int2 windowSize = { 0,0 };
        donut::math::int2 windowPos = { 0,0 };  // top-left; for a fullscreen window this is its monitor's origin
        bool isMaximized = false;
        bool isFullscreen = false;
    };

private:
    std::shared_ptr<RTXMGScene> m_scene;
    Camera m_camera;
    Camera m_prevCamera;
    bool m_cameraReset = false;
    Camera m_lodCamera;
    Trackball m_trackBall;

    // --dolly smooth-oscillation state: elapsed seconds within the fixed-period
    // triangle wave (delta-translated each frame).  Reset on dolly-enable /
    // ResetCamera so a full cycle nets zero.
    float m_dollyTime = 0.f;

    // Last value --lpe-sweep pushed to the renderer.  A member, not a function
    // static: a static survives a scene reload and would suppress the re-apply.
    float m_lastSweptLpe = -1.f;

    // --shot-list walk state: every config walked against every position
    // (config outer, so resolution/DLSS/LPE -- which reallocate -- change once
    // per config rather than once per position).  Each position cuts the
    // camera, waits for streaming residency to go flat, then (if the config has
    // any) accumulates and captures once per shot.
    std::vector<ShotPosition> m_shotPositions;
    std::vector<ShotConfig>   m_shotConfigs;
    enum class ShotPhase { Jump, Settle, Accumulate, Done };
    ShotPhase m_shotPhase         = ShotPhase::Jump;
    size_t    m_shotConfigIndex   = 0;
    size_t    m_shotPositionIndex = 0;
    size_t    m_shotIndex         = 0;  // index into m_shotConfigs[m_shotConfigIndex].shots
    uint32_t  m_shotPhaseFrames  = 0;  // frames spent in the current phase
    uint32_t  m_shotSettleFrames = 0;  // what Settle actually cost, for the record
    // Residency-flatness tracking for `settle: "auto"`.
    uint32_t  m_shotLastResidentGroups   = ~0u;
    uint32_t  m_shotLastResidentClusters = ~0u;
    float     m_shotLastEffectiveError   = -1.f;
    uint32_t  m_shotFlatFrames           = 0;
    // Set at the top of the frame whose framebuffer is the one to save; the
    // capture itself has to wait until that framebuffer exists.
    bool      m_shotPendingCapture       = false;
    // Where this position's records start, so its own dump carries only them.
    size_t    m_shotRecordsAtEntryStart  = 0;
    // True once RunShotList() (UI-triggered) is driving the walk, rather than a
    // --shot-list CLI arg -- gates whether completion closes the window or
    // hands control back to the interactive session.
    bool      m_shotInteractiveRun       = false;
    // What a UI-triggered walk changes, put back when it finishes or is aborted.
    struct ShotRestoreState
    {
        std::string                               camera;
        ShadingMode                               shadingMode;
        ColorMode                                 colorMode;
        bool                                      wireframe;
        float                                     exposure;
        bool                                      autoExposure;
        donut::app::StreamlineInterface::DLSSMode dlssMode;
        float                                     lodPixelError;
        bool                                      adaptiveLodError;
        bool                                      normalMapShading;
        GLFWmonitor*                              monitor;  // non-null when fullscreen
        int                                       windowX, windowY, windowW, windowH;
        bool                                      decorated;
    };
    std::optional<ShotRestoreState> m_shotRestore;
    void RestoreAfterShotList();
    // The trackball caches orientation and distance; re-derive them after any
    // direct m_camera change, or the next mouse/key move snaps back.
    void SyncTrackballToCamera();
    // Writes <shot-out>/stats/<config>/<label>.stats.json as the position finishes.
    void DumpPositionStats(const ShotPosition& pos, const ShotConfig& config);
    // Applies a config's "resolution"; a no-op in fullscreen.
    void SetShotOutputResolution(const ShotConfig& config);
    void AdvancePosition();

    // BeginShot force-resets the subframe index, which restarts DLSS-RR's
    // temporal history with it.  Capturing before that has re-accumulated
    // leaves the image dependent on how far along the denoiser happened to be,
    // so every shot waits this long however few frames its mode needs.
    static constexpr uint32_t kMinFramesAfterReset = 30;
    static uint32_t ShotAccumFrames(const Shot& shot)
    {
        return std::max(shot.frames, kMinFramesAfterReset);
    }

public:
    bool ShotListActive() const { return !m_shotPositions.empty() && !m_shotConfigs.empty(); }

    // A --shot-list walk or a -nf run: nothing interactive, and the window must
    // do what the command line says rather than what the last session left.
    bool IsAutomatedRun() const
    {
        return ShotListActive() || m_args.exitAfterFrame >= 0;
    }

    // Camera-only cut to a position's recorded pose; does not touch any
    // config setting (resolution/DLSS/LPE/exposure).
    void JumpCameraToShot(const ShotPosition& pos) { SetCameraString(pos.camera); }
    std::string GetCameraString() const { return m_camera.ToString(); }
    // A camera cut to `camera`; false (camera untouched) if it doesn't parse.
    bool SetCameraString(const std::string& camera)
    {
        if (!m_camera.Set(camera))
            return false;
        m_cameraReset = true;
        SyncTrackballToCamera();
        return true;
    }
    float GetCameraFovY() const { return m_camera.GetFovY(); }
    void  SetCameraFovY(float fovY) { m_camera.SetFovY(fovY); }

    // Starts a walk over `positions` x `configs` exactly like --shot-list, but
    // leaves the window open on completion and restores the camera and view
    // settings it started from -- this run started from the UI.
    void RunShotList(std::vector<ShotPosition> positions, std::vector<ShotConfig> configs);
    // Cancels an interactive run in progress; a no-op otherwise.
    void AbortShotList();
    bool IsShotListRunning() const { return ShotListActive() && m_shotPhase != ShotPhase::Done; }
    size_t GetShotConfigIndex() const { return m_shotConfigIndex; }
    size_t GetShotPositionIndex() const { return m_shotPositionIndex; }

private:
    std::filesystem::path ShotImageDir() const
    {
        return std::filesystem::path(m_args.shotOutDir) / "images";
    }
    std::filesystem::path ShotStatsDir() const
    {
        return std::filesystem::path(m_args.shotOutDir) / "stats";
    }
    // Process-resident video memory and the adapter's budget. False on Vulkan.
    bool QueryVideoMemory(uint64_t& usage, uint64_t& budget) const;

    void UpdateShotList();
    void BeginShot();
    void AdvanceShot();
    void CaptureShot(nvrhi::IFramebuffer* framebuffer);

    std::shared_ptr<donut::engine::DirectionalLight> m_sunLight;
    std::shared_ptr<donut::engine::SceneGraphNode>   m_sunLightNode;
    nvrhi::CommandListHandle m_commandList;

    // path to executable (because argv0 is not reliable)
    std::filesystem::path m_binaryPath;

    // path to local folder w/ media assets
    std::filesystem::path m_mediaPath;

    UIData m_ui;
    Args m_args;

    // Latched in Init() from the AdapterInfo the VRAM check already resolves.
    // The DXGI adapter is kept for QueryVideoMemoryInfo; on Vulkan the physical
    // device comes off the device manager instead.
    uint64_t    m_physicalVramBytes = 0;
    std::string m_adapterName;
#if DONUT_WITH_DX12
    nvrhi::RefCountPtr<IDXGIAdapter3> m_dxgiAdapter3;
#endif
    // VK_EXT_memory_budget, requested optional at device creation.
    bool m_vkMemoryBudgetAvailable = false;
    // NVAPI under -gd with the DXR cluster intrinsics; see the device creation.
    bool m_forceDxrClusterIntrinsics = false;

    // --tess-budget-sweep ladder base, 0 until latched.  Cleared on scene load.
    uint32_t m_tessSweepBaseMaxClusters = 0;

    // --normalmap-sweep: capture / enable / capture / disable / capture.  The
    // odd steps reload the scene, so the step counter deliberately survives a
    // reload while m_postLoadBaseFrame is reset by it.
    static constexpr int kNormalMapSweepSteps = 5;
    int         m_nmSweepStep = 0;
    std::string m_nmSweepCapture;  // filename awaiting the end-of-frame capture

    // UI-edited state captured before a (re)load and restored by
    // ReconcileLoadedScene, after the scene file and the command line have had
    // their say.  LoadScene reads the grid pair straight from here.
    uint32_t m_savedGridCopies = 1;
    float    m_savedGridGap = 0.f;
    bool     m_savedBlasSharing = false;
    bool     m_savedBlasCaching = false;
    bool     m_savedBlasMerging = false;
    uint32_t m_savedSharingLevels = 0;
    uint32_t m_savedCachingLevels = 0;

    // Zero-init: only a subset of fields is explicitly set at startup; the
    // rest (e.g. subFrameIndex) must not be MSVC debug-fill garbage.
    RenderParams m_renderParams = {};
    int2 m_displaySize;
    int2 m_renderSize;
    float m_lodBias = 1.0f;
    DenoiserMode m_denoiserMode = DenoiserMode::None; // Desired denoiser mode, but m_renderParams contains effective denoiser mode

    int2 m_loadFrameRange = { std::numeric_limits<int>::max(), std::numeric_limits<int>::min() };

    // Scene loading is a multi-phase pipeline driven by AdvanceLoad (see Render).
    // LoadScene() runs the whole CPU half on the load thread: scene construction,
    // the KTX2 texture-mip budget, RTXMGScene's load (cluster-LOD shard mmap /
    // bake, per-geometry GPU metadata, texture requests) and the instance grid.
    // Every phase after it is main-thread, because queue submission races the
    // Streamline-wrapped present.
    std::shared_ptr<donut::vfs::IFileSystem> m_pendingSceneFs;
    std::filesystem::path                    m_pendingSceneFile;
    // Closed upload command lists recorded by the load thread — one per load-pool
    // worker, plus RTXMGScene's own — submitted by the LoadPhase::Submit step.
    std::vector<nvrhi::CommandListHandle>    m_pendingUploadCLs;
    // The scene the load thread built; moved into m_scene by LoadPhase::Reconcile.
    std::unique_ptr<RTXMGScene>              m_loadingScene;
    std::future<bool>                        m_loadResult;
    // Set by the destructor so a load in flight when the window closes bails at
    // its next phase boundary instead of running to completion.
    std::atomic<bool>                        m_loadCancelled{ false };

    // Background pool for parallel texture file-read + (zstd) decode.  Persists
    // across loads; its tasks drain while the loading overlay renders and
    // ProcessRenderingThreadCommands finalizes textures on the main thread.
    std::unique_ptr<donut::engine::ThreadPool> m_textureLoadPool;

    // Where the scene load has got to.  Until it reaches Done the app shows the
    // loading screen, and the scene / camera / renderer are not wired up, so
    // Animate and every input handler must stay idle.
    enum class LoadPhase : uint8_t
    {
        None,       // no load in flight
        Import,     // load thread: bake, scene build, texture budget + requests, grid
        Submit,     // execute the command lists the load thread recorded
        Textures,   // budgeted texture finalize pump
        Reconcile,  // scene settings, CLI precedence, saved UI state, UpdateParams
        Build,      // FinishedLoading + accel structs + camera framing
        Done,
    };
    LoadPhase m_loadPhase = LoadPhase::None;
    // TextureCache finalized-count the Memory tab's texture bytes were last
    // summed at; a change detector only, so it does not matter that
    // TextureCache::Reset() leaves the counter running.  UINT32_MAX forces a
    // recount (armed on every scene load).
    uint32_t m_textureMemFinalizedMark = UINT32_MAX;
    std::chrono::steady_clock::time_point m_textureLoadStart = {};
    // Frame index at which the scene finished loading; -1 until then.  -nf exit /
    // -s screenshot / -dp pixel dump count relative to this, so a headless run
    // renders exitAfterFrame frames of the LOADED scene however long loading took.
    int      m_postLoadBaseFrame   = -1;

    std::chrono::steady_clock::time_point m_currFrameStart = {};
    std::chrono::steady_clock::time_point m_prevFrameStart = {};
    float m_animationTime = 0.0f;

    std::unique_ptr<RTXMGRenderer> m_renderer;

    nvrhi::BufferHandle m_lerpPosesBuffer;  // LerpKeyFramesPose per scene pose
    nvrhi::BindingLayoutHandle m_lerpVerticesBL;
    nvrhi::ComputePipelineHandle m_lerpVerticesPSO;

    bool m_reloadShaders = false;
    bool m_animationUpdated = false;
    bool m_accelBuilderNeedsUpdate = true;
    bool m_dumpFineTess = false;
    bool m_screenshot = false;
    bool m_screenshotWithUI = false;
    bool m_dumpDebugBuffer = false;
    bool m_dumpPixelDebug = false;
    bool m_readPixelPick = false;

    std::array<int, 3> m_debugSurfaceClusterLaneIndex = { -1, -1, -1 };

    // DLSS State
#if DONUT_WITH_STREAMLINE
    using StreamlineInterface = donut::app::StreamlineInterface;
    StreamlineInterface::DLSSPreset m_dlssLastPreset = StreamlineInterface::DLSSPreset::eDefault;
    StreamlineInterface::DLSSMode m_dlssLastMode = StreamlineInterface::DLSSMode::eOff;
    int2 m_dlssLastDisplaySize;

    StreamlineInterface::DLSSSettings m_RecommendedDLSSSettings;
    StreamlineInterface::DLSSRRSettings m_RecommendedDLSSRRSettings;
#endif

    int m_argc;
    const char** m_argv;

    UserInterface* m_gui;
    WindowState m_windowState;

    struct MessageCallback : public nvrhi::IMessageCallback
    {
        explicit MessageCallback(donut::app::DeviceManager* deviceManager)
            : m_deviceManager(deviceManager)
        {}
        const donut::app::DeviceManager* m_deviceManager;
        void message(nvrhi::MessageSeverity severity, const char* messageText) override;
    };
    MessageCallback m_messageCallback;

    static void InstallDeviceRemovedAwareLogCallback();

    void UpdateParams();
    void UpdateDLSSSettings();
    void LimitFrameRate();

    void DoSaveScreenshot(nvrhi::ITexture* framebufferTexture, std::string const& filename = "");
    void DumpStats(std::string const& filepath) const;
    void DoDumpFineTess(std::string const& filename = "");
    void DoDumpDebugBuffer(std::string const& filename = "");
        
    // Blends scene poses [firstPose, firstPose + numPoses), all of `mesh`, in one dispatch.
    void LerpPoses(const SubdivisionSurface& mesh, uint32_t firstPose, uint32_t numPoses);
    void DispatchGPUAnimation();

public:
    // AppBase Overrides
    // The load thread's body: see LoadPhase::Import.  We spawn the thread and
    // consume the result ourselves, so ApplicationBase never calls this.
    bool LoadScene(std::shared_ptr<donut::vfs::IFileSystem> fs,
        const std::filesystem::path& sceneFileName) override;

    bool KeyboardUpdate(int key, int scancode, int action, int mods) override;
    bool MousePosUpdate(double xpos, double ypos) override;
    bool MouseButtonUpdate(int button, int action, int mods) override;
    bool MouseScrollUpdate(double xoffset, double yoffset) override;

    void BackBufferResizing() override;
    void BackBufferResized(const uint32_t width, const uint32_t height,
        const uint32_t sampleCount) override;
    void WindowPosUpdate(int xpos, int ypos) override;
    void Animate(float fElapsedTimeSeconds) override;

    // ApplicationBase::Render models loading as one blocking LoadScene() call and
    // decides "all textures finalized" by draining a queue our textures have not
    // been queued into yet.  Ours is a multi-phase pipeline that submits GPU work,
    // so we drive it here instead and never call ApplicationBase::SceneLoaded().
    void Render(nvrhi::IFramebuffer* framebuffer) override;
    // One bounded step of the load pipeline, then the loading screen.  The UI pass
    // draws the progress bar on top.
    void AdvanceLoad(nvrhi::IFramebuffer* framebuffer, bool texturesProcessed);
    // LoadPhase::Reconcile: scene settings, CLI precedence, saved UI state.
    void ReconcileLoadedScene();
    void RenderScene(nvrhi::IFramebuffer* framebuffer) override;

    // Skipping Render() on an unfocused window would break three things.  Headless
    // (`-nf`): m_FrameIndex still advances, so exit-after-N fires without N
    // frames ever rendering.  Loading: every phase after the import runs inside
    // Render(), so a backgrounded load would stall until the user refocuses.
    // `--shot-list`: it clears exitAfterFrame to own the exit itself, so it needs
    // its own clause or the walk stalls the moment the window loses focus.
    bool ShouldRenderUnfocused() override
    {
        return m_args.exitAfterFrame >= 0 || ShotListActive() || IsSceneLoading();
    }

public:
    RTXMGDemoApp(donut::app::DeviceManager* deviceManager, std::string &windowTitle, int argc,
        const char** argv);
    virtual ~RTXMGDemoApp();

    bool Init();
    void ResetCamera();
    // ResetCamera plus the startup exposure / auto-exposure, which a shot-list
    // jump pins: the user-facing reset (F, the Reset Camera button).
    void ResetView();
    bool SetEnvmapTex(const std::string& filePath);

    // GPU scene build (instance buffers + cluster-LOD accel structs + camera
    // framing) — the LoadPhase::Build step.
    void BuildLoadedScene();

    // Hides ApplicationBase::IsSceneLoading(), which tracks a load thread we no
    // longer let it own.  Nothing in donut reads the base version.
    bool IsSceneLoading() const
    {
        return m_loadPhase != LoadPhase::None && m_loadPhase != LoadPhase::Done;
    }

    // True once the texture requests are in flight, so the loading bar can switch
    // from the bake/upload phases to plotting decoded/requested.
    bool IsLoadingTextures() const;
    struct TextureLoadProgress { uint32_t requested = 0, decoded = 0; };
    TextureLoadProgress GetTextureLoadProgress() const;

    // Refresh the Memory tab's resident material-texture bytes/count (the scene
    // totals it compares against come from the load-time budget pre-pass).
    // Cheap every frame: early-outs unless the finalized count moved.
    void UpdateTextureMemStats();

    // Re-bucket stats::vramBreakdown from the samplers + a driver query, once a
    // frame.  A view of state that already exists, not a second source of truth.
    void UpdateVramBreakdown();

    // Logs when cluster tessellation demand exceeds its allocation (the UI's
    // "Tessellation memory budget exceeded"), at most once per 10 s, and once
    // when it drops back under.
    void LogTessellationLimit();

    // Logs each 10% move of the GPU's adaptive tessellation rate scale.
    void LogAdaptiveTessRate();
    float m_adaptiveTessLoggedScale = 1.f;

#if RTXMG_DEV_FEATURES
    // --anim-dump: write this frame's albedo and motion vectors (stalls the GPU).
    void DumpAnimFrame(uint32_t frame);
#endif
    std::chrono::steady_clock::time_point m_tessLimitLogTime{};
    bool m_tessLimitLogged = false;

    // Reduce GeometryView::lodStats into stats::bakeStats, once per scene load.
    void RecordBakeStats();

    void HandleSceneLoad(std::string const& m_filepath,
        std::string const& mediapathm, int2 frameRange = { std::numeric_limits<int>::max(), std::numeric_limits<int>::min() });

    // Re-load the currently loaded scene (used by the Cluster-LoD grid UI to
    // rebuild the scene with a new instance count / spacing).
    void ReloadCurrentScene();

    // Bake config: JSON persistence in the cache directory.
    // GetCurrentBakerConfig() layers: BakerConfig defaults < bake_config.json < CLI args.
    // ApplyBakerConfigAndReload() saves the config to JSON then reloads the scene.
    BakerConfig GetCurrentBakerConfig() const;
    std::filesystem::path GetBakeConfigJsonPath() const;
    void SaveBakerConfigJson(const BakerConfig& cfg);
    void ApplyBakerConfigAndReload(const BakerConfig& cfg);

    // Post-rebake report: populated at the end of LoadScene(), committed in BuildLoadedScene().
    // Snapshot mirrors the fields of stats::BakeStats (see profiler/statistics.h).
    struct BakeReport {
        bool     valid         = false;
        double   bakeSeconds   = 0.0;
        uint32_t workerThreads = 0;
        int64_t  memDeltaMiB   = 0;   // available-RAM change (positive = consumed)
        struct Snapshot {
            bool     valid      = false;
            uint32_t geometries = 0;
            uint32_t groups     = 0;
            uint32_t clusters   = 0;
            uint64_t triangles  = 0;
            uint64_t bakedBytes  = 0;
            uint64_t deviceBytes = 0;
            bool     compressed  = false;
            bool     quantizedUv = false;
            uint32_t compressionPosDropBits = 0;
            uint32_t compressionTexDropBits = 0;
        } before, after;
    };
    const BakeReport& GetBakeReport() const { return m_bakeReport; }

    // Apply changed streaming budgets (resident groups / geometry pool / CLAS
    // pool) without reloading the scene: idle the GPU, then rebuild only the
    // acceleration-structure resources for the loaded scene (re-streams from
    // scratch into the resized pools).
    void ApplyStreamingBudgets();

    // Requested cluster-LoD budgets; they reach the renderer on ApplyStreamingBudgets
    // or the next scene load.
    const ClusterLodBudgets& GetClusterLodBudgets() const   { return m_args.clodBudgets; }
    void SetClusterLodBudgets(const ClusterLodBudgets& b)   { m_args.clodBudgets = b; }

    // Texture budget and normal-map LOADING decide what gets read off disk, so
    // unlike the pool budgets above they only take effect on a scene reload.
    void ApplyTextureSettingsAndReload(int textureBudgetMB, bool loadNormalMaps);
    int  GetTextureBudgetMB() const { return m_args.textureBudgetMB; }
    bool GetNormalMapsEnabled() const { return m_args.enableNormalMaps; }

    // Shading with the loaded maps, by contrast, is only a path-tracer
    // permutation, so it toggles live.
    bool GetNormalMapShading() const { return m_args.normalMapShading; }
    void SetNormalMapShading(bool b)
    {
        m_args.normalMapShading = b;
        GetRenderer().SetEnableClusterLodNormalMaps(m_args.enableNormalMaps && b);
    }

    // What the VRAM Budget window plots against and clamps to.  GetVramBytes()
    // honours the dev --vram-mb override; GetPhysicalVramBytes() never does, so
    // the window can label a simulated reading as such.
    uint64_t GetVramBytes() const
    {
        return m_args.vramOverrideMB > 0 ? (uint64_t(m_args.vramOverrideMB) << 20) : m_physicalVramBytes;
    }
    uint64_t GetPhysicalVramBytes() const { return m_physicalVramBytes; }
    bool     IsVramSimulated() const { return m_args.vramOverrideMB > 0; }
    const std::string& GetAdapterName() const { return m_adapterName; }

    // Driver-reported VRAM for this process, the only view that includes DLSS,
    // nvrhi and driver allocations the sample cannot itself account for.
    // False (leaving both untouched) when the API or driver does not expose it.
    bool QueryDriverVram(uint64_t& outUsage, uint64_t& outBudget) const;

    const Args& GetArgs() const { return m_args; }

    uint32_t GetLodGridCopies() const { return m_args.lodGridCopies; }
    void     SetLodGridCopies(uint32_t copies) { m_args.lodGridCopies = std::max(1u, copies); }
    float    GetLodGridGap() const { return m_args.lodGridGap; }
    void     SetLodGridGap(float gap) { m_args.lodGridGap = std::max(0.0f, gap); }
    bool     GetLodGridRandomize() const { return (m_args.lodGridBits & 0x38u) != 0; }
    void     SetLodGridRandomize(bool on)
    {
        if (on) m_args.lodGridBits |= 0x28u;
        else    m_args.lodGridBits &= ~0x38u;
    }

    // lpe-sweep diagnostic: cycle lodPixelError through 1,2,4,...,32,...,1 every
    // N frames (drives residency churn without camera input).
    bool     GetLpeSweepEnabled() const { return m_args.lpeSweep; }
    void     SetLpeSweepEnabled(bool e) { m_args.lpeSweep = e; }
    uint32_t GetLpeSweepFrameInterval() const { return m_args.lpeSweepFrameInterval; }
    void     SetLpeSweepFrameInterval(uint32_t n) { m_args.lpeSweepFrameInterval = std::max(1u, n); }

    // dolly diagnostic: smoothly translate the camera along its look direction,
    // oscillating one scene-diagonal forward then back at the camera move speed
    // (see Animate).
    bool  GetDollyEnabled() const { return m_args.dolly; }
    void  SetDollyEnabled(bool e);
    // Fly (WASD) move speed, world units/second.  Also driven by the mouse wheel
    // (no Alt); Alt+wheel zooms instead.  Clamped to the trackball's log bounds.
    float GetCameraMoveSpeed() const { return m_trackBall.MoveSpeed(); }
    void  SetCameraMoveSpeed(float s)
    {
        m_trackBall.SetMoveSpeed(std::clamp(s, Trackball::kMinMoveSpeed, Trackball::kMaxMoveSpeed));
    }
    bool     GetVsyncEnabled() const { return GetDeviceManager()->IsVsyncEnabled(); }
    void     SetVsyncEnabled(bool enabled) { m_args.vsync = enabled; GetDeviceManager()->SetVsyncEnabled(enabled); }
    uint32_t GetMaxFps() const { return m_args.maxFps; }
    void     SetMaxFps(int maxFps) { m_args.maxFps = uint32_t(std::max(0, maxFps)); }
    const RTXMGScene& GetScene() const { return *m_scene; }
    RTXMGRenderer& GetRenderer();

    float GetCPUFrameTime() const;

    ///////////////////////////////////////////////////////
    // GUI access
    ///////////////////////////////////////////////////////
    void SetGui(UserInterface* gui) { m_gui = gui; }
    const std::filesystem::path& GetBinaryPath() const { return m_binaryPath; }
    const std::filesystem::path& GetMediaPath() const { return m_mediaPath; }
    void SetMediaPath(const std::filesystem::path& path) { m_mediaPath = path; }
    UIData& GetUIData() { return m_ui; }

    WindowState GetWindowState() const { return m_windowState; }
    void SetWindowState(const WindowState& state);
    // Refresh m_windowState (position, monitor/fullscreen, maximized, size) from
    // the live GLFW window. Called on both size and position changes so the
    // persisted state stays current even when only the monitor changes.
    void CaptureWindowState();

    void DumpFineTess();
    void SaveScreenshot();
    void SaveScreenshotWithUI() { m_screenshotWithUI = true; }
    void CaptureScreenshotWithUI(nvrhi::IFramebuffer* framebuffer);
    void DumpDebugBuffer();
    void ReloadShaders() { m_reloadShaders = true; }
    void RebuildAS() { m_accelBuilderNeedsUpdate = true; }

    TessellatorConfig::MemorySettings GetTessMemSettings() const { return m_args.tessMemorySettings; }
    void SetTessMemSettings(const TessellatorConfig::MemorySettings& settings);

    bool GetUpdateLodCamera() const { return m_args.updateLodCamera; }
    void SetUpdateLodCamera(bool update);

    float GetFineTessellationRate() const { return m_args.fineTessellationRate; }
    void  SetFineTessellationRate(float rate);

    float GetCoarseTessellationRate() const { return m_args.coarseTessellationRate; }
    void  SetCoarseTessellationRate(float rate);

    // Under cluster_tess budget pressure the GPU scales the tessellation rates by
    // GetBuildStats().tessRateScale (<= 1), recovering slowly once pressure drops.
    bool  GetAdaptiveTessRate() const { return m_args.adaptiveTessRate; }
    void  SetAdaptiveTessRate(bool b) { m_args.adaptiveTessRate = b; m_accelBuilderNeedsUpdate = true; }
    const ClusterTessStatistics& GetBuildStats() const { return m_BuildStats; }
    // Max over the cluster_tess budgets of desired / allocated, from m_BuildStats.
    float GetTessBudgetLoad() const;

    int GetIsolationLevelSharp() const { return m_args.isoLevelSharp; }
    int GetIsolationLevelSmooth() const { return m_args.isoLevelSmooth; }
    
    void SetGlobalIsolationLevel(uint32_t isolationLevel);
    uint32_t GetGlobalIsolationLevel() const { return m_renderParams.isolationLevel; }

    TessellatorConfig::VisibilityMode GetTessellatorVisibilityMode() const { return m_args.visMode; }
    void                              SetTessellatorVisibilityMode(TessellatorConfig::VisibilityMode visMode);

    bool GetBackfaceVisibilityEnabled() const { return m_args.enableBackfaceVisibility; }
    void SetBackfaceVisibilityEnabled(bool enabled);

    void SetDisplacementScale(float scale);
    float GetDisplacementScale() const { return m_renderParams.globalDisplacementScale; }

    ClusterTessPattern GetClusterTessellationPattern() const
    {
        return (ClusterTessPattern)m_renderParams.clusterPattern;
    }
    void SetClusterTessellationPattern(ClusterTessPattern clusterPattern);

    TessellatorConfig::AdaptiveTessellationMode GetAdaptiveTessellationMode() const { return m_args.tessMode; }
    void SetAdaptiveTessellationMode(TessellatorConfig::AdaptiveTessellationMode mode);

    bool GetFrustumVisibilityEnabled() const { return m_args.enableFrustumVisibility; }
    void SetFrustumVisibilityEnabled(bool enabled);

    bool GetHiZVisibilityEnabled() const { return m_args.enableHiZVisibility; }
    void SetHiZVisibilityEnabled(bool enabled);

    bool GetVertexNormalsEnabled() const { return m_args.enableVertexNormals; }
    void SetVertexNormalsEnabled(bool enabled);

    bool GetClusterLodVertexNormalsEnabled() const { return m_args.enableClusterLodVertexNormals; }
    void SetClusterLodVertexNormalsEnabled(bool enabled);

    bool GetAccelBuildLoggingEnabled() const { return m_args.enableAccelBuildLogging; }
    void SetAccelBuildLoggingEnabled(bool enabled) { m_args.enableAccelBuildLogging = enabled; }

    std::array<int, 3>& GetDebugSurfaceClusterLaneIndex() { return m_debugSurfaceClusterLaneIndex; }

    // Desired denoiser mode
    void SetDenoiserMode(DenoiserMode denoiserMode);
    DenoiserMode GetDenoiserMode() const { return m_denoiserMode; }
    DenoiserMode GetEffectiveDenoiserMode() const { return m_renderParams.denoiserMode; }

    void NextShadingMode()
    {
        auto& renderer = GetRenderer();
        if (renderer.GetShowMicroTriangles())
            return;

        ShadingMode shadingMode = ShadingMode((int(renderer.GetShadingMode()) + 1) % int(ShadingMode::SHADING_MODE_COUNT));
        renderer.SetShadingMode(shadingMode);
    }

    void NextTonemapper()
    {
        auto& renderer = GetRenderer();
        if (renderer.GetShowMicroTriangles())
            return;

        TonemapOperator op = TonemapOperator((int(renderer.GetTonemapOperator()) + 1) % int(TonemapOperator::Count));
        renderer.SetTonemapOperator(op);
    }

    void IncrementMaxBounces(int delta)
    {
        auto& renderer = GetRenderer();
        if (renderer.GetEffectiveShadingMode() == ShadingMode::PT)
        {
            int maxBounces = std::min(renderer.GetPTMaxBounces() + delta, 10);
            renderer.SetPTMaxBounces(maxBounces);
        }
    }

    void ToggleWireframe()
    {
        auto& renderer = GetRenderer();
        if (renderer.GetShowMicroTriangles())
            return;

        bool wireframe = !renderer.GetWireframe();
        renderer.SetWireframe(wireframe);
    }

    // Bitmask of the geometry paths this scene actually renders with.  Both bits
    // on a mixed scene; the two are independent, not exclusive.
    uint32_t GetLiveColorModePaths() const
    {
        if (!m_scene)
            return kColorModeAnyPath;
        uint32_t paths = 0u;
        if (m_scene->HasClusterLod())
            paths |= kColorModeClusterLod;
        if (!m_scene->GetSubdMeshes().empty())
            paths |= kColorModeClusterTess;
        return paths ? paths : kColorModeAnyPath;
    }

    void IncrementColorMode(int delta)
    {
        auto& renderer = GetRenderer();
        if (renderer.GetShowMicroTriangles())
            return;

        // Skip modes no live path can serve, or the cycle lands on flat grey.
        const uint32_t live  = GetLiveColorModePaths();
        const int      count = int(ColorMode::COLOR_MODE_COUNT);
        int            index = int(renderer.GetColorMode());
        for (int step = 0; step < count; ++step)
        {
            index = (index + count + delta) % count;
            if (GetColorModePaths(ColorMode(index)) & live)
                break;
        }
        renderer.SetColorMode(ColorMode(index));
    }

    void ToggleTimeView()
    {
        auto& renderer = GetRenderer();
        if (renderer.GetEffectiveShadingMode() == ShadingMode::PT)
        {
            bool timeView = !renderer.GetTimeView();
            renderer.SetTimeView(timeView);
        }
    }

    void ToggleUpdateLodCamera()
    {
        SetUpdateLodCamera(!GetUpdateLodCamera());
    }

    ClusterTessStatistics m_BuildStats;

    BakeReport m_bakeReport;         // committed report, read by GUI
    BakeReport m_pendingBakeReport;  // accumulated by LoadScene() on the worker thread
    bool       m_rebakeTriggered = false;  // set by ApplyBakerConfigAndReload

    // Camera saved before a reload so ReconcileLoadedScene can restore it.
    float3 m_savedCamEye    = {};
    float3 m_savedCamLookat = {};
    float3 m_savedCamUp     = float3(0.f, 1.f, 0.f);
    float  m_savedCamFovY   = 35.f;
    bool   m_hasSavedCamera = false;
};
