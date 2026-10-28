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

#pragma once

#include <filesystem>
#include <map>

#include "rtxmg/cluster_lod/baker.h"

#include <donut/app/imgui_renderer.h>
#include <donut/app/StreamlineInterface.h>
#include <donut/core/math/math.h>
#include <donut/engine/TextureCache.h>

#include "implot.h"
#include "rtxmg_demo.h"
#include "shot_list.h"

#include "rtxmg/profiler/gui.h"

using namespace donut::math;

namespace fs = std::filesystem;

class RTXMGDemoApp;
struct GLFWindow;
class GeometryInspector;

#ifdef AUDIO_ENGINE_ENABLED
#include <audio/audio.h>
#include <audio/waveFile.h>
#else
namespace audio
{
    class Engine {};
    class Voice {};
}
#endif

struct MediaAsset
{
    enum class Type : uint8_t
    {
        OBJ_FILE = 0,
        OBJ_SEQUENCE
    } type = Type::OBJ_FILE;

    std::string name;

    std::string sequenceName; // decorated sequence name to display in GUI
    std::string
        sequenceFormat; // format string to generate paths to individual files
    int padding =
        0; // number of digits in sequence numbers (or 0 if no padding detected)

    std::string wavePath;
    float waveStartTime = 0.f; // offset on audio start time

    int2 frameRange = { std::numeric_limits<int>::max(),
                       std::numeric_limits<int>::min() };
    float frameRate = 24.f;

    bool IsSequence() const { return type == Type::OBJ_SEQUENCE; }

    char const* GetName() const
    {
        return IsSequence() ? sequenceName.c_str() : name.c_str();
    }

    void GrowFrameRange(int frame)
    {
        frameRange.x = std::min(frame, frameRange.x);
        frameRange.y = std::max(frame, frameRange.y);
    };
};

typedef std::map<std::string, MediaAsset> MediaAssetsMap;

struct TimeLineEditorState
{
    template <typename T> constexpr T clamp(T value, T lower, T upper)
    {
        return std::min(std::max(value, lower), upper);
    }

    enum class Playback : uint8_t { Pause = 0, Play } mode = Playback::Pause;
    bool loop = true;
    int2 frameRange = { 0, 0 };
    float frameRate = 30.f;
    float startTime = 0.f;
    float endTime = 0.f;
    float currentTime = 0.f;

    std::function<void(TimeLineEditorState const&)> playCallback;
    std::function<void(TimeLineEditorState const&)> pauseCallback;
    std::function<void(TimeLineEditorState const&)> setTimeCallback;

    void Update(float elapsedTime);
    float AnimationTime() const { return currentTime - startTime; }

    // programmatic manipulation
    inline bool IsPlaying() const { return mode == Playback::Play; }
    inline bool IsPaused() const { return mode == Playback::Pause; }

    inline void SetFrame(float time)
    {
        currentTime = clamp(time / frameRate, startTime, endTime);
    }
    inline void StepForward()
    {
        currentTime = clamp(currentTime + 1.f / frameRate, startTime, endTime);
        if (setTimeCallback)
            setTimeCallback(*this);
    }
    inline void StepBackward()
    {
        currentTime = clamp(currentTime - 1.f / frameRate, startTime, endTime);
        if (setTimeCallback)
            setTimeCallback(*this);
    }
    inline void Rewind()
    {
        currentTime = startTime;
        if (setTimeCallback)
            setTimeCallback(*this);
    }
    inline void FastForward()
    {
        currentTime = endTime;
        if (setTimeCallback)
            setTimeCallback(*this);
    }

    inline void PlayClicked()
    {
        bool paused = IsPaused();

        if (paused && playCallback)
            playCallback(*this);
        else if (pauseCallback)
            pauseCallback(*this);

        mode = paused ? TimeLineEditorState::Playback::Play
            : TimeLineEditorState::Playback::Pause;
    }
};

struct UIData
{
    bool showUI = true;

    // path to imgui.ini settings file (auto-saved by imgui)
    std::string iniFilepath;

    MediaAssetsMap mediaAssets;

    MediaAsset const* currentAsset = nullptr;

    void SelectCurrentAsset(const std::string& name)
    {
        currentAsset = nullptr;
        if (name.empty())
            return;
        if (auto it = mediaAssets.find(name); it != mediaAssets.end())
            currentAsset = &it->second;
    }

    bool audioMuted = false;
    float audioStartTime = 0.f;

    // various filters for the scene selector
    bool includeJsonAssets = true;
    bool includeObjAssets = true;
    bool includeGltfAssets = true;

    std::array<char const*, 5> formatFilters() const;

    std::array<char const*, 5> folderFilters() const;

    bool forceRebuildAccelStruct = true;
    bool enableMonolithicClusterBuild = false;

    bool showBakeConfigWindow      = false;
    bool showBakeReportWindow      = false;
    bool showGridInstancingWindow  = false;
    bool showShotListWindow    = false;
    bool focusBakeReport       = false;
    bool refreshBakeConfig     = false; // force re-init of baseline on next open
    
    TimeLineEditorState timeLineEditorState;

    std::shared_ptr<donut::engine::LoadedTexture> envmap = nullptr;
    std::string envmapFilepath = "";

    // DLSS
    using StreamlineInterface = donut::app::StreamlineInterface;
    StreamlineInterface::DLSSMode dlssMode = StreamlineInterface::DLSSMode::eMaxQuality;
    bool dlssUseLodBiasOverride = false;
    float dlssLodBiasOverride = 0.f;
    StreamlineInterface::DLSSPreset dlssPreset = StreamlineInterface::DLSSPreset::eDefault;
    StreamlineInterface::DLSSRRPreset dlssRRPreset = StreamlineInterface::DLSSRRPreset::eDefault;
};

class UserInterface : public donut::app::ImGui_Renderer
{
public:
    UserInterface(RTXMGDemoApp& app);
    void BackBufferResized(const uint32_t width,
        const uint32_t height,
        const uint32_t sampleCount) override;
    void buildUI() override;

    void SetAnimationRange(int2 frameRange, float frameRate);
    void Animate(float elapsedTimeSeconds) override;

    RTXMGDemoApp& GetApp() { return m_app; }

    ImFont* GetIconicFont() { return m_iconicFont; }

    ImGuiContext* GetImGuiContext() const { return m_imgui; }
    ImPlotContext* GetImPlotContext() const { return m_implot; }

    ProfilerGUI& GetProfilerGUI() { return m_profiler; }

    bool CustomInit(std::shared_ptr<donut::engine::ShaderFactory> shaderFactory);

    // Scene-attribute-dependent UI setup (animation range + audio voice), from
    // RTXMGDemoApp::ReconcileLoadedScene().  It cannot live in CustomInit, which
    // runs before the scene exists under asynchronous loading.
    void OnSceneLoaded();

    // ESC: show/hide every UI window for a clean viewport.
    void ToggleUIVisible() { m_uiVisible = !m_uiVisible; }

private:
    // Every VRAM budget is staged and committed by Apply, so the proposed plot
    // can show the whole configuration before any of it is allocated.
    struct StagedBudgets
    {
        int  textureBudgetMB   = 0;
        bool normalMaps        = false;
        int  maxResidentGroups = 0;
        int  geometryPoolMB    = 0;
        int  geometryBlockMB   = 0;
        int  clasPoolMB        = 0;
        int  blasCachingPoolMB = 0;
        int  maxLoadsPerFrame  = 0;
        int  renderClusterBits = 0;
        int  tessMaxKClusters  = 0;
        int  tessVertexMB      = 0;
        int  tessClasMB        = 0;
    };

    void BuildMemoryWarning(int2 windowSize);
    void BuildInspectorWindow();
    void BuildVramBudgetWindow();
    void BuildUIMain(int2 windowSize);
    // Fixed top-left strip of window toggles, above every other window.
    void BuildTopBar(ImVec2 itemSize);
    void BuildHelpWindow();

    void RegisterMidiBindings();
    // One "Settings" window; each section draws its own collapsing header.
    void BuildSettingsWindow(int2 windowSize, ImVec2 itemSize);
    void BuildSceneSection(ImVec2 itemSize);
#if RTXMG_DEV_FEATURES
    void BuildDebugSection(ImVec2 itemSize);
#endif
    void BuildRenderingSection(ImVec2 itemSize);
    void BuildClusterLodSection();
    void BuildTessellationSection();
    // The budget controls both sections used to own, now drawn by the VRAM
    // Budget window.  They edit m_staged; nothing commits until Apply.
    void BuildClusterLodBudgetControls();
    void BuildTessBudgetControls();
    // The two settings that need a scene reload share one warning and one
    // staged-value sync, so the Cluster LODs checkbox and the VRAM Budget
    // window's Apply cannot drift apart.
    void BuildReloadWarningText();
    // Live budget state, and the staged copy of it that Apply commits.
    StagedBudgets LiveBudgets();
    void SyncStagedBudgets();
    // Commits the staged pool budgets; reallocate re-streams the cluster-LoD
    // pools now, which a scene reload makes redundant.
    void ApplyStagedPoolBudgets(bool reallocate);
    // Total the proposed plot would show for `s`.  Also the 90%-of-VRAM ceiling
    // test, so a field can refuse an edit that would not fit.
    uint64_t ProposedBytes(const StagedBudgets& s);
    // One staged int field: highlights when it differs from `live`, clamps to
    // [minVal,maxVal], and rejects an increase that pushes past the ceiling.
    void StagedField(const char* label, int& staged, int live, int step,
                     int stepFast, int minVal, int maxVal);
#if DONUT_WITH_STREAMLINE
    void BuildDenoiserSection();
#endif
    void UpdateProfilerStats(int2 windowSize);

    void BuildUITimeline(int2 windowSize, float timeline_width);
    void BuildUIEnvmap(ImVec2 itemSize);

    void BuildMissColorUI();
    void BuildBakeConfigWindow();
    void BuildBakeReportWindow();
    void BuildGridInstancingWindow();
    void BuildShotListWindow();

    bool BuildTimeLineEditor(TimeLineEditorState& state, float2 size);

    void SetupAudioVoice(const std::string& wavepath, float startTime = 0.f);
    void SetupAudioEngine();
    void MuteAudio(bool mute);

    void LoadAsset(MediaAsset const& asset, std::string const& name,
        int2 frameRange);

    static MediaAssetsMap FindMediaAssets(fs::path const& mediapath,
        char const* const* folder_filters,
        char const* const* format_filters);
    // 'iconic' open-source TTF font (lots of standard icons to make buttons
    // with)

    static constexpr float const iconicFontSize = 18.f;
    static uint16_t const* GetOpenIconicFontGlyphRange();
    static char const* GetNVSansFontRgCompressedBase85TTF();
    static char const* GetNVSansFontBoldCompressedBase85TTF();
    static char const* GetOpenIconicFontCompressedBase85TTF();

    ImFont* AddFontFromMemoryCompressedBase85TTF(const char* data, float fontSize,
        const uint16_t* range);
    bool FolderDialog(std::string& m_filepath);
    bool FileDialog(bool bOpen, char const* filters, std::string& m_filepath);

    void SetupIniHandler();

private:
    RTXMGDemoApp& m_app;

    ImFont* m_iconicFont = nullptr;
    ImFont* m_nvidiaRgFont = nullptr;
    ImFont* m_nvidiaBldFont = nullptr;

    ImGuiContext* m_imgui = nullptr;
    ImPlotContext* m_implot = nullptr;
    
    ProfilerGUI m_profiler;

    // HUD FPS accumulator: 1/frameTime of a single frame swings wildly, so the
    // counter is averaged over a fixed refresh window (see BuildUI).
    float m_fpsWindowMs     = 0.f;
    int   m_fpsWindowFrames = 0;
    int   m_fpsDisplay      = -1;

    // VRAM Budget window: every memory knob, plus current-vs-proposed plots.
    bool m_showVramBudget = false;
    StagedBudgets m_staged;
    // False until the window first opens, when m_staged is filled from live state.
    bool m_stagedValid = false;
    // Set by the Cluster LOD panel's "Load Normal Maps" shortcut: flashes the
    // matching field so the window does not just open on an unrelated section.
    bool m_highlightLoadMaps = false;

    // Geometry Inspector window (per-mesh subdivision / cluster-LOD stats).
    bool m_showInspector = false;
    // Settings and Help windows, both toggled from the top bar.
    bool m_showSettings = true;
    bool m_showHelp     = false;
    // ESC hides every window; the loading splash is exempt.
    bool m_uiVisible = true;
    // Built on first use, once the fonts and the ImPlot context exist.
    std::shared_ptr<GeometryInspector> m_inspector;

    std::shared_ptr<audio::Engine> m_audioEngine;
    std::unique_ptr<audio::Voice> m_voice;

    // Audio fallback for a scene with no embedded audio: LoadAsset() (menu scene
    // switch) sets it to the selected media asset's wave track, and
    // OnSceneLoaded() consumes and clears it.  Empty for the startup load.
    std::string m_pendingAudioFallbackPath;
    float m_pendingAudioFallbackStart = 0.f;

    // Bake Config window state.
    BakerConfig m_baselineBakerConfig;    // config when window opened (for diff)
    BakerConfig m_editingBakerConfig;     // live-edited copy
    bool        m_bakeConfigWindowOpen = false; // previous-frame open state (rising-edge init)

    // Grid Instancing window state.
    uint32_t m_stagedGridCopies     = 1;
    float    m_stagedGridGap        = 1.0f;
    bool     m_stagedGridRandomize  = true;
    bool     m_gridInstancingWindowOpen = false;

    // Camera section's editable camera string.
    std::string m_cameraStringBuf;
    bool        m_cameraStringEdited  = false;
    bool        m_cameraStringInvalid = false;

    // Shot List window state.
    bool                       m_shotListWindowOpen = false;
    std::string                m_shotListPathBuf;         // ImGui::InputText target
    std::string                m_shotListDefaultScene;    // scene the default list was last looked up for
    std::filesystem::path      m_shotListLoadedPath;
    std::vector<ShotPosition>  m_shotListPositions;
    std::vector<ShotConfig>    m_shotListConfigs;
    std::vector<bool>          m_shotListConfigSelected;  // which configs "Run" includes
    int                        m_shotListSelectedPos = -1;
    std::string                m_shotListNewLabelBuf;
    std::string                m_shotListStatus;          // last load/save result
    size_t                     m_shotListRunningConfigCount = 0; // for the progress display
};