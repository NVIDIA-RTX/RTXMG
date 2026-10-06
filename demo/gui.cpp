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

#include <imgui_internal.h>

#ifndef _WIN32
#include <climits>
#include <cstdio>
#include <unistd.h>
#else
#include <ShObjIdl.h>
#include <ShlObj_core.h>
#include <Windows.h>
#include <codecvt>
#include <locale>
constexpr int const PATH_MAX = MAX_PATH;
#endif // _WIN32

#include "rtxmg_demo_app.h"
#include "gui.h"
#include "inspector.h"
#include "implot.h"

#include "rtxmg/cluster_lod/baking/bake_progress.h"

#include <charconv>
#include <filesystem>
#include <algorithm>
#include <type_traits>
#include <string>
#include <sstream>
#include <cfloat>
#include <cmath>

#include <donut/app/imgui_renderer.h>
#include "rtxmg/scene/scene.h"
#include "rtxmg/scene/texture_loader.h"
#include "rtxmg/cluster_lod/pass.h"
#include "rtxmg/cluster_lod/streaming.h"
#include "rtxmg/subdivision/subdivision_surface.h"
#include "rtxmg/utils/constants.h"
#include "rtxmg/utils/formatters.h"
#include "rtxmg/cluster_tess/tessellator_constants.h"

#include "korgi.h"
#include "rtxmg/profiler/gui.h"
#include "rtxmg/profiler/statistics.h"

#include <donut/app/StreamlineInterface.h>

namespace fs = std::filesystem;

using namespace donut;

#define UI_RED ImVec4(1.f, 0.f, 0.f, 1.f)
#define UI_SAGE ImVec4(.3f, .4f, .35f, 1.f)
#define UI_DARKBLUE ImVec4(.03f, .08f, .3f, 1.f)

constexpr float kItemWidth = 200.0f;

// Viewport centre in ImGui layout space.  GetMainViewport()->GetCenter() is in
// unscaled DisplaySize, so on a DPI-scaled display it lands off-screen.
static ImVec2 LayoutCenter()
{
    const ImGuiIO& io = ImGui::GetIO();
    return ImVec2(io.DisplaySize.x / io.DisplayFramebufferScale.x * 0.5f,
                  io.DisplaySize.y / io.DisplayFramebufferScale.y * 0.5f);
}

constexpr const char* kNoClusterLod = "This scene has no cluster-LOD geometry.";
constexpr const char* kNoTess       = "This scene has no subdivision geometry.";

// A collapsing header that greys out and stays shut when the scene has no
// geometry of that kind, rather than offering controls nothing can bind to.
// Returns true when the section is both enabled and open.
static bool BeginDisableableSection(const char* label, bool enabled, const char* whyDisabled)
{
    if (!enabled)
    {
        ImGui::SetNextItemOpen(false, ImGuiCond_Always);
        ImGui::BeginDisabled();
        ImGui::CollapsingHeader(label);
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", whyDisabled);
        return false;
    }
    return ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen);
}

UserInterface::UserInterface(RTXMGDemoApp& app)
    : ImGui_Renderer(app.GetDeviceManager()), m_app(app)
{
    char const* nvidiaRgGlyphData = GetNVSansFontRgCompressedBase85TTF();
    m_nvidiaRgFont =
        AddFontFromMemoryCompressedBase85TTF(nvidiaRgGlyphData, 15.f, nullptr);

    char const* nvidiaBldGlyphData = GetNVSansFontBoldCompressedBase85TTF();
    m_nvidiaBldFont =
        AddFontFromMemoryCompressedBase85TTF(nvidiaBldGlyphData, 30.f, nullptr);

    ImGui::GetIO().FontDefault = m_nvidiaRgFont;

    char const* iconicGlyphsData = GetOpenIconicFontCompressedBase85TTF();
    uint16_t const* iconicGlyphsRange = GetOpenIconicFontGlyphRange();

    m_iconicFont = AddFontFromMemoryCompressedBase85TTF(iconicGlyphsData, 14.f,
        iconicGlyphsRange);

    m_imgui = ImGui::GetCurrentContext();
    m_implot = ImPlot::CreateContext();

    ImPlotStyle& style = ImPlot::GetStyle();

    style.FitPadding = ImVec2(0.1f, 0.1f);
    style.PlotPadding = ImVec2(2, 5);
    style.LegendPadding = ImVec2(2, 2);

    m_app.SetGui(this);

    SetupIniHandler();

    SetupAudioEngine();
}

void UserInterface::SetupIniHandler()
{
    ImGuiIO& io = ImGui::GetIO();

    if (const fs::path& binaryPath = app::GetDirectoryWithExecutable(); !binaryPath.empty())
    {
        UIData& ui = m_app.GetUIData();
        ui.iniFilepath = (binaryPath / "imgui.ini").generic_string();
        io.IniFilename = ui.iniFilepath.c_str();
    }

    io.IniSavingRate = 60.f;  // save every minute only or on quit

    static struct Settings
    {
        bool audioMuted = false;

        bool jsonAssetsFilter = false;
        bool objAssetsFilter = false;
        bool gltfAssetsFilter = false;

        bool displayStats = true;  // Profiler window shown by default (until imgui.ini overrides)
        std::string profilerTab;  // persisted active Profiler tab (by name)

        bool wantApply = false;

        RTXMGDemoApp::WindowState windowState = {};
    } settings;

    ImGuiSettingsHandler ini_handler;
    ini_handler.TypeName = "RTXMG";
    ini_handler.TypeHash = ImHashStr(ini_handler.TypeName);
    ini_handler.UserData = this;

    ini_handler.ReadOpenFn = [](ImGuiContext*, ImGuiSettingsHandler*, const char* name) -> void* {
        settings.wantApply = true;
        return &settings;
    };


    ini_handler.ApplyAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler) {
        
        if (settings.wantApply)
        {
            auto* gui = reinterpret_cast<UserInterface*>(handler->UserData);
            auto* app = &gui->m_app;

            gui->GetProfilerGUI().displayGraphWindow = settings.displayStats;
            gui->GetProfilerGUI().requestedTab       = settings.profilerTab;

            gui->MuteAudio(settings.audioMuted);

            UIData& ui = app->GetUIData();
            ui.includeJsonAssets = settings.jsonAssetsFilter;
            ui.includeObjAssets  = settings.objAssetsFilter;
            ui.includeGltfAssets = settings.gltfAssetsFilter;

            const fs::path& mediaPath = app->GetMediaPath();
            if (!mediaPath.empty())
            {
                assert(ui.mediaAssets.empty());
                auto folder_filters = ui.folderFilters();
                auto format_filters = ui.formatFilters();
                ui.mediaAssets = FindMediaAssets(mediaPath, folder_filters.data(), format_filters.data());
            }

            // Not on an automated run: whatever a previous interactive session
            // left in imgui.ini would silently override -res and any per-entry
            // resolution -- a stale WindowIsFullscreen=1 pins every capture to
            // the display's size no matter what the shot list asked for.
            if (!app->IsAutomatedRun())
                app->SetWindowState(settings.windowState);

            settings.wantApply = false;
        }
    };


    ini_handler.ReadLineFn = [](ImGuiContext*, ImGuiSettingsHandler* handler, void* entry, const char* line) {

        int audioMuted = 0;
        if (std::sscanf(line, "AudioMuted=%d", &audioMuted) == 1)
            settings.audioMuted = audioMuted;

        uint32_t json = 0, obj = 0, gltf = 0;
        if (std::sscanf(line, "FormatFilters={ json=%d, obj=%d, gltf=%d }", &json, &obj, &gltf) == 3)
        {
            settings.jsonAssetsFilter = (bool)json;
            settings.objAssetsFilter  = (bool)obj;
            settings.gltfAssetsFilter = (bool)gltf;
        }
        else if (std::sscanf(line, "FormatFilters={ json=%d, obj=%d }", &json, &obj) == 2)
        {
            // legacy (pre-gltf) settings file
            settings.jsonAssetsFilter = (bool)json;
            settings.objAssetsFilter  = (bool)obj;
        }

        int displayStats = false;
        if (std::sscanf(line, "DisplayStatistics=%d", &displayStats) == 1)
            settings.displayStats = displayStats != 0;

        // Tab names can contain spaces, so copy the rest of the line (not %s).
        if (std::strncmp(line, "ProfilerTab=", 12) == 0)
        {
            settings.profilerTab = line + 12;
            while (!settings.profilerTab.empty() &&
                   (settings.profilerTab.back() == '\r' || settings.profilerTab.back() == '\n'))
                settings.profilerTab.pop_back();
        }

        donut::math::int2 windowSize{};
        if (std::sscanf(line, "WindowSize=%d,%d", &windowSize.x, &windowSize.y) == 2)
        {
            settings.windowState.windowSize = windowSize;
        }
        donut::math::int2 windowPos{};
        if (std::sscanf(line, "WindowPos=%d,%d", &windowPos.x, &windowPos.y) == 2)
        {
            settings.windowState.windowPos = windowPos;
        }
        int windowIsMaximized = false;
        if (std::sscanf(line, "WindowIsMaximized=%d", &windowIsMaximized) == 1)
        {
            settings.windowState.isMaximized = windowIsMaximized != 0;
        }
        int windowIsFullscreen = false;
        if (std::sscanf(line, "WindowIsFullscreen=%d", &windowIsFullscreen) == 1)
        {
            settings.windowState.isFullscreen = windowIsFullscreen != 0;
        }
    };

    ini_handler.WriteAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {

        auto* gui = reinterpret_cast<UserInterface*>(handler->UserData);
        auto* app = &gui->m_app;
        
        UIData& ui = app->GetUIData();
        
        settings.audioMuted = ui.audioMuted;
        settings.jsonAssetsFilter = ui.includeJsonAssets;
        settings.objAssetsFilter  = ui.includeObjAssets;
        settings.gltfAssetsFilter = ui.includeGltfAssets;
        settings.displayStats = gui->GetProfilerGUI().displayGraphWindow;
        settings.profilerTab  = gui->GetProfilerGUI().activeTab;

        settings.windowState = app->GetWindowState();
        
        buf->reserve(buf->size() + 2);  // ballpark reserve
        buf->appendf("[%s][%s]\n", handler->TypeName, "Settings");
        buf->appendf("AudioMuted=%d\n", settings.audioMuted);
        buf->appendf("FormatFilters={ json=%d, obj=%d, gltf=%d }\n",
                     settings.jsonAssetsFilter, settings.objAssetsFilter, settings.gltfAssetsFilter);
        buf->appendf("DisplayStatistics=%d\n", settings.displayStats);
        buf->appendf("ProfilerTab=%s\n", settings.profilerTab.c_str());
        buf->appendf("WindowSize=%d,%d\n", settings.windowState.windowSize.x, settings.windowState.windowSize.y);
        buf->appendf("WindowPos=%d,%d\n", settings.windowState.windowPos.x, settings.windowState.windowPos.y);
        buf->appendf("WindowIsMaximized=%d\n", settings.windowState.isMaximized);
        buf->appendf("WindowIsFullscreen=%d\n", settings.windowState.isFullscreen);

        buf->append("\n");
    };

    ImGui::AddSettingsHandler(&ini_handler);
}

void UserInterface::BackBufferResized(const uint32_t width,
    const uint32_t height,
    const uint32_t sampleCount)
{
}

void UserInterface::buildUI()
{
    // While the scene loads asynchronously, draw only a progress bar and skip the
    // main UI: BuildUIMain reads scene / renderer / m_args state that the load
    // thread is still writing.
    if (m_app.IsSceneLoading())
    {
        const rtxmg::BakeProgress& bp = rtxmg::GetBakeProgress();

        // Center in ImGui's layout space (DisplaySize / DisplayFramebufferScale);
        // GetMainViewport()->Size is the unscaled DisplaySize, so using it directly
        // mis-centers on a DPI-scaled display.
        const ImGuiIO& io = ImGui::GetIO();
        const float layoutW = io.DisplaySize.x / io.DisplayFramebufferScale.x;
        const float layoutH = io.DisplaySize.y / io.DisplayFramebufferScale.y;
        ImGui::SetNextWindowPos(ImVec2(layoutW * 0.5f, layoutH * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(std::min(620.f, layoutW * 0.7f), 0.f), ImGuiCond_Always);
        ImGui::Begin("Loading scene", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs);

        const uint32_t total  = bp.GetTotal();
        const uint32_t done   = bp.GetDone();
        const uint32_t cached = bp.GetCached();

        if (!m_app.IsLoadingTextures() && bp.IsActive() && total > 0)
        {
            // The bar is reused across load phases; the label names the current one.
            if (cached > 0)
                ImGui::Text("%s:  %u / %u   (%u cached)", bp.GetLabel().c_str(), done, total, cached);
            else
                ImGui::Text("%s:  %u / %u", bp.GetLabel().c_str(), done, total);
            char overlay[32];
            snprintf(overlay, sizeof(overlay), "%.0f%%", bp.GetFraction() * 100.f);
            ImGui::ProgressBar(bp.GetFraction(), ImVec2(-1.f, 0.f), overlay);

            // Per-worker in-flight names (bake phase only; the upload phase
            // publishes none).
            const std::vector<std::string> inFlight = bp.GetInFlight();
            if (!inFlight.empty())
            {
                ImGui::Spacing();
                ImGui::TextUnformatted("Currently processing:");
                constexpr size_t kMaxShow = 12;
                for (size_t i = 0; i < inFlight.size() && i < kMaxShow; ++i)
                    ImGui::BulletText("%s", inFlight[i].c_str());
                if (inFlight.size() > kMaxShow)
                    ImGui::Text("   ... and %zu more", inFlight.size() - kMaxShow);
            }
        }
        else if (m_app.IsLoadingTextures())
        {
            // Decode is the slow part and runs to completion on the load thread;
            // the GPU upload overlaps it and is drained before the phase ends.
            const RTXMGDemoApp::TextureLoadProgress tp = m_app.GetTextureLoadProgress();
            const float frac = tp.requested ? float(tp.decoded) / float(tp.requested) : 0.f;
            ImGui::Text("Loading textures:  %u / %u", tp.decoded, tp.requested);
            char overlay[32];
            snprintf(overlay, sizeof(overlay), "%.0f%%", frac * 100.f);
            ImGui::ProgressBar(frac, ImVec2(-1.f, 0.f), overlay);

            // Cluster-LOD metadata pre-upload, pumped on the main thread
            // alongside the texture finalize — show its bar too while active.
            if (bp.IsActive() && bp.GetTotal() > 0)
            {
                ImGui::Spacing();
                ImGui::Text("%s:  %u / %u", bp.GetLabel().c_str(), bp.GetDone(), bp.GetTotal());
                char overlay2[32];
                snprintf(overlay2, sizeof(overlay2), "%.0f%%", bp.GetFraction() * 100.f);
                ImGui::ProgressBar(bp.GetFraction(), ImVec2(-1.f, 0.f), overlay2);
            }
        }
        else
        {
            // Parse / cache-hit / GPU-upload phases: no per-geometry bake work.
            // No true indeterminate bar in ImGui, so animate a repeating fill.
            ImGui::TextUnformatted("Loading scene...");
            const float t    = float(ImGui::GetTime());
            const float frac = t - float(int(t));  // 0..1 sawtooth
            ImGui::ProgressBar(frac, ImVec2(-1.f, 0.f), "");
        }

        ImGui::End();
        return;
    }

    int width, height;
    m_app.GetDeviceManager()->GetWindowDimensions(width, height);
    float scaleX, scaleY;
    m_app.GetDeviceManager()->GetDPIScaleInfo(scaleX, scaleY);

    float layoutToDisplay = std::min(scaleX, scaleY);
    float contentScale = layoutToDisplay > 0.f ? (1.0f / layoutToDisplay) : 1.0f;

    // Layout is done at lower resolution than scaled up virtually past the render target m_size
    // any element beyond this range is clipped.
    width = int(width * contentScale);
    height = int(height * contentScale);

    const auto uiStart = std::chrono::steady_clock::now();
    if (m_uiVisible)
        BuildUIMain({ width, height });
    stats::frameSamplers.uiBuildTime.PushBack(
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - uiStart).count());
}

template<typename E, size_t N>
static int ImGuiComboFromArray(const char* name, E* selected, const std::array<const char*, N>& labels)
{
    bool valueChanged = false;

    int selectedIndex = int(*selected);
    const char* selectedLabel = selectedIndex < labels.size() ? labels[selectedIndex] : "Unknown";

    if (ImGui::BeginCombo(name, selectedLabel))
    {
        int index = 0;
        for (const auto& label : labels)
        {
            bool isSelected = selectedIndex == index;
            if (ImGui::Selectable(label, isSelected))
            {
                *selected = (E)index;
                valueChanged = true;
            }
            if (isSelected) ImGui::SetItemDefaultFocus();
            index++;
        }
        ImGui::EndCombo();
    }
    return valueChanged;
}

// Colour modes split by the geometry path that can render them.  Modes the live
// scene cannot serve stay selectable but are marked, because on a mixed scene
// "unsupported" means half the image goes grey, not all of it.
static bool ColorModeCombo(const char* name, ColorMode* selected, uint32_t livePaths)
{
    struct Group { const char* label; uint32_t path; };
    static const Group kGroups[] = {
        { "Common",       kColorModeAnyPath     },
        { "Cluster LOD",  kColorModeClusterLod  },
        { "Cluster Tess", kColorModeClusterTess },
    };

    bool valueChanged = false;
    const int selectedIndex = int(*selected);
    const char* selectedLabel =
        selectedIndex < int(kColorModeNames.size()) ? kColorModeNames[selectedIndex] : "Unknown";

    if (ImGui::BeginCombo(name, selectedLabel))
    {
        for (const Group& group : kGroups)
        {
            ImGui::SeparatorText(group.label);
            for (int index = 0; index < int(ColorMode::COLOR_MODE_COUNT); ++index)
            {
                const uint32_t paths = GetColorModePaths(ColorMode(index));
                if (paths != group.path)
                    continue;

                const bool supported = (paths & livePaths) != 0u;
                char label[128];
                snprintf(label, sizeof(label), supported ? "%s" : "%s  (not in this scene)",
                         kColorModeNames[index]);

                if (!supported)
                    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                const bool isSelected = selectedIndex == index;
                if (ImGui::Selectable(label, isSelected))
                {
                    *selected    = ColorMode(index);
                    valueChanged = true;
                }
                if (!supported)
                    ImGui::PopStyleColor();
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    return valueChanged;
}

void UserInterface::RegisterMidiBindings()
{
    auto& renderer = m_app.GetRenderer();

    KORGI_BUTTON_CALLBACK(0, Play, [this]()
    {
        TimeLineEditorState& state = GetApp().GetUIData().timeLineEditorState;
        state.PlayClicked();
    });
    KORGI_BUTTON_CALLBACK(0, Rewind, [this]()
    {
        TimeLineEditorState& state = GetApp().GetUIData().timeLineEditorState;
        state.Rewind();
    });
    KORGI_BUTTON_CALLBACK(0, FastForward, [this]()
    {
        TimeLineEditorState& state = GetApp().GetUIData().timeLineEditorState;
        state.FastForward();
    });

    KORGI_KNOB_CALLBACK(0, Slider1, 0.0, 10.0, [this, &renderer](float val)
    {
        renderer.SetExposure(val);
    });
    KORGI_BUTTON_CALLBACK(0, Record, [this]()
    {
        GetApp().SaveScreenshot();
    });
    KORGI_BUTTON_CALLBACK(0, S1, [this]()
    {
        GetApp().NextTonemapper();
    });
    KORGI_BUTTON_CALLBACK(0, Cycle, [this]()
    {
        GetApp().ResetCamera();
    });
    KORGI_BUTTON_CALLBACK(0, S2, [this]()
    {
        GetApp().IncrementMaxBounces(1);
    });
    KORGI_BUTTON_CALLBACK(0, M2, [this]()
    {
        GetApp().IncrementMaxBounces(-1);
    });
    KORGI_BUTTON_CALLBACK(0, S3, [this]()
    {
        GetApp().IncrementColorMode(1);
    });
    KORGI_BUTTON_CALLBACK(0, M3, [this]()
    {
        GetApp().IncrementColorMode(-1);
    });
    KORGI_BUTTON_CALLBACK(0, R3, [this]()
    {
        GetApp().ToggleWireframe();
    });
    KORGI_KNOB_CALLBACK(0, Slider2, 1, 2000, [this](float val)
    {
        GetApp().SetFineTessellationRate(val / 1000.0f);
    });
}

void UserInterface::BuildTopBar(ImVec2 itemSize)
{
    UIData& uiData = GetApp().GetUIData();

    ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(.65f);
    ImGui::Begin("##topbar", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings |
                 ImGuiWindowFlags_AlwaysAutoResize);

#ifdef AUDIO_ENGINE_ENABLED
    static const char* unmutedGlyph = (char*)(u8"\ue0d5" "## unmuted");
    static const char* mutedGlyph = (char*)(u8"\ue0d7" "## muted");

    bool muted = uiData.audioMuted;
    if (muted)
        ImGui::PushStyleColor(ImGuiCol_Button, UI_RED);
    ImGui::PushFont(m_iconicFont);
    if (ImGui::Button(muted ? mutedGlyph : unmutedGlyph, { 20.f, itemSize.y }))
    {
        if (m_audioEngine)
            m_audioEngine->mute(uiData.audioMuted = !muted);
    }
    if (muted)
        ImGui::PopStyleColor();
    ImGui::PopFont();
    if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
        ImGui::SetTooltip("Mute audio.");
    ImGui::SameLine();
#endif

    ImGui::PushFont(m_iconicFont);
    if (ImGui::Button((char const*)(u8"\ue02c"
        "## screenshot"),
        { 0.f, itemSize.y }))
    {
        m_app.SaveScreenshot();
    }
    ImGui::PopFont();
    if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
        ImGui::SetTooltip("Capture a screenshot.");

    // Snapshot each flag before its button: the button toggles it, so gating the
    // Pop on the post-click value would unbalance the style-color stack.
    auto toggleButton = [](const char* label, bool& flag, const char* tooltip)
    {
        const bool active = flag;
        if (active)
            ImGui::PushStyleColor(ImGuiCol_Button, UI_DARKBLUE);
        ImGui::SameLine();
        if (ImGui::Button(label))
            flag = !flag;
        if (active)
            ImGui::PopStyleColor();
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("%s", tooltip);
    };

    // Ordered by how often they get reached for.
    toggleButton("Settings", m_showSettings, "Toggle the Settings window.");
    toggleButton("Profiler", m_profiler.displayGraphWindow, "Toggle the Profiler window.");
    toggleButton("Inspector", m_showInspector,
                 "Toggle the geometry Inspector window\n(per-mesh subdivision / cluster-LOD stats).");
    toggleButton("VRAM", m_showVramBudget,
                 "Toggle the VRAM Budget window\n(every memory budget, plus current vs proposed consumption).");
    toggleButton("Help", m_showHelp, "Toggle the keyboard / mouse controls window.");

    ImGui::End();
}

void UserInterface::BuildSettingsWindow(int2 screenLayoutSize, ImVec2 itemSize)
{
    if (!m_showSettings)
        return;

    UIData& uiData = GetApp().GetUIData();

    auto& renderer = m_app.GetRenderer();

    // Below the top bar, which owns the top-left corner.
    const char* kWindowName = "Settings";
    SetConstrainedWindowPos(kWindowName, ImVec2(10, 50), ImVec2(0.0f, 0.0f), MakeImVec2(screenLayoutSize));
    ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSizeConstraints(ImVec2(100.f, 200.f), ImVec2(float(screenLayoutSize.x), screenLayoutSize.y - 110.0f));
    ImGui::Begin(kWindowName, &m_showSettings, ImGuiWindowFlags_None);

    ImGui::PushItemWidth(kItemWidth);

    ImGui::PushStyleColor(ImGuiCol_Header, UI_SAGE);
    BuildSceneSection(itemSize);
#if RTXMG_DEV_FEATURES
    BuildDebugSection(itemSize);
#endif
    BuildRenderingSection(itemSize);
    ImGui::PopStyleColor();
    ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_Header, UI_SAGE);
    BuildClusterLodSection();
    ImGui::PopStyleColor();
    ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_Header, UI_SAGE);
    BuildTessellationSection();
    ImGui::PopStyleColor();

    ImGui::Spacing();

#if DONUT_WITH_STREAMLINE
    ImGui::PushStyleColor(ImGuiCol_Header, UI_SAGE);
    BuildDenoiserSection();
    ImGui::PopStyleColor();
    ImGui::Spacing();
#endif
    ImGui::PopItemWidth();
    ImGui::End();
}

void UserInterface::BuildSceneSection(ImVec2 itemSize)
{
    UIData& uiData = GetApp().GetUIData();

    if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
    {
        fs::path mediapath = m_app.GetMediaPath();

        // media folder
        bool objFilesNeedUpdate = false;
        {
            ImGui::PushFont(m_iconicFont);
            if (ImGui::Button((char const*)(u8"\ue06b"
                "## media path"),
                { 0.f, itemSize.y }))
            {
                std::string folderpath = mediapath.generic_string();
                if (FolderDialog(folderpath))
                {
                    mediapath = fs::path(folderpath).lexically_normal();
                    m_app.SetMediaPath(mediapath);
                    uiData.currentAsset = nullptr;
                    objFilesNeedUpdate = true;
                }
            }
            ImGui::PopFont();
            ImGui::SameLine();

            float buttonWidth = ImGui::GetItemRectSize().x + ImGui::GetStyle().ItemSpacing.x;
            ImGui::SetNextItemWidth(kItemWidth - buttonWidth);
            char buf[1024] = { 0 };
            std::strncpy(buf, mediapath.generic_string().c_str(), std::size(buf));
            if (ImGui::InputText("Data Folder", buf, std::size(buf),
                ImGuiInputTextFlags_EnterReturnsTrue))
            {
                mediapath = buf;
                m_app.SetMediaPath(buf);
                objFilesNeedUpdate = true;
            }
            if (ImGui::IsItemHovered() &&
                ImGui::GetCurrentContext()->HoveredIdTimer > .5f && !mediapath.empty())
                ImGui::SetTooltip("%s", mediapath.generic_string().c_str());
        }

        // "##" suffix keeps the visible "Scene" label but gives a distinct ID: the
        // enclosing CollapsingHeader("Scene") pushes no ID scope, so a bare "Scene"
        // checkbox would hash to the same ID.
        if (ImGui::Checkbox("Scene##sceneFilter", &uiData.includeJsonAssets))
            objFilesNeedUpdate = true;
        ImGui::SameLine();
        if (ImGui::Checkbox("Obj", &uiData.includeObjAssets))
            objFilesNeedUpdate = true;
        ImGui::SameLine();
        if (ImGui::Checkbox("Gltf", &uiData.includeGltfAssets))
            objFilesNeedUpdate = true;

        if (objFilesNeedUpdate || uiData.mediaAssets.empty())
        {
            auto folderFilters = uiData.folderFilters();
            auto formatFilters = uiData.formatFilters();

            // Store current asset name before refreshing the map
            std::string currentAssetName = uiData.currentAsset ? uiData.currentAsset->GetName() : "";
            
            uiData.mediaAssets = FindMediaAssets(mediapath, folderFilters.data(),
                formatFilters.data());
            
            // Restore current asset selection if it still exists
            if (!currentAssetName.empty())
            {
                uiData.SelectCurrentAsset(currentAssetName);
            }
            else
            {
                uiData.currentAsset = nullptr;
            }
        }

        char const* currentAssetName =
            uiData.currentAsset ? uiData.currentAsset->GetName() : nullptr;
        // "##" suffix: distinct ID for the third widget labelled "Scene".
        if (ImGui::BeginCombo("Scene##sceneSelect", currentAssetName,
            ImGuiComboFlags_HeightLarge))
        {
            for (const auto& [key, asset] : uiData.mediaAssets)
            {
                bool isSequence = asset.IsSequence();

                std::string const& name = isSequence ? asset.sequenceName : key;

                bool isSelected = currentAssetName && (name == currentAssetName);

                if (ImGui::Selectable(name.c_str(), isSelected))
                {
                    LoadAsset(asset, name, asset.frameRange);
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered() &&
            ImGui::GetCurrentContext()->HoveredIdTimer > .5f && uiData.currentAsset &&
            !uiData.currentAsset->name.empty())
        {
            ImGui::SetTooltip("%s", uiData.currentAsset->GetName());
        }

        if (ImGui::Button("Grid Instancing"))
            m_app.GetUIData().showGridInstancingWindow = true;
    }

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button("Reset Camera"))
            m_app.ResetCamera();

        bool updateLodCamera = m_app.GetUpdateLodCamera();
        if (ImGui::Checkbox("Update LOD Camera", &updateLodCamera))
            m_app.SetUpdateLodCamera(updateLodCamera);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Keep the camera every view-dependent geometry decision runs\n"
                              "against in sync with the render camera. Uncheck to freeze it:\n"
                              "cluster-LoD detail selection, cluster_tess tessellation rate,\n"
                              "frustum culling, HiZ occlusion and the viewpoint the HiZ\n"
                              "pyramid is rendered from all lock together while the render\n"
                              "camera keeps moving.");

#if RTXMG_DEV_FEATURES
        bool dolly = m_app.GetDollyEnabled();
        if (ImGui::Checkbox("Dolly", &dolly))
            m_app.SetDollyEnabled(dolly);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Diagnostic: smoothly translate the camera one scene-diagonal\n"
                              "forward then back at \"Camera Speed\", looping.");
#endif

        // Slider range must match Trackball::kMin/kMaxMoveSpeed.
        float camSpeed = m_app.GetCameraMoveSpeed();
        if (ImGui::SliderFloat("Camera Speed", &camSpeed, 1e-3f, 1e5f,
                               "%.3f u/s", ImGuiSliderFlags_Logarithmic))
            m_app.SetCameraMoveSpeed(camSpeed);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Camera fly (WASD) speed in world units/second (log scale).\n"
                              "Mouse wheel adjusts it too; hold Alt + wheel to zoom instead.\n"
                              "Shift = x3 faster, Ctrl = x0.1 finer while moving.");
    }
}

#if RTXMG_DEV_FEATURES
void UserInterface::BuildDebugSection(ImVec2 itemSize)
{
    UIData& uiData = GetApp().GetUIData();

    auto& renderer = m_app.GetRenderer();

    if (ImGui::CollapsingHeader("Debug", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::PushFont(m_iconicFont);
        if (ImGui::Button((char const*)(u8"\ue0b3"
            "## reload shaders"),
            { 0.f, itemSize.y }))
        {
            m_app.ReloadShaders();
        }
        ImGui::PopFont();
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Reload Shaders (CTRL+R)");

        ImGui::SameLine();
        ImGui::PushFont(m_iconicFont);
        if (ImGui::Button((char const*)(u8"\ue071"
            "## dump fill clusters"),
            { 0.f, itemSize.y }))
        {
            m_app.DumpFineTess();
        }
        ImGui::PopFont();
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Dump Fill Clusters.");

#if ENABLE_SHADER_DEBUG
        ImGui::SameLine();
        ImGui::PushFont(m_iconicFont);
        if (ImGui::Button((char const*)(u8"\ue028"
            "## dump debug buffer"),
            { 0.f, itemSize.y }))
        {
            m_app.DumpDebugBuffer();
        }
        ImGui::PopFont();
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Dump Debug Buffer.");
#endif

        ImGui::SameLine();
        ImGui::Checkbox("Monolithic ClusterBuild", &uiData.enableMonolithicClusterBuild);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip(
                "Use a single shader for compute cluster tiling and fill clusters.\n"
                "Instead of splitting the dispatches by surface type");

        bool accelBuildLoggingEnabled = m_app.GetAccelBuildLoggingEnabled();
        if (ImGui::Checkbox("AS Log", &accelBuildLoggingEnabled))
        {
            m_app.SetAccelBuildLoggingEnabled(accelBuildLoggingEnabled);
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Log accel build buffers (Syncs GPU, Slow!)");

        ImGui::SameLine();
        ImGui::Checkbox("Build AS", &uiData.forceRebuildAccelStruct);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip(
                "Re-tessellate and rebuild acceleration structures every frame.\n"
                "Disable to freeze tessellation and move the camera around.\n"
                "Animation will always force a rebuild.");

        if (!uiData.forceRebuildAccelStruct)
        {
            ImGui::SameLine();
            if (ImGui::Button("Build AS Once"))
            {
                m_app.RebuildAS();
            }
            if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
                ImGui::SetTooltip("Force rebuild-AS one time, useful for debug logging.");
        }

#if ENABLE_SHADER_DEBUG || ENABLE_PIXEL_PICK
        // The one pixel both the pick and the shader-debug predicate key off.
        int2& debugPixel = renderer.GetDebugPixel();
        ImGui::InputInt2("DebugPixel (Right-click)", debugPixel.data());
#endif

#if ENABLE_PIXEL_PICK
        const auto& pick = renderer.GetPixelPick();
        if (pick.valid)
        {
            if (pick.isClusterLod)
            {
                ImGui::TextDisabled("(cluster-LOD pick - shown in the Inspector)");
            }
            else
            {
                ImGui::TextDisabled("Instance:%u  Surface:%u  MatID:%u",
                    pick.instanceID, pick.surfaceID, pick.materialID);
                ImGui::TextDisabled("Material: %s", pick.name.c_str());
            }
        }
        else
        {
            ImGui::TextDisabled("(right-click a pixel to inspect material)");
        }
#endif

#if ENABLE_SHADER_DEBUG
        if (ImGui::InputInt3("Tessellator Debug (Surface, Cluster, Lane)", m_app.GetDebugSurfaceClusterLaneIndex().data()))
        {
            // Update renderer's debug surface index for highlighting
            renderer.SetDebugSurfaceIndex(m_app.GetDebugSurfaceClusterLaneIndex()[0]);
            m_app.RebuildAS();
        }
        if (ImGui::IsItemHovered() &&
            ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Set surface >=0 and lane >=0 to debug compute cluster tiling.\n"
                "Set cluster >=0 and lane >=0 to debug fill clusters.\n");
#endif
    }
}
#endif

void UserInterface::BuildRenderingSection(ImVec2 itemSize)
{
    auto& renderer = m_app.GetRenderer();

    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
    {
        bool showMicroTriangles = renderer.GetShowMicroTriangles();
        if (ImGui::Checkbox("Micro Triangles View", &showMicroTriangles))
            renderer.SetShowMicroTriangles(showMicroTriangles);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Toggle micro triangle visualization mode with a unique color per triangle id.");

        bool wireframe = renderer.GetWireframe();
        if (ImGui::Checkbox("Wireframe", &wireframe))
        {
            renderer.SetWireframe(wireframe);
        }
        if (ImGui::IsItemHovered() &&
            ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Display micro-triangles wireframe over the geometry.");

        ImGui::SameLine();
        bool displayZBuffer = renderer.GetDisplayZBuffer();
        if (ImGui::Checkbox("Show Occlusion Depth", &displayZBuffer))
        {
            renderer.SetDisplayZBuffer(displayZBuffer);
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Show Hi-Z Occlusion Buffer of static geometry used for reducing tessellation");

        ImGui::PushItemWidth(90.f);
        int maxFps = int(m_app.GetMaxFps());
        if (ImGui::InputInt("Max FPS", &maxFps, 1, 10))
            m_app.SetMaxFps(maxFps);
        ImGui::PopItemWidth();
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Limit rendering to N frames per second. 0 disables the cap.");
        ImGui::SameLine();
        bool vsync = m_app.GetVsyncEnabled();
        if (ImGui::Checkbox("VSync", &vsync))
            m_app.SetVsyncEnabled(vsync);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Synchronize presentation to the display refresh.");
        
        bool denoiserEnabled = m_app.GetEffectiveDenoiserMode() != DenoiserMode::None;
        if (!denoiserEnabled && renderer.GetEffectiveShadingMode() == ShadingMode::PT)
        {
            bool timeView = renderer.GetTimeView();
            if (ImGui::Checkbox("Heatmap", &timeView))
            {
                renderer.SetTimeView(timeView);
            }
            ImGui::SameLine();

            int spp = static_cast<int>(std::sqrt(renderer.GetSPP()) - 1);
            ImGui::PushItemWidth(65);
            if (ImGui::Combo("SPP", &spp, " 1x\0 4x\0 9x\0 16x\0 25x\0 36x\0 49x\0 64x\0 81x\0 100x\0"))
            {
                renderer.SetSPP((spp + 1) * (spp + 1));
            }
            ImGui::PopItemWidth();
            if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
                ImGui::SetTooltip("Samples Per Pixel.");
        }

        if (renderer.GetWireframe())
        {
            float wireframeThickness = renderer.GetWireframeThickness();
            if (ImGui::SliderFloat("Wireframe Thickness", &wireframeThickness, 0.f,
                10.f, "%.3f", ImGuiSliderFlags_Logarithmic))
            {
                renderer.SetWireframeThickness(wireframeThickness);
            }
        }
        
        ShadingMode shadingMode = renderer.GetShadingMode();
        if (ImGuiComboFromArray("Shading Mode", &shadingMode, kShadingModeNames))
        {
            renderer.SetShadingMode(shadingMode);
        }

        ColorMode colorMode = renderer.GetColorMode();
        if (ColorModeCombo("Color Mode", &colorMode, m_app.GetLiveColorModePaths()))
        {
            renderer.SetColorMode(colorMode);
        }

        ShadingMode effectiveShadingMode = renderer.GetEffectiveShadingMode();
        if (effectiveShadingMode == ShadingMode::PT)
        {
            int maxBounces = std::max(1, std::min(10, renderer.GetPTMaxBounces()));
            if (ImGui::InputInt("Max Bounces", &maxBounces, 1, 10))
            {
                renderer.SetPTMaxBounces(maxBounces);
            }

            if (!denoiserEnabled)
            {
                float fireflyMaxIntensity = renderer.GetFireflyMaxIntensity();
                if (ImGui::SliderFloat("Firefly Max Intensity", &fireflyMaxIntensity, 0.f, 10.f))
                {
                    renderer.SetFireflyMaxIntensity(fireflyMaxIntensity);
                }
            }

            float roughness = renderer.GetRoughnessOverride();
            if (ImGui::SliderFloat("Roughness Override", &roughness, 0.f, 1.f))
            {
                renderer.SetRoughnessOverride(roughness);
            }
            if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
                ImGui::SetTooltip(
                    "Overrides the roughness coefficient for all the materials\n"
                    "in the scene. Useful for debugging materials.\n");
        
            float exposure = renderer.GetExposure();
            ImGui::SetNextItemWidth(120.f);
            if (ImGui::SliderFloat("Exposure", &exposure, 0.f, 1000.f, "%.3f", ImGuiSliderFlags_Logarithmic))
                renderer.SetExposure(exposure);
            ImGui::SameLine();
            bool autoExposure = renderer.GetAutoExposure();
            if (ImGui::Checkbox("Adaptive##Exposure", &autoExposure))
                renderer.SetAutoExposure(autoExposure);
            if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
                ImGui::SetTooltip(
                    "Auto-exposure: donut histogram eye-adaptation drives exposure\n"
                    "from average scene luminance. The Exposure slider then acts as\n"
                    "an exposure-compensation multiplier on top of the auto value.");

            TonemapOperator tonemap = renderer.GetTonemapOperator();
            if (ImGuiComboFromArray("Tonemapping Operator", &tonemap, kToneMapOperatorNames))
            {
                renderer.SetTonemapOperator(tonemap);
            }
        }

        BuildUIEnvmap(itemSize);
    }
}

// Cluster-LoD budgets, drawn by the VRAM Budget window.  Every budget here sizes
// GPU pools at ClusterLodStreaming::Init; committing a change reallocates them
// and re-streams without reloading the scene.  The pool budgets need a live
// streaming config, but the render-cluster budget applies to --preload too, so
// the block only needs cluster LoD.
void UserInterface::BuildClusterLodBudgetControls()
{
    auto& renderer = m_app.GetRenderer();
    if (!renderer.GetClusterLodResources())
        return;

    const StagedBudgets live = LiveBudgets();

    rtxmg::StreamingConfig cfg;
    const bool hasStreamingBudgets = renderer.GetClusterLodStreamingConfig(cfg);

    // No pool may exceed a sane fraction of the card; on a --vram-mb run that
    // ceiling moves with the simulated card, which is the point of the flag.
    const int poolMaxMB = std::max(128, int((m_app.GetVramBytes() * 3 / 4) >> 20));

    if (hasStreamingBudgets)
    {
        ImGui::TextUnformatted("Streaming budgets");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Edits are staged; Apply reallocates the GPU pools and re-streams,\n"
                              "and the scene stays loaded. CLAS is floored to one allocator\n"
                              "sector (below that nothing streams).");

        // The persistent allocator quantizes its pool into sectors, so a pool
        // smaller than one sector has zero capacity and nothing streams.
        int clasMinMB = 1;
        if (cfg.usePersistentClasAllocator)
        {
            const uint64_t granuleBytes = uint64_t(128) << cfg.clasAllocatorGranularityShift;
            const uint64_t sectorBytes  = (uint64_t(1) << cfg.clasAllocatorSectorSizeShift) * 32ull * granuleBytes;
            clasMinMB = std::max(1, int((sectorBytes + (1ull << 20) - 1) >> 20));  // >= 1 sector, ceil MB
        }

        // Byte pools first, then the two slot/rate caps that bound them.
        // The geometry pool is sub-allocated in blocks, so its minimum tracks
        // the staged block size; a sub-block pool pins the residency watermark.
        StagedField("Geometry pool (MB)", m_staged.geometryPoolMB, live.geometryPoolMB,
                    128, 1024, std::max(1, m_staged.geometryBlockMB), poolMaxMB);
        StagedField("Geometry block (MB)", m_staged.geometryBlockMB, live.geometryBlockMB,
                    32, 128, 1, poolMaxMB);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Block size the geometry pool grows in (acquired/freed on demand).\n"
                              "The pool is floored to at least one block; the Memory tab's\n"
                              "Geometry high-water marker steps in this granularity.");
        StagedField("CLAS pool (MB)", m_staged.clasPoolMB, live.clasPoolMB,
                    128, 1024, clasMinMB, std::max(clasMinMB, poolMaxMB));
        // Only a cap: the pool is created lazily and grows in 16 MiB blocks, so a
        // scene that caches little never pays for the headroom.
        ImGui::BeginDisabled(!cfg.allowBlasCaching);
        // Capped below 4 GB: the pool size becomes the MOVE op's uint32 maxBytes.
        StagedField("Cached BLAS pool (MB)", m_staged.blasCachingPoolMB, live.blasCachingPoolMB,
                    64, 256, 16, std::min(poolMaxMB, 4095));
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip(cfg.allowBlasCaching
                ? "Ceiling on the BLAS-caching pool. Unlike the two above it is not\n"
                  "pre-allocated: blocks are acquired on demand, so raising it costs\n"
                  "nothing until geometry actually caches. Hitting it stops new\n"
                  "geometry from caching and those instances rebuild every frame."
                : "Needs BLAS caching (--no-blascaching is set).");
        StagedField("Max resident groups", m_staged.maxResidentGroups, live.maxResidentGroups,
                    1024, 16384, 1, INT_MAX);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Slot cap on resident groups, independent of the byte pools: when\n"
                              "it is hit streaming wedges with the pools still under budget.\n"
                              "Sizes fixed tables at ~800 B/group - 44 B for the group plus\n"
                              "24 B for each of its 32 cluster slots - so the 128K default is\n"
                              "~100 MB of Cluster LOD Metadata before any geometry streams.");
        StagedField("Max loads / frame", m_staged.maxLoadsPerFrame, live.maxLoadsPerFrame,
                    128, 1024, 1, INT_MAX);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Group loads consumed per frame. Higher converges faster after a\n"
                              "camera cut, but the per-frame CLAS-build staging scales with it:\n"
                              "~600 B per load-cluster (32 per group, so ~19 KB per load) plus\n"
                              "the driver-sized Implicit build scratch, which is the larger of\n"
                              "the two. Both land in Cluster LOD Metadata.");
    }

    // Per-frame render-cluster budget (1<<N); commits through the same
    // ApplyStreamingBudgets path, which reallocates the pass buffers.
    StagedField("Render cluster bits", m_staged.renderClusterBits, live.renderClusterBits,
                1, 1, 16, 25);
    if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
        ImGui::SetTooltip("Per-frame render-cluster budget = 1<<N clusters (clamp 16..25,\n"
                          "default 20 = 1M).  Raise if \"Render cluster budget exceeded\"\n"
                          "flickers; costs ~16 B/cluster of index/VA arrays (20 = 1M ~=\n"
                          "16 MB, under Cluster LOD Metadata) and each bit doubles it.\n"
                          "The CLAS data pool is budgeted separately.");

    if (hasStreamingBudgets)
    {
        // Not a budget edit: it re-streams with what is already applied.
        if (ImGui::Button("Reset streaming state"))
            m_app.ApplyStreamingBudgets();
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Evict everything and re-stream the current view with the\n"
                              "applied budgets (same path Apply takes).");
    }
}

void UserInterface::BuildClusterLodSection()
{
    auto& renderer = m_app.GetRenderer();

    if (BeginDisableableSection("Cluster LOD", m_app.GetScene().HasClusterLod(), kNoClusterLod))
    {
        if (ImGui::Button("Bake Config"))
            m_app.GetUIData().showBakeConfigWindow = true;
        ImGui::Spacing();

        // ---- Shading ------------------------------------------------------------
        ImGui::SeparatorText("Shading");

        bool clusterLodVertexNormals = m_app.GetClusterLodVertexNormalsEnabled();
        // "##clusterlod": distinct ID from the Tessellation section's "Vertex
        // Normals", which is a separate toggle driving the subd path.
        if (ImGui::Checkbox("Vertex Normals##clusterlod", &clusterLodVertexNormals))
            m_app.SetClusterLodVertexNormalsEnabled(clusterLodVertexNormals);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Shade cluster-LoD hits with baked per-vertex normals\n"
                              "(off = facet/flat shading).  While off, the streamed\n"
                              "group blobs omit their normal words entirely (smaller\n"
                              "geometry pool); toggling re-streams the current view.");

        {
            const bool mapsLoaded = m_app.GetNormalMapsEnabled();
            const bool canShade   = mapsLoaded && clusterLodVertexNormals;
            // Masked, not raw: a ticked box under a disabled control claims a
            // shading path that is not running.
            bool normalMapShading = canShade && m_app.GetNormalMapShading();
            ImGui::BeginDisabled(!canShade);
            if (ImGui::Checkbox("Shading Normals##shading", &normalMapShading))
                m_app.SetNormalMapShading(normalMapShading);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
                ImGui::SetTooltip(!clusterLodVertexNormals
                    ? "Needs Vertex Normals: a normal map perturbs the interpolated\n"
                      "vertex normal, so it has nothing to sit on while that is off."
                    : !mapsLoaded
                    ? "Normal maps are not loaded."
                    : "Sample the material's normal map and perturb the shading normal\n"
                      "with a tangent frame derived from the hit triangle's du/dv.\n"
                      "Toggles live -- it only picks the path-tracer permutation.");

            ImGui::SameLine();
            bool normalMaps = mapsLoaded;
            if (ImGui::Checkbox("Normal Maps##shading", &normalMaps))
            {
                SyncStagedBudgets();
                m_staged.normalMaps = normalMaps;
                ImGui::OpenPopup("##NormalMapsReloadConfirm");
            }
            if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
                ImGui::SetTooltip("Load or unload all normal-map textures.\n"
                                  "Reloads the scene (re-reads every texture from disk).");

            ImGui::SetNextWindowPos(LayoutCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
            if (ImGui::BeginPopupModal("##NormalMapsReloadConfirm", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
            {
                ImGui::BulletText("Load normal maps: %s -> %s",
                                  mapsLoaded ? "on" : "off",
                                  m_staged.normalMaps ? "on" : "off");
                ImGui::Spacing();
                BuildReloadWarningText();
                ImGui::Spacing();
                if (ImGui::Button("Reload", ImVec2(90, 0)))
                {
                    ApplyStagedPoolBudgets();
                    m_app.ApplyTextureSettingsAndReload(m_staged.textureBudgetMB, m_staged.normalMaps);
                    m_stagedValid = false;
                    ImGui::CloseCurrentPopup();
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel", ImVec2(90, 0)))
                    ImGui::CloseCurrentPopup();
                ImGui::EndPopup();
            }
        }

        // ---- LOD / Culling ------------------------------------------------------
        ImGui::SeparatorText("LOD / Culling");

        {
            bool adaptiveLodError = renderer.GetAdaptiveLodError();
            float lodPixelError = renderer.GetLodPixelError();
            ImGui::SetNextItemWidth(120.f);
            if (ImGui::SliderFloat("LOD Pixel Error", &lodPixelError, 0.1f, 16.0f, "%.3f", ImGuiSliderFlags_Logarithmic))
                renderer.SetLodPixelError(std::max(0.001f, lodPixelError));
            if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
                ImGui::SetTooltip(
                    "Target screen-space error in pixels at which cluster-LoD\n"
                    "traversal stops descending.\n\n"
                    "Smaller values = finer LoDs (more detail, higher cost).\n"
                    "Larger values = coarser LoDs (less detail, blockier silhouette).\n\n"
                    "Threshold = 2*tan(fov/2) * lodPixelError / viewportHeight.");
            ImGui::SameLine();
            if (ImGui::Checkbox("Adaptive##lpe", &adaptiveLodError))
                renderer.SetAdaptiveLodError(adaptiveLodError);
            if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
                ImGui::SetTooltip("Raise the effective LoD Pixel Error while the streaming pools run\n"
                                  "hot (>85%% load: +2%%/frame; <70%%: slow recovery), so the desired\n"
                                  "resident set fits the budgets instead of churning.  Never goes\n"
                                  "below the slider value.");
            if (adaptiveLodError)
            {
                ImGui::SameLine();
                ImGui::TextDisabled("(%.3f)", renderer.GetEffectiveLodPixelError());
            }
        }

        // Map the three runtime flags to one mode selector.
        int cullMode = 0;  // 0 off, 1 soft, 2 hard, 3 hard+invisible
        if (renderer.GetUseCulling())
            cullMode = renderer.GetUseHardCull() ? (renderer.GetHardCullForcesInvisible() ? 3 : 2) : 1;
        const char* cullItems[] = { "Off", "Soft (coarsen)", "Hard (low-detail)", "Hard (invisible)" };
        if (ImGui::Combo("Culling", &cullMode, cullItems, IM_ARRAYSIZE(cullItems)))
        {
            renderer.SetUseCulling(cullMode != 0);
            renderer.SetUseHardCull(cullMode >= 2);
            renderer.SetHardCullForcesInvisible(cullMode == 3);
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Offscreen or occluded instances (current frame's view):\n"
                              "  Soft     - kept in TLAS, biased coarser via culled error scale.\n"
                              "  Hard     - not traversed; fall back to low-detail BLAS.\n"
                              "  Invisible- BLAS nulled; dropped from the TLAS entirely.");

        // HiZ occlusion runs inside the cull, so it needs culling on.
        ImGui::BeginDisabled(cullMode == 0);
        bool hizOcclusion = renderer.GetUseHizOcclusion();
        if (ImGui::Checkbox("HiZ occlusion", &hizOcclusion))
            renderer.SetUseHizOcclusion(hizOcclusion);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Occlusion-cull instances/nodes against the previous frame's\n"
                              "depth pyramid (soft: coarsen, hard: skip). The pyramid lags the\n"
                              "camera by a frame - turn OFF to check whether streaming flicker\n"
                              "(e.g. while dollying) comes from the occlusion test.");
        ImGui::EndDisabled();

        ImGui::BeginDisabled(cullMode != 1);  // culledErrorScale only affects soft cull
        float culledErrorScale = renderer.GetCulledErrorScale();
        if (ImGui::SliderFloat("Culled error scale", &culledErrorScale, 1.0f, 16.0f, "%.2f"))
            renderer.SetCulledErrorScale(std::max(1.0f, culledErrorScale));
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("LoD coarsening bias applied to soft-culled (off-screen) instances.\n"
                              "Higher = coarser off-screen LoD = cheaper.");
        ImGui::EndDisabled();

        // ---- BLAS reuse -------------------------------------------------------
        // Live-safe to toggle: the traversal pass builds both permutations at Init
        // and merging is a runtime-gated constant.
        ImGui::SeparatorText("BLAS Reuse");

        bool sharing = renderer.GetUseBlasSharing();
        if (ImGui::Checkbox("Sharing##blas", &sharing))
        {
            renderer.SetUseBlasSharing(sharing);
            if (!sharing)
            {
                // merging + caching are both subsets of sharing
                renderer.SetUseBlasMerging(false);
                renderer.SetUseBlasCaching(false);
            }
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Distant low-detail instances reuse a canonical instance's BLAS\n"
                              "instead of each building their own (~3x fewer traversal builds).");
        ImGui::SameLine();
        ImGui::BeginDisabled(!renderer.GetUseBlasSharing());
        int sharingLevels = int(renderer.GetBlasSharingEnabledLevels());
        ImGui::SetNextItemWidth(80.f);
        if (ImGui::SliderInt("Shared Tail Levels", &sharingLevels, 0, 32))
            renderer.SetBlasSharingEnabledLevels(uint32_t(std::max(0, sharingLevels)));
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Number of coarse tail LoD levels eligible for sharing.\n"
                              "Higher = more aggressive reuse.");
        ImGui::EndDisabled();

        // Live-toggleable: the BLAS-pass scratch is always sized for the streaming
        // path and the cached pool is allocated lazily on first use.
        const bool cachingAvailable = renderer.GetUseBlasSharing() && renderer.GetUseStreaming();
        ImGui::BeginDisabled(!cachingAvailable);
        bool caching = renderer.GetUseBlasCaching();
        if (ImGui::Checkbox("Caching##blas", &caching))
            renderer.SetUseBlasCaching(caching);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Build a per-geometry discrete-LoD BLAS once into a persistent pool\n"
                              "and reuse it across frames for fully-resident coarse LoDs.\n\n"
                              "Requires BLAS Sharing + streaming.");
        ImGui::SameLine();
        ImGui::BeginDisabled(!renderer.GetUseBlasCaching());
        int cachingLevels = int(renderer.GetBlasCachingEnabledLevels());
        ImGui::SetNextItemWidth(80.f);
        if (ImGui::SliderInt("Cached Tail Levels", &cachingLevels, 0, 32))
            renderer.SetBlasCachingEnabledLevels(uint32_t(std::max(0, cachingLevels)));
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Number of coarse tail LoD levels eligible for caching.");
        ImGui::EndDisabled();
        ImGui::EndDisabled();

        const bool mergingAvailable = renderer.GetUseBlasSharing() && renderer.GetUseStreaming();
        ImGui::BeginDisabled(!mergingAvailable);
        bool merging = renderer.GetUseBlasMerging();
        if (ImGui::Checkbox("Merging##blas", &merging))
            renderer.SetUseBlasMerging(merging);
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Collapse all high-detail per-instance builds of a geometry into\n"
                              "one merged BLAS per geometry (big renderedClusters drop).\n\n"
                              "Requires BLAS Sharing + streaming (unavailable under --preload).");

    }

}

// One wording for both confirmations, so they cannot drift apart.
void UserInterface::BuildReloadWarningText()
{
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.75f, 0.f, 1.f));
    ImGui::TextUnformatted("This decides what is read off disk, so it takes effect only on a");
    ImGui::TextUnformatted("full scene reload: every texture is re-read and the geometry is");
    ImGui::TextUnformatted("re-loaded from the shard cache.  On a large scene this takes minutes.");
    ImGui::PopStyleColor();
}

UserInterface::StagedBudgets UserInterface::LiveBudgets()
{
    StagedBudgets b;
    b.textureBudgetMB = m_app.GetTextureBudgetMB();
    b.normalMaps      = m_app.GetNormalMapsEnabled();

    auto& renderer = m_app.GetRenderer();
    rtxmg::StreamingConfig cfg;
    if (renderer.GetClusterLodStreamingConfig(cfg))
    {
        b.maxResidentGroups = int(cfg.maxGroups);
        b.geometryPoolMB    = int(cfg.maxGeometryMegaBytes);
        b.geometryBlockMB   = int(cfg.geometryBlockMegaBytes);
        b.clasPoolMB        = int(cfg.maxClasMegaBytes);
        b.blasCachingPoolMB = int(cfg.maxBlasCachingMegaBytes);
        b.maxLoadsPerFrame  = int(cfg.maxPerFrameLoadRequests);
    }
    b.renderClusterBits = int(renderer.GetRenderClusterBits());

    const TessellatorConfig::MemorySettings ms = m_app.GetTessMemSettings();
    b.tessMaxKClusters = int(ms.maxClusters >> 10u);
    b.tessVertexMB     = int(ms.vertexBufferBytes >> 20ull);
    b.tessClasMB       = int(ms.clasBufferBytes >> 20ull);
    return b;
}

void UserInterface::SyncStagedBudgets()
{
    m_staged      = LiveBudgets();
    m_stagedValid = true;
}

// Everything the Apply button commits that does not need a scene reload.
void UserInterface::ApplyStagedPoolBudgets()
{
    auto& renderer = m_app.GetRenderer();

    if (renderer.GetClusterLodResources())
    {
        renderer.SetMaxResidentGroups(uint32_t(std::max(1, m_staged.maxResidentGroups)));
        renderer.SetMaxGeometryMB(uint32_t(std::max(1, m_staged.geometryPoolMB)));
        renderer.SetGeometryBlockMB(uint32_t(std::max(1, m_staged.geometryBlockMB)));
        renderer.SetMaxClasMB(uint32_t(std::max(1, m_staged.clasPoolMB)));
        renderer.SetMaxBlasCachingMB(uint32_t(std::max(1, m_staged.blasCachingPoolMB)));
        renderer.SetMaxFrameLoadRequests(uint32_t(std::max(1, m_staged.maxLoadsPerFrame)));
        renderer.SetRenderClusterBits(std::clamp(uint32_t(m_staged.renderClusterBits), 16u, 25u));
        m_app.ApplyStreamingBudgets();
    }

    TessellatorConfig::MemorySettings ms = m_app.GetTessMemSettings();
    ms.maxClusters       = std::min(uint32_t(std::max(m_staged.tessMaxKClusters, 64)) << 10u,
                                    kMaxApiClusterCount);
    ms.vertexBufferBytes = size_t(std::max(m_staged.tessVertexMB, 128)) << 20ull;
    ms.clasBufferBytes   = size_t(std::max(m_staged.tessClasMB, 128)) << 20ull;
    m_app.SetTessMemSettings(ms);
}

// What the Pending Budget bar totals for `s`: every budget at its ceiling, plus
// the buckets that have no budget to edit.
uint64_t UserInterface::ProposedBytes(const StagedBudgets& s)
{
    const stats::VramBreakdown&   vb = stats::vramBreakdown;
    const stats::TextureMemStats& tm = stats::memUsageSamplers.textures;

    const uint64_t texCap = uint64_t(std::max(0, s.textureBudgetMB)) << 20;
    const uint64_t textures =
        tm.budgetableCount == 0
            ? vb.textures
            : (texCap ? std::min(texCap, tm.budgetableFullBytes) : tm.budgetableFullBytes)
              + tm.loadedOtherBytes;

    const bool hasClod = m_app.GetRenderer().GetClusterLodResources() != nullptr;
    const bool hasTess = !m_app.GetScene().GetSubdMeshes().empty();

    const uint64_t clodGeo   = hasClod ? uint64_t(std::max(0, s.geometryPoolMB))    << 20 : 0;
    const uint64_t clodClas  = hasClod ? uint64_t(std::max(0, s.clasPoolMB))        << 20 : 0;
    const uint64_t cachedBlas= hasClod ? uint64_t(std::max(0, s.blasCachingPoolMB)) << 20 : 0;
    const uint64_t tessVert  = hasTess ? uint64_t(std::max(0, s.tessVertexMB))      << 20 : 0;
    const uint64_t tessClas  = hasTess ? uint64_t(std::max(0, s.tessClasMB))        << 20 : 0;

    return textures + clodGeo + clodClas + cachedBlas + vb.clodMetadata
         + tessVert + tessClas + vb.tessClusterData
         + vb.blas + vb.renderTargets + vb.envmap + vb.Unaccounted();
}

// A staged field: highlighted while it differs from the live value, and unable
// to grow past the point where the proposed plot would fill 90% of the card.
void UserInterface::StagedField(const char* label, int& staged, int live, int step,
                                int stepFast, int minVal, int maxVal)
{
    const bool dirty = staged != live;
    if (dirty)
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.16f, 0.31f, 0.42f, 1.f));

    const int before = staged;
    ImGui::InputInt(label, &staged, step, stepFast);
    staged = std::clamp(staged, minVal, maxVal);

    if (staged > before)
    {
        const uint64_t ceiling = uint64_t(double(m_app.GetVramBytes()) * 0.90);
        if (ProposedBytes(m_staged) > ceiling)
            staged = before;  // refuse the growth; shrinking is always allowed
    }

    if (dirty)
        ImGui::PopStyleColor();
}

// Tessellation memory budgets, drawn by the VRAM Budget window.  A change is
// picked up by the next accel-structure update; the scene stays loaded.
void UserInterface::BuildTessBudgetControls()
{
    const StagedBudgets live = LiveBudgets();
    const auto& stats = GetApp().m_BuildStats;

    // Neither tess buffer may exceed a sane fraction of the card; --vram-mb
    // moves that ceiling so the clamp is testable on a bigger GPU.
    const int tessMaxMB = std::max(128, int((m_app.GetVramBytes() * 3 / 4) >> 20));

    // A budget the scene has already outgrown flashes red, which outranks the
    // staged-edit highlight because it is a live problem, not a pending one.
    const bool flash = fmodf(float(ImGui::GetTime()), 1.0f) < 0.5f;
    const bool overClusters = flash && stats.desired.m_numClusters > stats.allocated.m_numClusters;
    const bool overVertex   = flash && (stats.desired.m_vertexBufferSize > stats.allocated.m_vertexBufferSize ||
                                        stats.desired.m_vertexNormalsBufferSize > stats.allocated.m_vertexNormalsBufferSize);
    const bool overClas     = flash && stats.desired.m_clasSize > stats.allocated.m_clasSize;

    const ImVec4 kOverBudget(0.5f, 0.0f, 0.0f, 1.0f);

    auto tessField = [&](const char* label, int& staged, int live, int step, int stepFast,
                         int minVal, int maxVal, bool over, const char* tip)
    {
        if (over)
            ImGui::PushStyleColor(ImGuiCol_FrameBg, kOverBudget);
        StagedField(label, staged, live, step, stepFast, minVal, maxVal);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("%s", tip);
        if (over)
            ImGui::PopStyleColor();
    };

    tessField("Max Clusters (K)", m_staged.tessMaxKClusters, live.tessMaxKClusters, 64, 256,
              64, int(kMaxApiClusterCount >> 10u), overClusters,
              "Max clusters in a scene which affects the size of cluster data buffer and BLAS memory");
    tessField("Vertex Memory (MB)", m_staged.tessVertexMB, live.tessVertexMB, 128, 512,
              128, tessMaxMB, overVertex,
              "Max memory in MB allocated for tessellated vertices (positions + normals when enabled)");
    tessField("CLAS Memory (MB)", m_staged.tessClasMB, live.tessClasMB, 128, 512,
              128, tessMaxMB, overClas,
              "Max memory in megabytes allocated for cluster acceleration structures CLAS");
}

void UserInterface::BuildTessellationSection()
{
    if (BeginDisableableSection("Cluster Tess", !m_app.GetScene().GetSubdMeshes().empty(), kNoTess))
    {
        int   clusterPattern = static_cast<int>(m_app.GetClusterTessellationPattern());
        float comboBoxWidth = ImGui::GetTextLineHeightWithSpacing() + ImGui::CalcTextSize("Slanted  ").x
            + ImGui::GetStyle().FramePadding.x * 2.0f;
        ImGui::SetNextItemWidth(comboBoxWidth);
        if (ImGui::Combo("Tess Pattern", &clusterPattern, "Regular\0Slanted\0"))
        {
            m_app.SetClusterTessellationPattern(ClusterTessPattern(clusterPattern));
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip(
                "Toggles the 'slanted' grid pattern on. 'Slanted' grids\n"
                "allow for a smoother transition when the number of edge segments on\n"
                "opposite sides of a quad don't match.\n\n");

        ImGui::SameLine();
        bool tessVertexNormals = m_app.GetVertexNormalsEnabled();
        // "##tess": distinct ID from the Cluster LODs section's "Vertex Normals",
        // which is a separate toggle driving the cluster-LoD path.
        if (ImGui::Checkbox("Vertex Normals##tess", &tessVertexNormals))
            m_app.SetVertexNormalsEnabled(tessVertexNormals);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Compute + shade tessellated (subd) geometry with interpolated\n"
                              "per-vertex normals (off = facet/flat shading).");

        bool enableFrustumVisibility = m_app.GetFrustumVisibilityEnabled();
        if (ImGui::Checkbox("Frustum", &enableFrustumVisibility))
            m_app.SetFrustumVisibilityEnabled(enableFrustumVisibility);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Offscreen geometry uses coarse tessellation rates");

        ImGui::SameLine();
        bool enableHiZVisibility = m_app.GetHiZVisibilityEnabled();
        if (ImGui::Checkbox("HiZ", &enableHiZVisibility))
            m_app.SetHiZVisibilityEnabled(enableHiZVisibility);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Dynamic geometry occluded by static geometry uses coarse tessellation rate");

        TessellatorConfig::VisibilityMode visMode = m_app.GetTessellatorVisibilityMode();
        if (visMode == TessellatorConfig::VisibilityMode::VIS_LIMIT_EDGES)
        {
            ImGui::SameLine();
            bool enableBackFaceVisibility = m_app.GetBackfaceVisibilityEnabled();
            if (ImGui::Checkbox("Backface", &enableBackFaceVisibility))
                m_app.SetBackfaceVisibilityEnabled(enableBackFaceVisibility);
            if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
                ImGui::SetTooltip("Back faces use coarse tessellation rate");
        }

        float tessRates[] = { m_app.GetFineTessellationRate(), m_app.GetCoarseTessellationRate() };
        if (ImGui::SliderFloat2("Fine | Coarse Tess Rate", tessRates, 0.001f, 2.f))
        {
            if (tessRates[0] > 0.0f)
            {
                m_app.SetFineTessellationRate(tessRates[0]);
            }

            if (tessRates[1] > 0.0f)
            {
                m_app.SetCoarseTessellationRate(tessRates[1]);
            }
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Tessellation rates for metric:\n"
                "- Fine: Affects primary visible ray geometry\n"
                "- Coarse: Affects \"culled\" geometry: offscreen, backfacing, occluded\n");

        TessellatorConfig::AdaptiveTessellationMode tessMode = m_app.GetAdaptiveTessellationMode();
        if (ImGuiComboFromArray("Tessellation Metric", &tessMode, kAdaptiveTessellationModeNames))
        {
            m_app.SetAdaptiveTessellationMode(tessMode);
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip(
                "Tessellation metrics:\n\n"
                "- Uniform: uniform tessellation factors (all clusters have the\n"
                "  same number of triangles).\n\n"
                "- World space edge length: tessellation factors are derived from\n"
                "  the length of the control cage edges in world space (independent\n"
                "  the camera position).\n\n"
                "- Spherical projection: tessellation factors are derived from the\n"
                "  length of the control cage edges scaled by their distance to the\n"
                "  camera location.\n");

        
        if (ImGuiComboFromArray("Visibility Mode", &visMode, kVisibilityModeNames))
        {
            m_app.SetTessellatorVisibilityMode(visMode);
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip(
                "Tessellation visibility predicates:\n\n"
                "- Surface 1-Ring: uses the 1-ring control cage of a surface to derive\n"
                "  visibility for an entire surface\n\n"
                "- Limit edge: generates visibility predicate for each limit edge a\n"
                "  surface only.\n");
       
        int isolationLevel = m_app.GetGlobalIsolationLevel();
        if (ImGui::SliderInt("Global Isolation Level", &isolationLevel, 1, 6))
        {
            m_app.SetGlobalIsolationLevel(uint32_t(isolationLevel));
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Global isolation level for all meshes.\n"
                "- Needs to be >= the highest finite sharpness, but the default of 6 is sufficient.\n"
                "- Finite sharpness is any sharpness < 10.0f.\n"
                "- Infinite sharpness = 10.0f has an optimization that doesn't require isolation\n");

        float displacementScale = m_app.GetDisplacementScale();
        if (ImGui::SliderFloat("Displacement Scale", &displacementScale, 0.0f, 3.0f))
        {
            m_app.SetDisplacementScale(displacementScale);
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Scaling factor for displacement maps");
    }
}

#if DONUT_WITH_STREAMLINE
void UserInterface::BuildDenoiserSection()
{
    UIData& uiData = GetApp().GetUIData();

    auto& renderer = m_app.GetRenderer();

    if (!renderer.GetShowMicroTriangles() && ImGui::CollapsingHeader("Denoiser and Upscaling", ImGuiTreeNodeFlags_DefaultOpen))
    {
        using StreamlineInterface = donut::app::StreamlineInterface;

        DenoiserMode denoiserMode = m_app.GetDenoiserMode();
#if ENABLE_DLSS_SR
        if (ImGui::Combo("Denoiser Mode", (int*)&denoiserMode, "None\0DLSS-SR\0DLSS-RR\0"))
        {
            m_app.SetDenoiserMode(denoiserMode);
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Set Denoiser mode");
#else
        bool isDlssEnabled = denoiserMode == DenoiserMode::DlssRr;
        if (ImGui::Checkbox("Enable DLSS-RR", &isDlssEnabled))
        {
            m_app.SetDenoiserMode(isDlssEnabled ? DenoiserMode::DlssRr : DenoiserMode::None);
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Enable DLSS-RR for upscale and denoising");
#endif

        if (denoiserMode != DenoiserMode::None)
        {
            if (denoiserMode == DenoiserMode::DlssSr ||
                denoiserMode == DenoiserMode::DlssRr)
            {
                const std::array<std::pair<StreamlineInterface::DLSSMode, const char*>, 5> kVisibleDlssModes = { {
                    {StreamlineInterface::DLSSMode::eUltraPerformance, "Ultra-Performance"},
                    {StreamlineInterface::DLSSMode::eMaxPerformance, "Performance"},
                    {StreamlineInterface::DLSSMode::eBalanced, "Balanced"},
                    {StreamlineInterface::DLSSMode::eMaxQuality, "Quality"},
                    {StreamlineInterface::DLSSMode::eDLAA, "DLAA"}
                } };

                auto iter = std::find_if(kVisibleDlssModes.begin(), kVisibleDlssModes.end(), [&uiData](auto& m) { return m.first == uiData.dlssMode; });
                if (iter == kVisibleDlssModes.end())
                {
                    // Reset to eMaxQuality if we can't find the option
                    uiData.dlssMode = StreamlineInterface::DLSSMode::eMaxQuality;
                    iter = std::find_if(kVisibleDlssModes.begin(), kVisibleDlssModes.end(), [&uiData](auto& m) { return m.first == uiData.dlssMode; });
                }

                if (ImGui::BeginCombo("DLSS Mode", iter->second))
                {
                    for (const auto& mode : kVisibleDlssModes)
                    {
                        bool isSelected = (mode.first == uiData.dlssMode);
                        if (ImGui::Selectable(mode.second, isSelected))
                        {
                            uiData.dlssMode = mode.first;
                        }
                        if (isSelected) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

#if ENABLE_DLSS_DEV_FEATURE
                if (uiData.dlssMode != StreamlineInterface::DLSSMode::eUltraQuality &&
                    uiData.dlssMode != StreamlineInterface::DLSSMode::eOff)
                {
                    using SrPreset = StreamlineInterface::DLSSPreset;
                    using RrPreset = StreamlineInterface::DLSSRRPreset;
                    static constexpr std::pair<const char*, SrPreset> kDlssPresets[] = {
                        { "Default",  SrPreset::eDefault },
                        { "Preset J", SrPreset::ePresetJ },
                        { "Preset K", SrPreset::ePresetK },
                    };
                    static constexpr std::pair<const char*, RrPreset> kDlssRRPresets[] = {
                        { "Default",  RrPreset::eDefault },
                        { "Preset D", RrPreset::ePresetD },
                        { "Preset E", RrPreset::ePresetE },
                        { "Preset F", RrPreset::ePresetF },
                    };

                    auto presetCombo = [](const char* label, const auto& presets, auto& current)
                    {
                        const char* preview = presets[0].first;
                        for (const auto& [name, value] : presets)
                            if (value == current) preview = name;

                        if (ImGui::BeginCombo(label, preview))
                        {
                            for (const auto& [name, value] : presets)
                            {
                                bool isSelected = value == current;
                                if (ImGui::Selectable(name, isSelected)) current = value;
                                if (isSelected) ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }
                    };

                    if (denoiserMode == DenoiserMode::DlssSr)
                        presetCombo("DLSS SR Preset", kDlssPresets, uiData.dlssPreset);
                    else
                        presetCombo("DLSS RR Preset", kDlssRRPresets, uiData.dlssRRPreset);
                }

                ImGui::Checkbox("Overide LOD Bias", &uiData.dlssUseLodBiasOverride);
                if (uiData.dlssUseLodBiasOverride)
                {
                    ImGui::SameLine();
                    ImGui::SliderFloat("", &uiData.dlssLodBiasOverride, -2, 2);
                }
#endif
            }

            if (ImGui::BeginCombo("Output", renderer.GetOutputLabel(renderer.GetOutputIndex()),
                ImGuiComboFlags_HeightLarge))
            {
                for (uint32_t outputIndex = uint32_t(RTXMGRenderer::Output::Accumulation);
                    outputIndex < uint32_t(RTXMGRenderer::Output::Count);
                    outputIndex++)
                {
                    bool isSelected = outputIndex == uint32_t(renderer.GetOutputIndex());

                    if (ImGui::Selectable(renderer.GetOutputLabel(RTXMGRenderer::Output(outputIndex)), isSelected))
                    {
                        renderer.SetOutputIndex(RTXMGRenderer::Output(outputIndex));
                        renderer.ResetSubframes();
                    }
                    if (isSelected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            float denoiserSeparator = renderer.GetDenoiserSeparator();
            if (ImGui::SliderFloat("Output | Denoised", &denoiserSeparator, 0.0f, 1.0f, "%.2f"))
            {
                renderer.SetDenoiserSeparator(denoiserSeparator);
            }

            MvecDisplacement mvecDisplacement = renderer.GetMVecDisplacement();
            if (ImGuiComboFromArray("Motion Vectors", &mvecDisplacement, kMvecDisplacementNames))
            {
                renderer.SetMvecDisplacement(mvecDisplacement);
            }
            if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            {
                ImGui::SetTooltip(
                    "Motion Vectors Calculation Mode:\n\n"
                    "- From Subd Eval: Compute displacement using the delta between\n"
                    "  gbuffer hit point and current frame limit surface.\n"
                    "  Expensive since it re-evalutes limit surface again, but compensates for tess rates\n\n"
                    "- From Material: Resample displacement from texture and apply to prev frame limit surface\n"
                    "  If tess rates vary then there can be a mismatch with the current frame hit point.\n");
            }
        }
    }
}
#endif

void UserInterface::UpdateProfilerStats(int2 screenLayoutSize)
{
    // FPS over a ~0.5s window rather than the current frame's 1/dt, which
    // flickers by tens of fps frame to frame.
    {
        constexpr float kFpsRefreshMs = 500.f;
        m_fpsWindowMs += m_app.GetCPUFrameTime();
        ++m_fpsWindowFrames;
        if (m_fpsWindowMs >= kFpsRefreshMs)
        {
            m_fpsDisplay      = (int)std::lround(1000.f * m_fpsWindowFrames / m_fpsWindowMs);
            m_fpsWindowMs     = 0.f;
            m_fpsWindowFrames = 0;
        }
        m_profiler.fps = m_fpsDisplay;
    }
    m_profiler.desiredTris = stats::clusterAccelSamplers.numTriangles.latest;
    m_profiler.allocatedTris = stats::clusterAccelSamplers.numTriangles.max;

    // Per-frame TLAS unique / total triangles, only valid where the 64-bit-atomic
    // readback they are accumulated with is supported.
    {
        auto& clusterLodRenderer = m_app.GetRenderer();
        if (clusterLodRenderer.GetClusterLodResources() && clusterLodRenderer.GetAtomicInt64OnHeapSupported())
        {
            const auto& clc = clusterLodRenderer.GetClusterLodCounters();
            const uint32_t tessTris = stats::clusterAccelSamplers.numTriangles.latest;
            m_profiler.clusterLodTrisValid  = true;
            m_profiler.clusterLodUniqueTris = clc.uniqueTriangles + tessTris;
            m_profiler.clusterLodTotalTris  = clc.totalTriangles  + tessTris;
        }
        else
        {
            m_profiler.clusterLodTrisValid = false;
        }
    }
    m_profiler.desiredClusters = stats::clusterAccelSamplers.numClusters.latest;
    m_profiler.allocatedClusters = stats::clusterAccelSamplers.numClusters.max;

    m_profiler.controllerWindow = {
        .pos = ImVec2(float(screenLayoutSize.x) - 10.f, float(screenLayoutSize.y) - 10.f),
        .pivot = ImVec2(1.f, 1.f),
        .size = ImVec2(115, 0)
    };

    m_profiler.profilerWindow = {
        .pos = ImVec2(float(screenLayoutSize.x) - 10.f, 10.f),
        .pivot = ImVec2(1.f, 0.f),
        .size = ImVec2(800.f, 450.f),
        .screenLayoutSize = ImVec2(float(screenLayoutSize.x), float(screenLayoutSize.y))
    };
}

void UserInterface::BuildUIMain(int2 screenLayoutSize)
{
    RegisterMidiBindings();

    ImVec2 itemSize = ImGui::GetItemRectSize();

    BuildTopBar(itemSize);
    BuildSettingsWindow(screenLayoutSize, itemSize);

    UpdateProfilerStats(screenLayoutSize);

    // Subdivision Evaluator last (after Streaming); each tab self-hides when its
    // data is irrelevant (TabEnabled()).
    m_profiler.BuildUI<stats::FrameSamplers, stats::ClusterTessTab, stats::ClusterLodTab, stats::MemUsageSamplers, stats::StreamingSamplers, stats::EvaluatorSamplers>(m_iconicFont, m_implot,
        stats::frameSamplers, stats::clusterTessTab, stats::clusterLodTab, stats::memUsageSamplers, stats::streamingSamplers, stats::evaluatorSamplers);

    if (stats::evaluatorSamplers.m_topologyQualityButtonPressed)
    {
        m_app.GetRenderer().SetColorMode(ColorMode::COLOR_BY_TOPOLOGY);
    }

    if (stats::evaluatorSamplers.m_openInspectorRequested)
    {
        m_showInspector = true;
        stats::evaluatorSamplers.m_openInspectorRequested = false;
    }

    float profilerWidth = m_profiler.controllerWindow.size.x;
    float timelineWidth = float(screenLayoutSize.x) - 30.f - profilerWidth;

    BuildUITimeline(screenLayoutSize, timelineWidth);

    BuildMemoryWarning(screenLayoutSize);

    BuildInspectorWindow();
    BuildVramBudgetWindow();
    BuildBakeConfigWindow();
    BuildBakeReportWindow();
    BuildGridInstancingWindow();
    BuildHelpWindow();
}

void UserInterface::BuildInspectorWindow()
{
    if (!m_inspector)
        m_inspector = std::make_shared<GeometryInspector>(m_app, m_iconicFont, m_implot);
    m_inspector->Draw(m_showInspector);
}

void UserInterface::BuildHelpWindow()
{
    if (!m_showHelp)
        return;

    ImGui::SetNextWindowSize(ImVec2(430.f, 0.f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(320.f, 60.f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Help - Controls", &m_showHelp, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::End();
        return;
    }

    struct Binding { const char* keys; const char* action; };
    auto section = [](const char* title, const Binding* rows, size_t count)
    {
        if (!ImGui::CollapsingHeader(title, ImGuiTreeNodeFlags_DefaultOpen))
            return;
        if (!ImGui::BeginTable(title, 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
            return;
        ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthStretch, 1.f);
        ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthStretch, 2.f);
        for (size_t i = 0; i < count; ++i)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted(rows[i].keys);
            ImGui::TableNextColumn(); ImGui::TextUnformatted(rows[i].action);
        }
        ImGui::EndTable();
    };

    static const Binding kCamera[] = {
        { "W / S",          "Move forward / backward" },
        { "A / D",          "Move left / right" },
        { "Q / E",          "Move down / up" },
        { "Z / X",          "Roll left / right" },
        { "Shift (hold)",   "Move 3x faster" },
        { "Ctrl (hold)",    "Move 10x finer" },
        { "Alt (hold)",     "Orbit mode" },
        { "F",              "Reset camera to the scene default" },
        { "C",              "Print the camera parameters to stdout" },
        { "/",              "Freeze / unfreeze the LOD camera" },
    };
    static const Binding kMouse[] = {
        { "Left drag",      "Look around (or orbit with Alt)" },
        { "Right click",    "Pick a mesh into the Inspector" },
        { "Wheel",          "Adjust camera speed" },
        { "Alt + wheel",    "Zoom" },
    };
    static const Binding kView[] = {
        { "1",              "Next shading mode" },
        { "2 / 4",          "Next / previous color mode" },
        { "3",              "Toggle wireframe" },
        { "5",              "Next tonemapper" },
        { "T",              "Toggle the time view" },
        { "Left / Right",   "Decrease / increase max path bounces" },
    };
    static const Binding kApp[] = {
        { "Esc",            "Show / hide all UI" },
        { "P",              "Save a screenshot" },
        { "Shift + P",      "Save a screenshot with the UI" },
        { "Ctrl + R",       "Reload shaders" },
        { "Alt + F4",       "Quit" },
    };

    section("Camera", kCamera, std::size(kCamera));
    section("Mouse", kMouse, std::size(kMouse));
    section("View", kView, std::size(kView));
    section("Application", kApp, std::size(kApp));

    ImGui::End();
}

namespace
{
    struct VramSegment
    {
        const char* label;
        // `used` is occupancy inside `allocated`, and only means anything where
        // `tracked` is set; elsewhere the two are the same number and the bucket
        // draws no bar and no tick.
        uint64_t    used;
        uint64_t    allocated;  // VRAM the card has actually given up
        uint64_t    budget;     // what the applied budget permits
        uint64_t    pending;    // what the staged budget would permit
        bool        tracked;
        ImVec4      color;
        const char* tip;
    };

    enum class VramField { Allocated, Budget, Pending };

    uint64_t SegBytes(const VramSegment& s, VramField f)
    {
        return f == VramField::Budget  ? s.budget
             : f == VramField::Pending ? s.pending
                                       : s.allocated;
    }

    // Occupancy inside a pool's allocation, drawn like the profiler Memory tab's
    // rows so the two windows read the same.  `used` may exceed `allocated` on
    // the cluster_tess buckets, where it is this frame's demand.
    void UsageBar(uint64_t used, uint64_t allocated)
    {
        const ImVec4 satGreen (0.16f, 0.40f, 0.18f, 1.f);
        const ImVec4 satYellow(0.50f, 0.42f, 0.10f, 1.f);
        const ImVec4 satRed   (0.55f, 0.15f, 0.15f, 1.f);

        const float frac = float(double(used) / double(allocated));
        char u[24], a[24], overlay[64];
        MemoryFormatter(double(used),      u, int(std::size(u)));
        MemoryFormatter(double(allocated), a, int(std::size(a)));
        snprintf(overlay, std::size(overlay), "%s / %s (%.0f%%)", u, a, frac * 100.f);

        ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                              frac >= 0.90f ? satRed : (frac >= 0.80f ? satYellow : satGreen));
        ImGui::ProgressBar(frac, ImVec2(-1.f, 0.f), overlay);
        ImGui::PopStyleColor();
    }

    // Stacked horizontal bar over the card's VRAM: each segment is as wide as
    // `field`.  On the Allocated bar a vertical tick marks occupancy inside each
    // tracked pool; `mark` adds a tick across the whole bar (the driver-reported
    // process total).  ImGui has no stacked ProgressBar, so this is drawn
    // directly.
    void VramStackBar(const char* id, const VramSegment* segs, int count,
                      uint64_t capacity, uint64_t mark, VramField field)
    {
        const bool showUsedTicks = field == VramField::Allocated;
        const float height = ImGui::GetFrameHeight();
        const ImVec2 size(ImGui::GetContentRegionAvail().x, height);
        const ImVec2 p0 = ImGui::GetCursorScreenPos();
        const ImVec2 p1(p0.x + size.x, p0.y + size.y);
        ImDrawList* dl = ImGui::GetWindowDrawList();

        dl->AddRectFilled(p0, p1, ImGui::GetColorU32(ImGuiCol_FrameBg));

        uint64_t total = 0;
        int hovered = -1;
        if (capacity)
        {
            float x = p0.x;
            for (int i = 0; i < count; ++i)
            {
                const uint64_t bytes = SegBytes(segs[i], field);
                total += bytes;
                const float w = size.x * float(double(bytes) / double(capacity));
                if (w <= 0.f)
                    continue;
                const float xEnd = std::min(x + w, p1.x);
                dl->AddRectFilled(ImVec2(x, p0.y), ImVec2(xEnd, p1.y),
                                  ImGui::GetColorU32(segs[i].color));

                // Occupancy inside this pool's allocation, so a mostly-empty pool
                // reads as one at a glance.
                if (showUsedTicks && segs[i].tracked && segs[i].used < bytes && segs[i].used > 0)
                {
                    const float ux = x + w * float(double(segs[i].used) / double(bytes));
                    if (ux < xEnd)
                        dl->AddLine(ImVec2(ux, p0.y + 1.f), ImVec2(ux, p1.y - 1.f),
                                    ImGui::GetColorU32(ImVec4(1.f, 1.f, 1.f, 0.85f)), 1.5f);
                }

                if (ImGui::IsMouseHoveringRect(ImVec2(x, p0.y), ImVec2(xEnd, p1.y)))
                    hovered = i;
                x = xEnd;
            }
        }

        if (mark && capacity)
        {
            const float x = p0.x + size.x * std::min(1.f, float(double(mark) / double(capacity)));
            dl->AddLine(ImVec2(x, p0.y), ImVec2(x, p1.y),
                        ImGui::GetColorU32(ImVec4(1.f, 0.85f, 0.25f, 0.9f)), 2.f);
        }

        dl->AddRect(p0, p1, ImGui::GetColorU32(ImGuiCol_Border));
        ImGui::Dummy(size);

        if (hovered >= 0 && ImGui::IsWindowHovered())
        {
            const VramSegment& s = segs[hovered];
            const uint64_t bytes = SegBytes(s, field);
            char b[24], u[24];
            MemoryFormatter(double(bytes),  b, int(std::size(b)));
            MemoryFormatter(double(s.used), u, int(std::size(u)));
            if (showUsedTicks && s.tracked)
                ImGui::SetTooltip("%s\n%s used of %s allocated (%.0f%%)\n\n%s", s.label, u, b,
                                  bytes ? 100.0 * double(s.used) / double(bytes) : 0.0, s.tip);
            else
                ImGui::SetTooltip("%s\n%s\n\n%s", s.label, b, s.tip);
        }
        (void)id;

        char used[24], cap[24], overlay[80];
        MemoryFormatter(double(total),    used, int(std::size(used)));
        MemoryFormatter(double(capacity), cap,  int(std::size(cap)));
        const float frac = capacity ? float(double(total) / double(capacity)) : 0.f;
        snprintf(overlay, std::size(overlay), "%s / %s (%.0f%%)", used, cap, frac * 100.f);
        // The overlay carries the saturation warning; the segments are already
        // spoken for by their bucket colors.
        const ImVec4 textColor = frac >= 0.90f ? ImVec4(1.0f, 0.45f, 0.40f, 1.f)
                               : frac >= 0.75f ? ImVec4(1.0f, 0.85f, 0.25f, 1.f)
                                               : ImGui::GetStyleColorVec4(ImGuiCol_Text);
        const ImVec2 ts = ImGui::CalcTextSize(overlay);
        dl->AddText(ImVec2(p0.x + (size.x - ts.x) * 0.5f, p0.y + (size.y - ts.y) * 0.5f),
                    ImGui::GetColorU32(textColor), overlay);
    }
}

void UserInterface::BuildVramBudgetWindow()
{
    if (!m_showVramBudget)
        return;

    // First open, and after every commit, the staged set mirrors the live one.
    if (!m_stagedValid)
        SyncStagedBudgets();

    // Layout space, not DisplaySize: on a DPI-scaled display the two differ by
    // DisplayFramebufferScale and a fixed size lands off-screen.
    const ImGuiIO& io = ImGui::GetIO();
    const float layoutW = io.DisplaySize.x / io.DisplayFramebufferScale.x;
    const float layoutH = io.DisplaySize.y / io.DisplayFramebufferScale.y;
    ImGui::SetNextWindowSize(ImVec2(std::min(560.f, layoutW * 0.5f),
                                    std::min(660.f, layoutH - 40.f)), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(std::min(330.f, layoutW * 0.35f), 10.f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(360.f, 200.f), ImVec2(layoutW, layoutH - 20.f));
    // Arriving via the Cluster LOD panel's shortcut: raise the window, or the
    // field it is pointing at may be behind whatever had focus.
    if (m_highlightLoadMaps)
        ImGui::SetNextWindowFocus();
    bool open = true;
    if (!ImGui::Begin("VRAM Budget", &open))
    {
        ImGui::End();
        m_showVramBudget = open;
        return;
    }

    const stats::VramBreakdown& vb = stats::vramBreakdown;
    const stats::TextureMemStats& tm = stats::memUsageSamplers.textures;
    const uint64_t capacity = m_app.GetVramBytes();

    // ---- Header: the card, and what the driver says it will actually grant ----
    {
        char cap[24];
        MemoryFormatter(double(capacity), cap, int(std::size(cap)));
        ImGui::TextWrapped("%s: %s", m_app.GetAdapterName().c_str(), cap);
        if (m_app.IsVramSimulated())
        {
            char real[24];
            MemoryFormatter(double(m_app.GetPhysicalVramBytes()), real, int(std::size(real)));
            ImGui::TextColored(ImVec4(1.f, 0.85f, 0.f, 1.f),
                               "simulated (--vram-mb); real %s", real);
        }
        if (vb.driverValid)
        {
            char budget[24];
            MemoryFormatter(double(vb.driverBudget), budget, int(std::size(budget)));
            ImGui::TextDisabled("OS budget: %s", budget);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("What the driver is currently willing to grant this process.\n"
                                  "Routinely below the card's physical VRAM under a desktop\n"
                                  "compositor, so headroom against the card can be illusory.");
        }
    }

    ImGui::Spacing();

    // ---- The two bars -------------------------------------------------------
    const ImVec4 kColTextures  (0.20f, 0.34f, 0.50f, 1.f);
    const ImVec4 kColClodGeo   (0.16f, 0.40f, 0.18f, 1.f);
    const ImVec4 kColClodClas  (0.28f, 0.52f, 0.24f, 1.f);
    const ImVec4 kColCachedBlas(0.40f, 0.48f, 0.20f, 1.f);
    const ImVec4 kColClodMeta  (0.22f, 0.46f, 0.44f, 1.f);
    const ImVec4 kColTess      (0.50f, 0.36f, 0.14f, 1.f);
    const ImVec4 kColTessClas  (0.62f, 0.46f, 0.18f, 1.f);
    const ImVec4 kColTessData  (0.72f, 0.56f, 0.26f, 1.f);
    const ImVec4 kColBlas      (0.36f, 0.22f, 0.46f, 1.f);
    const ImVec4 kColTargets   (0.46f, 0.20f, 0.30f, 1.f);
    const ImVec4 kColOther     (0.34f, 0.34f, 0.34f, 1.f);

    // Three stacks over the same buckets: what the card has actually given up,
    // what the applied budgets permit, and what the staged ones would.  The
    // question the fields below are edited to answer is "does this fit".
    const StagedBudgets liveB  = LiveBudgets();
    const bool hasClod = m_app.GetRenderer().GetClusterLodResources() != nullptr;
    const bool hasTess = !m_app.GetScene().GetSubdMeshes().empty();

    auto texBytesFor = [&](int budgetMB) -> uint64_t
    {
        const uint64_t cap = uint64_t(std::max(0, budgetMB)) << 20;
        if (tm.budgetableCount == 0)
            return vb.textures;
        return (cap ? std::min(cap, tm.budgetableFullBytes) : tm.budgetableFullBytes)
               + tm.loadedOtherBytes;
    };
    auto mb = [](int v) { return uint64_t(std::max(0, v)) << 20; };

    const uint64_t rt        = vb.renderTargets + vb.envmap;
    const uint64_t unaccount = vb.Unaccounted();

    // Buckets with no budget to edit repeat their allocation in all three
    // columns, so the bars stay comparable end to end.  `tracked` marks the ones
    // whose occupancy we measure, which are exactly the Memory tab's barred rows.
    const VramSegment segs[] = {
        { "Textures", vb.textures, vb.textures,
          texBytesFor(liveB.textureBudgetMB), texBytesFor(m_staged.textureBudgetMB), false, kColTextures,
          "Material textures resident on the GPU. The budget is met by dropping\n"
          "high-resolution mips at load, so it only moves on a scene reload." },
        { "Cluster LOD Geometry", vb.clodGeometryUsed, vb.clodGeometry,
          hasClod ? mb(liveB.geometryPoolMB) : 0, hasClod ? mb(m_staged.geometryPoolMB) : 0, true, kColClodGeo,
          "Per-cluster vertex attributes of the resident LOD groups (quantized\n"
          "texcoords, plus packed normals when Vertex Normals is on). Positions\n"
          "are fetched from the AS, so they are not pooled. Blocks are acquired\n"
          "on demand, so Allocated steps in block-sized jumps." },
        { "Cluster LOD CLAS", vb.clodClasUsed, vb.clodClas,
          hasClod ? mb(liveB.clasPoolMB) : 0, hasClod ? mb(m_staged.clasPoolMB) : 0, true, kColClodClas,
          "Cluster acceleration structures for the resident clusters. One fixed\n"
          "buffer at the full budget, so Allocated always equals the budget and\n"
          "only the used fraction moves." },
        { "Cluster LOD Cached BLAS", vb.clodCachedBlas, vb.clodCachedBlas,
          vb.clodCachedBlasBudget, hasClod ? mb(m_staged.blasCachingPoolMB) : 0, false, kColCachedBlas,
          "Per-geometry BLAS kept across frames by BLAS caching, so a converged\n"
          "instance needs no rebuild. Not pre-allocated: blocks are acquired on\n"
          "demand up to the budget. Empty with --no-blascaching." },
        { "Cluster LOD Metadata", vb.clodMetadata, vb.clodMetadata,
          vb.clodMetadata, vb.clodMetadata, false, kColClodMeta,
          "Everything cluster-LOD outside the three pools, and the only place the\n"
          "count budgets show up: residency slot tables (Max resident groups),\n"
          "the traversal queues and render list (Render cluster bits), the\n"
          "per-frame CLAS-build staging (Max loads / frame), plus each geometry's\n"
          "LOD hierarchy and pinned low-detail data. Sized at load, not budgeted." },
        { "Cluster Tess Vertex Memory", vb.tessVerticesUsed, vb.tessVertices,
          hasTess ? mb(liveB.tessVertexMB) : 0, hasTess ? mb(m_staged.tessVertexMB) : 0, true, kColTess,
          "Positions and normals of the tessellated subdivision vertices,\n"
          "re-tessellated every frame. Used is this frame's demand, which can\n"
          "exceed the buffer - that is what the budget field flashes red for." },
        { "Cluster Tess CLAS", vb.tessClasUsed, vb.tessClas,
          hasTess ? mb(liveB.tessClasMB) : 0, hasTess ? mb(m_staged.tessClasMB) : 0, true, kColTessClas,
          "Cluster acceleration structures over the tessellated clusters." },
        { "Cluster Tess Shading Data", vb.tessClusterDataUsed, vb.tessClusterData,
          vb.tessClusterData, vb.tessClusterData, true, kColTessData,
          "Per-cluster surface descriptors the hit shader reads to evaluate the\n"
          "limit surface. Sized by Max Clusters, not by a byte budget." },
        { "BLAS + scratch", vb.blas, vb.blas, vb.blas, vb.blas, false, kColBlas,
          "Bottom-level acceleration structures over both geometry paths, and the\n"
          "build scratch. Just-sized from the scene, so there is no budget." },
        { "Render targets", rt, rt, rt, rt, false, kColTargets,
          "Output and history textures: the path-tracer targets, denoiser set,\n"
          "HiZ chain, z-prepass and the environment map. Scales with resolution." },
        { "Unaccounted", unaccount, unaccount, unaccount, unaccount, false, kColOther,
          "Driver total minus the buckets above: DLSS/NGX, nvrhi's descriptor\n"
          "heaps, the cluster_tess working buffers, and driver overhead." },
    };

    ImGui::TextUnformatted("Allocated");
    VramStackBar("##alloc", segs, int(std::size(segs)), capacity,
                 vb.driverValid ? vb.driverUsage : 0, VramField::Allocated);

    ImGui::TextUnformatted("Current Budget");
    VramStackBar("##cur", segs, int(std::size(segs)), capacity, 0, VramField::Budget);

    ImGui::TextUnformatted("Pending Budget");
    VramStackBar("##pend", segs, int(std::size(segs)), capacity, 0, VramField::Pending);

    // ---- Legend -------------------------------------------------------------
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Buckets", ImGuiTreeNodeFlags_DefaultOpen))
    {
    if (ImGui::BeginTable("##vram_legend", 4, ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg))
    {
        ImGui::TableSetupColumn("Bucket",           ImGuiTableColumnFlags_WidthStretch, 2.f);
        ImGui::TableSetupColumn("Used / Allocated", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn("Current Budget",   ImGuiTableColumnFlags_WidthStretch, 1.f);
        ImGui::TableSetupColumn("Pending Budget",   ImGuiTableColumnFlags_WidthStretch, 1.f);
        ImGui::TableHeadersRow();
        // Whole-row hover, so the bucket's tip is reachable from any column and
        // reads the same as its segment in the bars.  One frame behind, which a
        // tooltip does not care about.
        const int hoveredRow = ImGui::TableGetHoveredRow();
        char b[24];
        auto cell = [&](uint64_t bytes)
        {
            ImGui::TableNextColumn();
            MemoryFormatter(double(bytes), b, int(std::size(b)));
            ImGui::TextUnformatted(b);
        };
        for (const VramSegment& s : segs)
        {
            if (s.allocated == 0 && s.budget == 0 && s.pending == 0)
                continue;
            ImGui::TableNextRow();
            const bool rowHovered = ImGui::TableGetRowIndex() == hoveredRow;
            ImGui::TableNextColumn();
            ImGui::ColorButton("##c", s.color,
                               ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop,
                               ImVec2(12, 12));
            ImGui::SameLine();
            ImGui::TextUnformatted(s.label);
            if (rowHovered)
                ImGui::SetTooltip("%s", s.tip);
            // Only the pools whose occupancy we measure get a bar; the rest just
            // print what they hold, as the Memory tab's unbudgeted rows do.
            ImGui::TableNextColumn();
            if (s.tracked && s.allocated)
                UsageBar(s.used, s.allocated);
            else
            {
                MemoryFormatter(double(s.allocated), b, int(std::size(b)));
                ImGui::TextUnformatted(b);
            }
            cell(s.budget);
            // A staged change shows in the Pending column before it is applied.
            ImGui::TableNextColumn();
            MemoryFormatter(double(s.pending), b, int(std::size(b)));
            if (s.pending != s.budget)
                ImGui::TextColored(ImVec4(0.45f, 0.75f, 1.f, 1.f), "%s", b);
            else
                ImGui::TextUnformatted(b);
        }
        ImGui::EndTable();
    }
    if (!vb.driverValid)
        ImGui::TextDisabled("(no driver VRAM query on this API - DLSS/driver allocations are invisible)");
    } // Buckets

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Everything below scrolls; Apply/Revert are pinned under it so they stay
    // reachable however long the budget list gets.
    const float footerH = ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y * 2.f
                        + ImGui::GetTextLineHeightWithSpacing();
    ImGui::BeginChild("##vram_scroll", ImVec2(0.f, -footerH), false);

    // ---- Textures: the only two settings that need a scene reload -----------
    const bool clodVertexNormals = m_app.GetClusterLodVertexNormalsEnabled();
    if (m_highlightLoadMaps)
        ImGui::SetNextItemOpen(true, ImGuiCond_Always);  // the flashing field is in here
    if (ImGui::CollapsingHeader("Textures", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::PushItemWidth(160.f);
        StagedField("Texture budget (MB)", m_staged.textureBudgetMB, liveB.textureBudgetMB,
                    256, 1024, 0, std::max(0, int(capacity >> 20)));
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Cap on resident texture memory, met by dropping high-resolution\n"
                              "mips at load (KTX2/DDS only).  0 = unlimited.  Takes effect on\n"
                              "Apply, which reloads the scene.  CLI: --texture-budget-mb.");
        ImGui::PopItemWidth();

        // Flashed when the Cluster LOD panel sent the user here.  Cleared as soon
        // as the field is touched, so it is a pointer rather than a mode.
        const bool flashLoadMaps = m_highlightLoadMaps
                                && fmodf(float(ImGui::GetTime()), 1.0f) < 0.5f;
        if (flashLoadMaps)
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.20f, 0.52f, 0.75f, 1.f));
        ImGui::BeginDisabled(!clodVertexNormals);
        if (ImGui::Checkbox("Load Normal Maps", &m_staged.normalMaps))
            m_highlightLoadMaps = false;
        ImGui::EndDisabled();
        if (flashLoadMaps)
            ImGui::PopStyleColor();
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip(clodVertexNormals
                ? "Read the material normal maps off disk.  On a large scene they are\n"
                  "~42%% of the texture bytes, so this is the single biggest lever\n"
                  "here.  Takes effect on Apply, which reloads the scene.  Shading\n"
                  "with them is a separate live toggle in the Cluster LOD panel."
                : "Needs cluster-LOD Vertex Normals: a normal map perturbs the\n"
                  "interpolated vertex normal, so it has nothing to sit on while\n"
                  "that is off.");
    }

    // ---- Pool budgets: staged like the rest, applied without a reload -------
    if (BeginDisableableSection("Cluster LOD", hasClod, kNoClusterLod))
    {
        ImGui::PushItemWidth(160.f);
        BuildClusterLodBudgetControls();
        ImGui::PopItemWidth();
    }
    if (BeginDisableableSection("Cluster Tess", hasTess, kNoTess))
    {
        ImGui::PushItemWidth(160.f);
        BuildTessBudgetControls();
        ImGui::PopItemWidth();
    }

    ImGui::EndChild();

    ImGui::Separator();

    const bool texBudgetChanged  = m_staged.textureBudgetMB != liveB.textureBudgetMB;
    const bool normalMapsChanged = m_staged.normalMaps      != liveB.normalMaps;
    const bool needsReload = texBudgetChanged || normalMapsChanged;
    const bool poolsChanged =
        m_staged.maxResidentGroups != liveB.maxResidentGroups ||
        m_staged.geometryPoolMB    != liveB.geometryPoolMB    ||
        m_staged.geometryBlockMB   != liveB.geometryBlockMB   ||
        m_staged.clasPoolMB        != liveB.clasPoolMB        ||
        m_staged.blasCachingPoolMB != liveB.blasCachingPoolMB ||
        m_staged.maxLoadsPerFrame  != liveB.maxLoadsPerFrame  ||
        m_staged.renderClusterBits != liveB.renderClusterBits ||
        m_staged.tessMaxKClusters  != liveB.tessMaxKClusters  ||
        m_staged.tessVertexMB      != liveB.tessVertexMB      ||
        m_staged.tessClasMB        != liveB.tessClasMB;
    const bool anyChanged = needsReload || poolsChanged;

    // Apply itself is highlighted while it has something to commit, so a change
    // scrolled out of view is still visible down here.
    if (anyChanged)
    {
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.16f, 0.42f, 0.62f, 1.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.52f, 0.75f, 1.f));
    }
    ImGui::BeginDisabled(!anyChanged);
    if (ImGui::Button("Apply"))
    {
        m_highlightLoadMaps = false;
        // Only the reload-requiring pair needs confirming; the pools can just go.
        if (needsReload)
            ImGui::OpenPopup("##VramReloadConfirm");
        else
        {
            ApplyStagedPoolBudgets();
            SyncStagedBudgets();
        }
    }
    ImGui::EndDisabled();
    if (anyChanged)
        ImGui::PopStyleColor(2);
    ImGui::SameLine();
    if (ImGui::Button("Revert"))
        SyncStagedBudgets();
    ImGui::SameLine();
    if (needsReload)
        ImGui::TextDisabled("(Apply reloads the scene)");
    else if (poolsChanged)
        ImGui::TextDisabled("(Apply reallocates the pools)");
    else
        ImGui::TextDisabled("(no pending changes)");

    ImGui::SetNextWindowPos(LayoutCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("##VramReloadConfirm", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::TextUnformatted("The following settings changed:");
        ImGui::Spacing();
        if (texBudgetChanged)
        {
            auto fmtMB = [](int mb) { return mb > 0 ? std::to_string(mb) + " MB" : std::string("unlimited"); };
            ImGui::BulletText("Texture budget: %s -> %s",
                              fmtMB(liveB.textureBudgetMB).c_str(),
                              fmtMB(m_staged.textureBudgetMB).c_str());
        }
        if (normalMapsChanged)
            ImGui::BulletText("Load normal maps: %s -> %s",
                              liveB.normalMaps ? "on" : "off",
                              m_staged.normalMaps ? "on" : "off");

        ImGui::Spacing();
        BuildReloadWarningText();
        ImGui::Spacing();

        if (ImGui::Button("Reload", ImVec2(90, 0)))
        {
            // Pool budgets ride along, so one Apply commits everything staged.
            ApplyStagedPoolBudgets();
            m_app.ApplyTextureSettingsAndReload(m_staged.textureBudgetMB, m_staged.normalMaps);
            m_stagedValid = false;  // re-sync from the reloaded scene
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(90, 0)))
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    ImGui::End();
    m_showVramBudget = open;
    if (!open)
        m_highlightLoadMaps = false;
}

void UserInterface::BuildBakeReportWindow()
{
    UIData& uiData = m_app.GetUIData();
    if (!uiData.showBakeReportWindow)
        return;

    const RTXMGDemoApp::BakeReport& r = m_app.GetBakeReport();
    if (!r.valid)
        return;

    if (uiData.focusBakeReport)
    {
        ImGui::SetNextWindowFocus();
        uiData.focusBakeReport = false;
    }
    ImGui::SetNextWindowSize(ImVec2(520, 0), ImGuiCond_Appearing);
    {
        const ImGuiIO& _io = ImGui::GetIO();
        ImGui::SetNextWindowPos(
            ImVec2(_io.DisplaySize.x / (_io.DisplayFramebufferScale.x * 2.f),
                   _io.DisplaySize.y / (_io.DisplayFramebufferScale.y * 2.f)),
            ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    }
    bool open = true;
    if (!ImGui::Begin("Bake Report", &open))
    {
        ImGui::End();
        if (!open) uiData.showBakeReportWindow = false;
        return;
    }
    if (!open) { uiData.showBakeReportWindow = false; ImGui::End(); return; }

    // --- Metadata row --------------------------------------------------------
    {
        const double s = r.bakeSeconds;
        char timeBuf[32];
        if (s >= 60.0)
            snprintf(timeBuf, sizeof(timeBuf), "%dm %.0fs", int(s) / 60, std::fmod(s, 60.0));
        else
            snprintf(timeBuf, sizeof(timeBuf), "%.1fs", s);

        ImGui::Text("Bake time: %s", timeBuf);
        ImGui::SameLine(0, 20);
        ImGui::Text("Workers: %u", r.workerThreads);
        ImGui::SameLine(0, 20);
        ImGui::Text("RAM: %+lld MiB", (long long)r.memDeltaMiB);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // --- Stats table ---------------------------------------------------------
    const bool hasBefore = r.before.valid;
    const int cols = hasBefore ? 4 : 2;

    const ImGuiTableFlags tflags = ImGuiTableFlags_BordersOuter
                                 | ImGuiTableFlags_BordersInnerH
                                 | ImGuiTableFlags_RowBg
                                 | ImGuiTableFlags_SizingStretchSame;
    if (ImGui::BeginTable("##BakeReportTable", cols, tflags))
    {
        ImGui::TableSetupColumn("",       ImGuiTableColumnFlags_WidthStretch, 2.0f);
        if (hasBefore)
            ImGui::TableSetupColumn("Before", ImGuiTableColumnFlags_WidthStretch, 1.5f);
        ImGui::TableSetupColumn("After",  ImGuiTableColumnFlags_WidthStretch, 1.5f);
        if (hasBefore)
            ImGui::TableSetupColumn("Delta",  ImGuiTableColumnFlags_WidthStretch, 1.5f);
        ImGui::TableHeadersRow();

        // --- Format helpers --------------------------------------------------
        auto fmtCount = [](uint64_t n, char* buf, int sz) {
            if (n >= 1000000000) snprintf(buf, sz, "%.2fB", n / 1e9);
            else if (n >= 1000000) snprintf(buf, sz, "%.2fM", n / 1e6);
            else if (n >= 1000) snprintf(buf, sz, "%.1fK", n / 1e3);
            else snprintf(buf, sz, "%llu", (unsigned long long)n);
        };
        auto fmtSize = [](uint64_t bytes, char* buf, int sz) {
            if (bytes >= (1ull << 30))
                snprintf(buf, sz, "%.2f GiB", bytes / double(1ull << 30));
            else if (bytes >= (1ull << 20))
                snprintf(buf, sz, "%.1f MiB", bytes / double(1ull << 20));
            else
                snprintf(buf, sz, "%llu B", (unsigned long long)bytes);
        };

        // Numeric row: label | [before] | after | [delta with color]
        auto numRow = [&](const char* label,
                          uint64_t vBefore, uint64_t vAfter,
                          bool isSizeRow, bool smallerIsBetter)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted(label);
            char b1[32], b2[32];
            auto fmt = isSizeRow ? fmtSize : fmtCount;
            if (hasBefore)
            {
                ImGui::TableNextColumn();
                fmt(vBefore, b1, sizeof(b1)); ImGui::TextUnformatted(b1);
            }
            ImGui::TableNextColumn();
            fmt(vAfter, b2, sizeof(b2)); ImGui::TextUnformatted(b2);
            if (hasBefore)
            {
                ImGui::TableNextColumn();
                const int64_t delta = (int64_t)vAfter - (int64_t)vBefore;
                if (delta == 0 || vBefore == 0)
                {
                    ImGui::TextDisabled("=");
                }
                else
                {
                    const double pct = 100.0 * delta / double(vBefore);
                    const bool good = (delta < 0) == smallerIsBetter;
                    ImVec4 col = good ? ImVec4(0.3f, 0.9f, 0.3f, 1.f)
                                      : ImVec4(1.f,  0.4f, 0.4f, 1.f);
                    ImGui::TextColored(col, "%+.1f%%", pct);
                }
            }
        };

        // Bool row
        auto boolRow = [&](const char* label, bool vBefore, bool vAfter)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted(label);
            if (hasBefore)
            {
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(vBefore ? "Yes" : "No");
            }
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(vAfter ? "Yes" : "No");
            if (hasBefore)
            {
                ImGui::TableNextColumn();
                if (vBefore != vAfter)
                    ImGui::TextColored(ImVec4(1.f, 0.85f, 0.f, 1.f), "changed");
                else
                    ImGui::TextDisabled("=");
            }
        };

        const auto& b = r.before;
        const auto& a = r.after;
        numRow("Geometries", b.geometries,  a.geometries,  false, false);
        numRow("Groups",     b.groups,      a.groups,      false, true);
        numRow("Clusters",   b.clusters,    a.clusters,    false, true);
        numRow("Triangles",  b.triangles,   a.triangles,   false, false);
        numRow("Disk size",  b.bakedBytes,  a.bakedBytes,  true,  true);
        numRow("Geo GPU size", b.deviceBytes, a.deviceBytes, true,  true);
        boolRow("Compressed",    b.compressed,  a.compressed);
        boolRow("Quantized UVs", b.quantizedUv, a.quantizedUv);
        if (b.compressed || a.compressed)
        {
            numRow("Pos drop bits",
                b.compressed ? b.compressionPosDropBits : 0u,
                a.compressed ? a.compressionPosDropBits : 0u, false, false);
            numRow("Tex drop bits",
                b.compressed ? b.compressionTexDropBits : 0u,
                a.compressed ? a.compressionTexDropBits : 0u, false, false);
        }

        ImGui::EndTable();
    }

    ImGui::Spacing();
    if (ImGui::Button("Close"))
        uiData.showBakeReportWindow = false;

    ImGui::End();
}

void UserInterface::BuildBakeConfigWindow()
{
    UIData& uiData = m_app.GetUIData();
    if (!uiData.showBakeConfigWindow)
    {
        m_bakeConfigWindowOpen = false;
        return;
    }

    // Force a re-init after a rebake (baseline must reflect the new baked state).
    if (uiData.refreshBakeConfig)
    {
        m_bakeConfigWindowOpen = false;
        uiData.refreshBakeConfig = false;
    }

    // Rising-edge: initialize from the current baked config each time the window opens.
    if (!m_bakeConfigWindowOpen)
    {
        m_baselineBakerConfig = m_app.GetCurrentBakerConfig();
        m_editingBakerConfig  = m_baselineBakerConfig;
        m_bakeConfigWindowOpen = true;
    }

    ImGui::SetNextWindowSize(ImVec2(460, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(220, 80), ImGuiCond_FirstUseEver);
    bool open = true;
    if (!ImGui::Begin("Bake Config", &open))
    {
        ImGui::End();
        if (!open) uiData.showBakeConfigWindow = false;
        return;
    }
    if (!open)
    {
        uiData.showBakeConfigWindow = false;
        ImGui::End();
        return;
    }

    // Info row: cache directory and config file path
    const fs::path jsonPath = m_app.GetBakeConfigJsonPath();
    {
        ImGui::TextDisabled("Cache Dir: ");
        ImGui::SameLine(0, 0);
        const std::string& cd = m_app.GetArgs().clusterCacheDir;
        if (cd.empty())
            ImGui::TextUnformatted("(default: next to scene file)");
        else
            ImGui::TextUnformatted(cd.c_str());
    }
    {
        ImGui::TextDisabled("Config File:");
        ImGui::SameLine(0, 0);
        if (jsonPath.empty())
            ImGui::TextUnformatted(" (no scene loaded)");
        else if (fs::exists(jsonPath))
            ImGui::TextUnformatted(jsonPath.generic_string().c_str());
        else
            ImGui::Text(" %s  (not saved yet)", jsonPath.filename().generic_string().c_str());
    }

    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextDisabled("Hover any field for a tooltip.");
    ImGui::Spacing();

    // --- Field helpers -------------------------------------------------------
    const ImVec4 kChangedColor(0.55f, 0.46f, 0.0f, 0.70f);

    auto hiBegin = [&](bool changed) {
        if (changed) ImGui::PushStyleColor(ImGuiCol_FrameBg, kChangedColor);
    };
    auto hiEnd = [&](bool changed) {
        if (changed) ImGui::PopStyleColor();
    };

    // Tooltip shown after hovering any part of an item (including its label).
    auto tip = [](const char* text) {
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
            ImGui::SetTooltip("%s", text);
    };

    ImGui::PushItemWidth(160.f);

    // ---- Cluster Geometry ---------------------------------------------------
    if (ImGui::CollapsingHeader("Cluster Geometry", ImGuiTreeNodeFlags_DefaultOpen))
    {
        auto fieldU32 = [&](const char* label, uint32_t& val, uint32_t base, int minV, int maxV, const char* tooltip) {
            bool ch = val != base;
            hiBegin(ch);
            int iv = int(val);
            if (ImGui::InputInt(label, &iv, 1, 8))
                val = uint32_t(std::clamp(iv, minV, maxV));
            hiEnd(ch);
            tip(tooltip);
        };
        fieldU32("Cluster Triangles",  m_editingBakerConfig.clusterTriangles,   m_baselineBakerConfig.clusterTriangles,   1, 256,
            "Max triangles per cluster (meshoptimizer constraint).\n"
            "Smaller values multiply group count and blow the streaming\n"
            "working set past the CLAS pool, wedging streaming.\n"
            "Default 128.  CLI: --clustersize <tris> <verts>.");
        fieldU32("Cluster Vertices",   m_editingBakerConfig.clusterVertices,    m_baselineBakerConfig.clusterVertices,    1, 256,
            "Max vertices per cluster (meshoptimizer constraint).\n"
            "Paired with Cluster Triangles.  Default 128.\n"
            "CLI: --clustersize <tris> <verts>.");
        fieldU32("Cluster Group Size", m_editingBakerConfig.clusterGroupSize,   m_baselineBakerConfig.clusterGroupSize,   1, 128,
            "Clusters per LOD group, decimated together and sharing a\n"
            "common error metric.  Max 128.  Default 32.");
        fieldU32("Node Width",         m_editingBakerConfig.preferredNodeWidth, m_baselineBakerConfig.preferredNodeWidth, 2, 32,
            "Preferred number of children per LOD hierarchy node\n"
            "(branching factor of the spatial tree).  Default 8.");
    }

    // ---- Simplification -----------------------------------------------------
    if (ImGui::CollapsingHeader("Simplification", ImGuiTreeNodeFlags_DefaultOpen))
    {
        auto fieldF32 = [&](const char* label, float& val, float base, float lo, float hi, const char* tooltip, const char* fmt = "%.3f") {
            bool ch = val != base;
            hiBegin(ch);
            ImGui::SliderFloat(label, &val, lo, hi, fmt);
            hiEnd(ch);
            tip(tooltip);
        };
        fieldF32("Normal Weight",   m_editingBakerConfig.simplifyNormalWeight,     m_baselineBakerConfig.simplifyNormalWeight,     0.f, 1.f,
            "How much surface normal error counts in the simplification\n"
            "metric.  Larger values preserve normals at the cost of fewer\n"
            "edge collapses (higher detail, bigger streaming working set).\n"
            "Keep non-zero: at 0, coplanar geometry collapses with shredded\n"
            "shading normals.  Default 0.5.  CLI: --simplifyweights.");
        fieldF32("Texcoord Weight", m_editingBakerConfig.simplifyTexCoordWeight,   m_baselineBakerConfig.simplifyTexCoordWeight,   0.f, 1.f,
            "How much UV error counts in the simplification metric.\n"
            "Keep non-zero: at 0, flat/fan geometry reports near-zero\n"
            "position error and a UV-shredded coarse LOD passes the\n"
            "lodPixelError test from a few meters away.  Default 0.5.\n"
            "CLI: --simplifyweights.");
        fieldF32("Tangent Weight",  m_editingBakerConfig.simplifyTangentWeight,    m_baselineBakerConfig.simplifyTangentWeight,    0.f, 1.f,
            "How much tangent vector error counts in the simplification\n"
            "metric.  0 = disabled (default).");
        fieldF32("Tangent Sign Wt", m_editingBakerConfig.simplifyTangentSignWeight,m_baselineBakerConfig.simplifyTangentSignWeight,0.f, 1.f,
            "How much tangent handedness (sign) error counts.\n"
            "0 = disabled (default).");
        fieldF32("Material Weight", m_editingBakerConfig.simplifyMaterialWeight,   m_baselineBakerConfig.simplifyMaterialWeight,   0.f, 64.f,
            "Penalty for collapsing edges that cross material boundaries.\n"
            "Deliberately large default (32) to preserve material seams\n"
            "on multi-material geometries.  0 = disabled.",
            "%.1f");
    }

    // ---- LOD Error ----------------------------------------------------------
    if (ImGui::CollapsingHeader("LOD Error"))
    {
        auto fieldF32i = [&](const char* label, float& val, float base, const char* tooltip) {
            bool ch = val != base;
            hiBegin(ch);
            ImGui::InputFloat(label, &val, 0.01f, 0.1f, "%.4f");
            hiEnd(ch);
            tip(tooltip);
        };
        fieldF32i("Merge Previous", m_editingBakerConfig.lodErrorMergePrevious, m_baselineBakerConfig.lodErrorMergePrevious,
            "Error propagation scale across LOD levels.\n"
            "Group error = max(childError * Previous, ownError).\n"
            "Values < 1 break DAG error monotonicity — a parent could\n"
            "then advertise less error than its children, corrupting\n"
            "LOD selection.  Default 1.0.  CLI: --loderrormerge.");
        fieldF32i("Merge Additive", m_editingBakerConfig.lodErrorMergeAdditive, m_baselineBakerConfig.lodErrorMergeAdditive,
            "Additive error term: group error += Additive * ownError\n"
            "after the max computation.  Default 0.0.\n"
            "CLI: --loderrormerge.");
        fieldF32i("Edge Limit",     m_editingBakerConfig.lodErrorEdgeLimit,     m_baselineBakerConfig.lodErrorEdgeLimit,
            "Limit LOD error by edge length, aimed at removing sub-pixel\n"
            "triangles even when attribute error is high.\n"
            "Default 0.0 (disabled).");
    }

    // ---- Meshopt ------------------------------------------------------------
    if (ImGui::CollapsingHeader("Meshopt"))
    {
        {
            bool ch = m_editingBakerConfig.meshoptPreferRayTracing != m_baselineBakerConfig.meshoptPreferRayTracing;
            hiBegin(ch); ImGui::Checkbox("Prefer Ray Tracing", &m_editingBakerConfig.meshoptPreferRayTracing); hiEnd(ch);
            tip("Configure meshoptimizer's cluster builder for ray tracing\n"
                "layouts (SAH-optimized) rather than rasterization.\n"
                "Default ON.");
        }
        {
            bool ch = m_editingBakerConfig.meshoptFillWeight != m_baselineBakerConfig.meshoptFillWeight;
            hiBegin(ch); ImGui::SliderFloat("Fill Weight",  &m_editingBakerConfig.meshoptFillWeight, 0.f, 1.f); hiEnd(ch);
            tip("When Prefer Ray Tracing is ON: balance between SAH-optimized\n"
                "clusters (near 0) and tightly-filled clusters (near 1).\n"
                "Default 0.5.");
        }
        {
            bool ch = m_editingBakerConfig.meshoptSplitFactor != m_baselineBakerConfig.meshoptSplitFactor;
            hiBegin(ch); ImGui::InputFloat("Split Factor", &m_editingBakerConfig.meshoptSplitFactor, 0.1f, 0.5f, "%.3f"); hiEnd(ch);
            tip("When Prefer Ray Tracing is OFF: influences the maximum\n"
                "cluster size before splitting.  Default 1.5.");
        }
    }

    // ---- Compression --------------------------------------------------------
    if (ImGui::CollapsingHeader("Compression"))
    {
        {
            bool ch = m_editingBakerConfig.useCompressedData != m_baselineBakerConfig.useCompressedData;
            hiBegin(ch); ImGui::Checkbox("Use Compressed Data", &m_editingBakerConfig.useCompressedData); hiEnd(ch);
            tip("Arithmetic-pack vertex data at bake time; decompress per\n"
                "group at upload.  Reduces cache file size and host RAM.\n"
                "Composes with UV Quantization: quantized delta words are\n"
                "packed instead of raw float2s (better ratio).\n"
                "CLI: --compress.");
        }
        if (m_editingBakerConfig.useCompressedData)
        {
            auto fieldBits = [&](const char* label, uint32_t& val, uint32_t base, const char* tooltip) {
                bool ch = val != base;
                hiBegin(ch);
                int iv = int(val);
                if (ImGui::SliderInt(label, &iv, 0, 16)) val = uint32_t(iv);
                hiEnd(ch);
                tip(tooltip);
            };
            fieldBits("Pos Drop Bits", m_editingBakerConfig.compressionPosDropBits, m_baselineBakerConfig.compressionPosDropBits,
                "Position mantissa bits zeroed before arithmetic packing.\n"
                "More bits dropped = better compression, lower precision.\n"
                "Default 7.");
            fieldBits("Tex Drop Bits", m_editingBakerConfig.compressionTexDropBits, m_baselineBakerConfig.compressionTexDropBits,
                "Texcoord mantissa bits zeroed before arithmetic packing.\n"
                "Default 7.");
        }
    }

    // ---- UV Quantization ----------------------------------------------------
    {
        bool ch = m_editingBakerConfig.quantizeTexCoords != m_baselineBakerConfig.quantizeTexCoords;
        hiBegin(ch); ImGui::Checkbox("Quantize Tex Coords", &m_editingBakerConfig.quantizeTexCoords); hiEnd(ch);
        tip("Store per-cluster UVs as power-of-2 grid-quantized values\n"
            "instead of raw float2, roughly halving their bytes\n"
            "(~4 B/vertex vs ~8 B/vertex).\n"
            "When 'Use Compressed Data' is also on, the quantized delta\n"
            "words are arithmetic-packed on top (beats packing raw float2s\n"
            "because delta words carry no exponent field).\n"
            "Adaptive per cluster: falls back to raw float2 when the\n"
            "UV range needs a step coarser than 2^-14\n"
            "(~0.13 texel error at 4K resolution).\n"
            "Default ON.  CLI: --no-uvquant.");
    }

    ImGui::PopItemWidth();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    const bool hasDelta = m_editingBakerConfig != m_baselineBakerConfig;

    if (ImGui::Button("Apply"))
    {
        if (hasDelta)
            ImGui::OpenPopup("##BakeConfirm");
        else
            uiData.showBakeConfigWindow = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel"))
        uiData.showBakeConfigWindow = false;

    if (hasDelta)
    {
        ImGui::SameLine();
        ImGui::TextDisabled("(highlighted fields changed)");
    }

    // ---- Confirmation popup -------------------------------------------------
    {
        const ImGuiIO& _io = ImGui::GetIO();
        ImGui::SetNextWindowPos(
            ImVec2(_io.DisplaySize.x / (_io.DisplayFramebufferScale.x * 2.f),
                   _io.DisplaySize.y / (_io.DisplayFramebufferScale.y * 2.f)),
            ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    }
    if (ImGui::BeginPopupModal("##BakeConfirm", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::TextUnformatted("The following bake settings have changed:");
        ImGui::Spacing();

        // List all changed fields
        auto showDelta = [&](const char* name, auto a, auto b) {
            if (a != b)
            {
                ImGui::BulletText("%s", name);
                ImGui::SameLine();
                // Format from->to inline
                char buf[128];
                if constexpr (std::is_same_v<decltype(a), bool>)
                    snprintf(buf, sizeof(buf), ": %s -> %s", a ? "true" : "false", b ? "true" : "false");
                else if constexpr (std::is_integral_v<decltype(a)>)
                    snprintf(buf, sizeof(buf), ": %u -> %u", unsigned(a), unsigned(b));
                else
                    snprintf(buf, sizeof(buf), ": %.4g -> %.4g", double(a), double(b));
                ImGui::TextDisabled("%s", buf);
            }
        };
        const BakerConfig& A = m_baselineBakerConfig;
        const BakerConfig& B = m_editingBakerConfig;
        showDelta("Cluster Triangles",    A.clusterTriangles,         B.clusterTriangles);
        showDelta("Cluster Vertices",     A.clusterVertices,          B.clusterVertices);
        showDelta("Cluster Group Size",   A.clusterGroupSize,         B.clusterGroupSize);
        showDelta("Node Width",           A.preferredNodeWidth,       B.preferredNodeWidth);
        showDelta("LOD Merge Previous",   A.lodErrorMergePrevious,    B.lodErrorMergePrevious);
        showDelta("LOD Merge Additive",   A.lodErrorMergeAdditive,    B.lodErrorMergeAdditive);
        showDelta("LOD Edge Limit",       A.lodErrorEdgeLimit,        B.lodErrorEdgeLimit);
        showDelta("Meshopt Prefer RT",    A.meshoptPreferRayTracing,  B.meshoptPreferRayTracing);
        showDelta("Meshopt Fill Weight",  A.meshoptFillWeight,        B.meshoptFillWeight);
        showDelta("Meshopt Split Factor", A.meshoptSplitFactor,       B.meshoptSplitFactor);
        showDelta("Use Compressed Data",  A.useCompressedData,        B.useCompressedData);
        showDelta("Pos Drop Bits",        A.compressionPosDropBits,   B.compressionPosDropBits);
        showDelta("Tex Drop Bits",        A.compressionTexDropBits,   B.compressionTexDropBits);
        showDelta("Normal Weight",        A.simplifyNormalWeight,     B.simplifyNormalWeight);
        showDelta("Texcoord Weight",      A.simplifyTexCoordWeight,   B.simplifyTexCoordWeight);
        showDelta("Tangent Weight",       A.simplifyTangentWeight,    B.simplifyTangentWeight);
        showDelta("Tangent Sign Wt",      A.simplifyTangentSignWeight,B.simplifyTangentSignWeight);
        showDelta("Material Weight",      A.simplifyMaterialWeight,   B.simplifyMaterialWeight);
        showDelta("Quantize Tex Coords",  A.quantizeTexCoords,        B.quantizeTexCoords);

        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.75f, 0.0f, 1.f));
        ImGui::TextUnformatted("Changing bake settings will invalidate the geometry cache.");
        ImGui::TextUnformatted("Re-baking may take a long time depending on scene complexity.");
        ImGui::PopStyleColor();
        ImGui::Spacing();

        if (ImGui::Button("OK", ImVec2(80, 0)))
        {
            m_app.ApplyBakerConfigAndReload(m_editingBakerConfig);
            uiData.showBakeConfigWindow = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(80, 0)))
            ImGui::CloseCurrentPopup();

        ImGui::EndPopup();
    }

    ImGui::End();
}

void UserInterface::BuildMemoryWarning(int2 screenLayoutSize)
{
    auto& stats = GetApp().m_BuildStats;
    bool clusterCountExceeded = stats.desired.m_numClusters > stats.allocated.m_numClusters;
    bool clasMemoryExceeded = stats.desired.m_clasSize > stats.allocated.m_clasSize;
    bool vertexMemoryExceeded = stats.desired.m_vertexBufferSize > stats.allocated.m_vertexBufferSize;
    bool vertexNormalsMemoryExceeded = stats.desired.m_vertexNormalsBufferSize > stats.allocated.m_vertexNormalsBufferSize;
    bool tessExceeded = clusterCountExceeded || clasMemoryExceeded || vertexMemoryExceeded || vertexNormalsMemoryExceeded;

    // traversal_setup preserves the raw pre-clamp demand in desiredRenderClusters and
    // the cap it applied in effectiveMaxRenderClusters (the full budget minus the
    // cached-cluster reservation).  Clusters were dropped — and the scene
    // flickers — exactly when the demand exceeds that effective cap.
    auto& renderer = m_app.GetRenderer();
    uint32_t desiredRenderClusters = 0;
    uint32_t maxRenderClusters     = 0;   // full budget (kMaxRenderClusters)
    uint32_t cachedClusterReserved = 0;   // budget withheld for cached clusters
    bool renderClustersExceeded = false;
    // Same signal for the fixed-size traversal queues (kMaxTraversalInfos):
    // over the cap means traversal work was dropped, so the LoD is wrong rather
    // than merely coarse.
    uint32_t desiredTraversalNodes  = 0;
    uint32_t desiredTraversalGroups = 0;
    bool traversalQueueExceeded = false;
    if (renderer.GetClusterLodResources())
    {
        const auto& c = renderer.GetClusterLodCounters();
        desiredRenderClusters  = c.desiredRenderClusters;
        maxRenderClusters      = renderer.GetClusterLodMaxRenderClusters();
        const uint32_t effectiveMax = c.effectiveMaxRenderClusters;
        cachedClusterReserved  = (maxRenderClusters > effectiveMax) ? maxRenderClusters - effectiveMax : 0;
        // effectiveMax == 0 means no valid readback yet; don't false-positive.
        renderClustersExceeded = effectiveMax != 0 && desiredRenderClusters > effectiveMax;

        desiredTraversalNodes  = c.desiredTraversalNodes;
        desiredTraversalGroups = c.desiredTraversalGroups;
        traversalQueueExceeded = desiredTraversalNodes  > ClusterLodPass::kMaxTraversalInfos
                              || desiredTraversalGroups > ClusterLodPass::kMaxTraversalInfos;
    }

    if (!tessExceeded && !renderClustersExceeded && !traversalQueueExceeded)
        return;

    ImVec2 overlayPos(screenLayoutSize.x * 0.5f, 10.0f); // Center X, 10px from the top
    ImVec2 overlayPivot(0.5f, 0.0f); // Center horizontally, stick to the top

    ImGui::SetNextWindowPos(overlayPos, ImGuiCond_Always, overlayPivot);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, IM_COL32(128, 0, 0, 200));  // Dark red
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 255));    // White text

    // Not NoInputs: the "Adjust VRAM Budget" button below has to be clickable.
    ImGui::Begin("Memory Exceeded", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                  ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::PushFont(m_nvidiaBldFont);
    if (tessExceeded)
        ImGui::Text("Tessellation memory budget exceeded");
    if (renderClustersExceeded)
        ImGui::Text("Render cluster budget exceeded");
    if (traversalQueueExceeded)
        ImGui::Text("Traversal queue capacity exceeded");
    ImGui::PopFont();
    ImGui::Text("Expect flickering.");
    ImGui::SameLine();
    if (ImGui::Button("Adjust VRAM Budget"))
        m_showVramBudget = true;

    if (renderClustersExceeded)
    {
        char bufDesired[64];
        char bufMax[64];
        HumanFormatter(desiredRenderClusters, bufDesired, sizeof(bufDesired));
        HumanFormatter(maxRenderClusters, bufMax, sizeof(bufMax));
        if (cachedClusterReserved > 0)
        {
            // Show desired + cached reservation vs the full budget: clusters drop
            // when those two together exceed kMaxRenderClusters.
            char bufReserved[64];
            HumanFormatter(cachedClusterReserved, bufReserved, sizeof(bufReserved));
            ImGui::Text("Render Clusters %s + %s cached reserved / %s", bufDesired, bufReserved, bufMax);
        }
        else
        {
            ImGui::Text("Render Clusters %s / %s", bufDesired, bufMax);
        }
    }

    if (traversalQueueExceeded)
    {
        char bufNodes[64];
        char bufGroups[64];
        char bufCap[64];
        HumanFormatter(desiredTraversalNodes, bufNodes, sizeof(bufNodes));
        HumanFormatter(desiredTraversalGroups, bufGroups, sizeof(bufGroups));
        HumanFormatter(ClusterLodPass::kMaxTraversalInfos, bufCap, sizeof(bufCap));
        ImGui::Text("Traversal nodes %s, groups %s / %s each", bufNodes, bufGroups, bufCap);
    }

    if (clasMemoryExceeded)
    {
        char bufDesired[64];
        char bufAllocated[64];
        MemoryFormatter(stats.desired.m_clasSize, bufDesired, sizeof(bufDesired));
        MemoryFormatter(stats.allocated.m_clasSize, bufAllocated, sizeof(bufDesired));
        ImGui::Text("CLAS %s / %s", bufDesired, bufAllocated);
    }

    if (clusterCountExceeded)
    {
        char bufDesired[64];
        char bufAllocated[64];
        HumanFormatter(stats.desired.m_numClusters, bufDesired, sizeof(bufDesired));
        HumanFormatter(stats.allocated.m_numClusters, bufAllocated, sizeof(bufDesired));
        ImGui::Text("Cluster Count %s/%s", bufDesired, bufAllocated);

        MemoryFormatter(stats.desired.m_clusterDataSize, bufDesired, sizeof(bufDesired));
        MemoryFormatter(stats.allocated.m_clusterDataSize, bufAllocated, sizeof(bufDesired));
        ImGui::Text("Cluster Data %s/%s", bufDesired, bufAllocated);
    }

    if (vertexMemoryExceeded)
    {
        char bufDesired[64];
        char bufAllocated[64];
        MemoryFormatter(stats.desired.m_vertexBufferSize, bufDesired, sizeof(bufDesired));
        MemoryFormatter(stats.allocated.m_vertexBufferSize, bufAllocated, sizeof(bufDesired));
        ImGui::Text("Vertex Buffer %s / %s", bufDesired, bufAllocated);
    }

    if (vertexNormalsMemoryExceeded)
    {
        char bufDesired[64];
        char bufAllocated[64];
        MemoryFormatter(stats.desired.m_vertexNormalsBufferSize, bufDesired, sizeof(bufDesired));
        MemoryFormatter(stats.allocated.m_vertexNormalsBufferSize, bufAllocated, sizeof(bufDesired));
        ImGui::Text("Vertex Normals Buffer %s / %s", bufDesired, bufAllocated);
    }
    ImGui::End();

    ImGui::PopStyleColor(2);  // Restore previous colors
}

void UserInterface::BuildUITimeline(int2 screenLayoutSize, float timelineWidth)
{
    auto& state = GetApp().GetUIData().timeLineEditorState;

    if (state.frameRate == 0.0f || (state.frameRange.y - state.frameRange.x) == 0)
        return;

    float tw = timelineWidth;
    ImGui::SetNextWindowPos(ImVec2(tw + 10.f, float(screenLayoutSize.y) - 10.f), 0,
        ImVec2(1.f, 1.f));
    ImGui::SetNextWindowSize(ImVec2(tw, 0.f));
    ImGui::SetNextWindowBgAlpha(.65f);
    if (ImGui::Begin("TimeLine Editor", nullptr,
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoTitleBar))
    {
        if (BuildTimeLineEditor(state, float2(tw, 0.f)))
        {
        }
    }
    ImGui::End();
}

void UserInterface::BuildUIEnvmap(ImVec2 itemSize)
{
    auto& renderer = m_app.GetRenderer();

    if (ImGui::CollapsingHeader("Environment Map", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (renderer.GetEffectiveShadingMode() == ShadingMode::PT)
        {
            ImGui::PushFont(m_iconicFont);

            if (ImGui::Button((char const*)(u8"\ue06b" "## env map"), { 0.f, itemSize.y }))
            {
                if (FileDialog(true, "All files\0*.*\0EXR files\0*.exr\0HDR files\0*.hdr\0\0", GetApp().GetUIData().envmapFilepath))
                {
                    const std::string filePath = GetApp().GetUIData().envmapFilepath;
                    if (!filePath.empty())
                        m_app.SetEnvmapTex(filePath);
                }
            }
            ImGui::PopFont();
            ImGui::SameLine();

            float buttonWidth = ImGui::GetItemRectSize().x + ImGui::GetStyle().ItemSpacing.x;
            ImGui::SetNextItemWidth(kItemWidth - buttonWidth);

            std::shared_ptr<engine::LoadedTexture> envmap = renderer.GetEnvMap();
            GetApp().GetUIData().envmapFilepath = "";
            GetApp().GetUIData().envmap = envmap;
            if (envmap)
            {
                GetApp().GetUIData().envmapFilepath = envmap->path;
            }

            char buf[1024] = { 0 };
            if (renderer.GetEnvMap() != nullptr)
            {
                std::strncpy(buf, renderer.GetEnvMap()->path.c_str(), std::size(buf));
            }
            if (ImGui::InputText("Env Map", buf, std::size(buf), ImGuiInputTextFlags_EnterReturnsTrue))
            {
                if (m_app.SetEnvmapTex(std::string(buf)))
                    GetApp().GetUIData().envmapFilepath = buf;
            }
            if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f && !GetApp().GetUIData().envmapFilepath.empty())
                ImGui::SetTooltip("Path to HDR environment map.");

            if (renderer.GetEnvMap() != nullptr)
            {
                static const char* debugGlyph = (char*)(u8"\ue028" "## envmap debug");

                bool debugView = !GetApp().GetUIData().envmapFilepath.empty() && renderer.GetEnableEnvmapHeatmap();

                ImGui::SameLine();
                if (debugView)
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.f, 0.f, 0.f, 1.f));
                ImGui::PushFont(m_iconicFont);
                if (ImGui::Button(debugGlyph, { 20.f, itemSize.y }))
                {
                    renderer.SetEnableEnvmapHeatmap(!debugView);
                }
                if (debugView)
                    ImGui::PopStyleColor();
                ImGui::PopFont();
                if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
                    ImGui::SetTooltip("Heatmap of The Envmap Impostance Sampling.");

                float intensity = renderer.GetEnvMapIntensity();
                if (ImGui::SliderFloat("Intensity", &intensity, .001f, 2.f))
                {
                    renderer.SetEnvMapIntensity(intensity);
                }
                if (ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f)
                    ImGui::SetTooltip("Intensity Scale");

                float azimuth = 180.f * renderer.GetEnvMapAzimuth() / M_PIf;
                if (ImGui::SliderFloat("Azimuth", &azimuth, 0.f, 360.f))
                {
                    renderer.SetEnvMapAzimuth((azimuth / 180.f) * M_PIf);
                }
                if (ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f)
                    ImGui::SetTooltip("Rotation Around Y Axis");

                float elevation = 180.f * renderer.GetEnvMapElevation() / M_PIf;
                if (ImGui::SliderFloat("Elevation", &elevation, -90.f, 90.f))
                {
                    renderer.SetEnvMapElevation((elevation / 180.f) * M_PIf);
                }
                if (ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f)
                    ImGui::SetTooltip("Rotation Around X Axis");
            }
            else
            {
                BuildMissColorUI();
            }
        }
        else
        {
            BuildMissColorUI();
        }
    }
}

void UserInterface::BuildMissColorUI()
{
    auto& renderer = m_app.GetRenderer();

    float3 missColor = renderer.GetMissColor();
    if (ImGui::ColorEdit3("Miss Color", &missColor.x, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel))
    {
        renderer.SetMissColor(missColor);
    }
}

void UserInterface::Animate(float elapsedTimeSeconds)
{
    ImGui_Renderer::Animate(elapsedTimeSeconds);

    if (GetApp().GetUIData().timeLineEditorState.IsPlaying())
    {
        GetApp().GetUIData().timeLineEditorState.Update(elapsedTimeSeconds);
    }
}

bool UserInterface::CustomInit(std::shared_ptr<engine::ShaderFactory> shaderFactory)
{
    // Runs BEFORE the scene exists (the load thread is still baking), so nothing
    // here may read GetScene() — see OnSceneLoaded() for that half.
    return Init(shaderFactory);
}

void UserInterface::OnSceneLoaded()
{
    const RTXMGScene::Attributes& attrs = m_app.GetScene().GetAttributes();

    SetAnimationRange(attrs.frameRange, attrs.frameRate);

    // Prefer the scene's embedded audio; otherwise use the per-load fallback set
    // by LoadAsset() (empty on the startup load, which clears any prior voice).
    if (!attrs.audio.empty())
        SetupAudioVoice(attrs.audio, GetApp().GetUIData().audioStartTime = attrs.audioStartTime);
    else
        SetupAudioVoice(m_pendingAudioFallbackPath, GetApp().GetUIData().audioStartTime = m_pendingAudioFallbackStart);

    // Consume the one-shot fallback so the next (non-menu) load doesn't reuse it.
    m_pendingAudioFallbackPath.clear();
    m_pendingAudioFallbackStart = 0.f;
}

bool UserInterface::BuildTimeLineEditor(TimeLineEditorState& state,
    float2 size)
{
    assert(state.startTime <= state.endTime);

    float fontScale = ImGui::GetIO().FontGlobalScale;

    static float const buttonPanelWidth =
        200 + fontScale * 230; // assumes a text font-m_size of ~ 14.f

    bool result = false;

    // current time slider

    ImGui::SetNextItemWidth(
        size.x -
        buttonPanelWidth); // anchor the button panel to the right of the window
    float currentTime = state.currentTime;
    if (ImGui::SliderFloat("##Time", &currentTime, state.startTime, state.endTime,
        "%.3f s."))
    {
        state.currentTime = clamp(currentTime, state.startTime, state.endTime);
        if (state.setTimeCallback)
            state.setTimeCallback(state);
        result = true;
    }
    ImVec2 sliderSize = ImGui::GetItemRectSize();
    ImGui::SameLine();

    // current frame number (editable)
    ImGui::SetNextItemWidth(fontScale * 45.f);
    float currentFrame = state.currentTime * state.frameRate;
    if (ImGui::InputFloat("##CurrentFrame", &currentFrame, 0.f, 0.f, "%.1f"))
    {
        state.currentTime =
            clamp(currentFrame / state.frameRate, state.startTime, state.endTime);
        if (state.setTimeCallback)
            state.setTimeCallback(state);
        result = true;
    }
    ImGui::SameLine();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Current frame of animation sequence.\n");

    // start & end frame numbers (read-only)
    float frameStart = state.startTime * state.frameRate;
    ImGui::SetNextItemWidth(fontScale * 45.f);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(.5f, .5f, .5f, 1.f));
    ImGui::InputFloat("##FrameStart", &frameStart, 0.f, 0.f, "%.1f",
        ImGuiInputTextFlags_ReadOnly);
    ImGui::PopStyleColor();
    ImGui::SameLine();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("First frame of animation sequence.\n");

    float frameEnd = state.endTime * state.frameRate;
    ImGui::SetNextItemWidth(fontScale * 45.f);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(.5f, .5f, .5f, 1.f));
    ImGui::InputFloat("##FrameEnd", &frameEnd, 0.f, 0.f, "%.1f",
        ImGuiInputTextFlags_ReadOnly);
    ImGui::PopStyleColor();
    ImGui::SameLine(0.f, 10.f);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Last frame of animation sequence.\n");

    // playback media buttons
    static const char* playGlyph = (char*)u8"\ue093";
    static const char* pauseGlyph = (char*)u8"\ue092";
    static const char* skip_backGlyph = (char*)u8"\ue097";
    static const char* skip_fwdGlyph = (char*)u8"\ue098";
    static const char* rewindGlyph = (char*)u8"\ue095";
    static const char* fast_fwdGlyph = (char*)u8"\ue096";
    static const char* repeatGlyph = (char*)u8"\ue08e";

    ImGui::PushFont(m_iconicFont);
    if (ImGui::Button(rewindGlyph, ImVec2(0.f, sliderSize.y)))
    {
        state.Rewind();
        result = true;
    }
    ImGui::SameLine();

    if (ImGui::Button(skip_backGlyph, ImVec2(0.f, sliderSize.y)))
    {
        state.StepBackward();
        result = true;
    }
    ImGui::SameLine();

    bool paused = state.IsPaused();
    if (paused)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.f, 0.f, 0.f, 1.f));
    if (ImGui::Button(paused ? playGlyph : pauseGlyph, { 0.f, sliderSize.y }))
    {
        state.PlayClicked();
    }
    if (paused)
        ImGui::PopStyleColor();

    ImGui::SameLine();

    if (ImGui::Button(skip_fwdGlyph, ImVec2(0.f, sliderSize.y)))
    {
        state.StepForward();
        result = true;
    }
    ImGui::SameLine();

    if (ImGui::Button(fast_fwdGlyph, ImVec2(0.f, sliderSize.y)))
    {
        state.FastForward();
        result = true;
    }
    ImGui::SameLine(0.f, 10.f);

    bool loop = state.loop;
    if (loop)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(.03f, .08f, .3f, 1.f));
    if (ImGui::Button(repeatGlyph, { 0.f, sliderSize.y }))
        state.loop = !loop;
    if (loop)
        ImGui::PopStyleColor();
    ImGui::PopFont();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Loop animation sequence.\n");

    ImGui::SameLine(0.f, 10.f);

    float frameRate = state.frameRate;
    ImGui::SetNextItemWidth(fontScale * 40.f);
    if (ImGui::InputFloat("##FrameRate", &frameRate, 0.f, 0.f, "%.1f"))
    {
        if (state.frameRange.y > state.frameRange.x && frameRate > 0)
        {
            state.startTime = float(state.frameRange.x) / frameRate;
            state.endTime = float(state.frameRange.y) / frameRate;
        }
        else if (frameRate == 0.f)
        {
            state.startTime = float(state.frameRange.x);
            state.endTime = float(state.frameRange.y);
        }
        state.frameRate = frameRate;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Animation frame rate (in frames per seconds).\n");

    return result;
}

void UserInterface::LoadAsset(MediaAsset const& asset, std::string const& name,
    int2 frameRange)
{
    bool isSequence = asset.IsSequence();

    const fs::path& mediapath = m_app.GetMediaPath();

    std::string shapePath =
        (mediapath /
            (isSequence ? fs::path(asset.sequenceFormat) : fs::path(name)))
        .generic_string();

    // Set the fallback BEFORE the load: OnSceneLoaded() consumes it, and that
    // can fire as soon as the load thread joins.
    m_pendingAudioFallbackPath  = asset.wavePath;
    m_pendingAudioFallbackStart = asset.waveStartTime;

    m_app.HandleSceneLoad(shapePath, mediapath.generic_string(), frameRange);

    GetApp().GetUIData().currentAsset = &asset;
}

void UserInterface::SetAnimationRange(int2 frameRange, float frameRate)
{
    float startTime = 0.f;
    float endTime = 0.f;

    if (frameRange.y > frameRange.x)
    {
        startTime = float(frameRange.x) / frameRate;
        endTime = float(frameRange.y) / frameRate;
    }
    else
    {
        assert(frameRate == 0.f);
        startTime = endTime = frameRate = 0.f;
    }
    auto& editor = GetApp().GetUIData().timeLineEditorState;
    editor.frameRange = frameRange;
    editor.startTime = startTime;
    editor.endTime = endTime;
    editor.currentTime = startTime;
    editor.frameRate = frameRate;
}

// null-terminated array of filter strings

std::array<char const*, 5> UIData::formatFilters() const
{
    std::array<char const*, 5> filters;

    std::fill(filters.begin(), filters.end(), nullptr);

    int idx = 0;
    if (includeJsonAssets)
        filters[idx++] = ".scene.json";
    if (includeObjAssets)
        filters[idx++] = ".obj";
    if (includeGltfAssets)
        filters[idx++] = ".gltf";

    assert(filters.back() == nullptr);

    return filters;
}

std::array<char const*, 5> UIData::folderFilters() const
{
    std::array<char const*, 5> filters;

    std::fill(filters.begin(), filters.end(), nullptr);

    int idx = 0;

    // Unused: way to filter which scenes are excluded from the scene selector
    // Example of use (exclude any files that include "do_not_show" in their path:
    // if (!includePrivateAssets)
    //     filters[idx++] = "do_not_show";

    assert(filters.back() == nullptr);

    return filters;
}

static inline bool isFolderFiltered(fs::path const& p,
    char const* const* filters)
{
    for (char const* const* filter = filters; *filter != nullptr; ++filter)
        if (p.generic_string().find(*filter) != std::string::npos)
            return true;
    return false;
}

// Matches the trailing suffix, not just the extension, so ".scene.json" can
// select loadable scenes while excluding other .json files (metadata.json etc.).
static inline bool isFormatFiltered(fs::path const& filename,
    char const* const* filters)
{
    std::string const name = filename.generic_string();
    for (char const* const* filter = filters; *filter != nullptr; ++filter)
    {
        std::string_view const suffix = *filter;
        if (name.size() >= suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            return true;
    }
    return false;
}

static void postProcessMediaAssets(MediaAssetsMap& assets)
{
    for (auto& asset : assets)
    {
        if (asset.first.find("barbarian") != std::string::npos)
        {
            asset.second.frameRate = 30.f;
        }
        else if (asset.first.find("rain_restaurant") != std::string::npos)
        {
            // Amy's monologue starts around frame 75, and we need to cut
            // some silence at the beginning.
            asset.second.waveStartTime = (100.f / 24.f) - 1.083f;
        }
    }
}
MediaAssetsMap
UserInterface::FindMediaAssets(fs::path const& mediapath,
    char const* const* folderFilters,
    char const* const* formatFilters)
{
    auto ToInt = [](std::string_view str) -> std::optional<int>
        {
            int value = 0;
            if (std::from_chars(str.data(), str.data() + str.size(), value).ec ==
                std::errc{})
                return value;
            return {};
        };

    auto IsPadded = [](std::string_view digits) { return digits[0] == '0'; };

    auto GetSequenceStr = [](std::string const& str) -> std::string_view
        {
            if (auto last = std::find_if(str.rbegin(), str.rend(), ::isdigit);
                last != str.rend())
                if (auto first = std::find_if(last, str.rend(),
                    [](char c) { return !std::isdigit(c); });
                    first != str.rend())
                    return { first.base(), last.base() };
            return {};
        };

    if (!fs::is_directory(mediapath))
        return {};

    MediaAssetsMap assets;

    auto InsertAsset = [&mediapath, &assets](fs::path const& rp,
        std::string const& name = {})
        {
            auto [it, success] =
                assets.insert({ name.empty() ? rp.generic_string() : name, {} });
            assert(success);
            it->second.name = it->first;
            return it;
        };

    auto opts = std::filesystem::directory_options::follow_directory_symlink;
    for (auto it = fs::recursive_directory_iterator(mediapath, opts);
        it != fs::recursive_directory_iterator(); ++it)
    {
        if (it->is_directory() && isFolderFiltered(it->path(), folderFilters))
            it.disable_recursion_pending();

        if (!isFormatFiltered(it->path().filename(), formatFilters))
        {
            continue;
        }

        fs::path rp = fs::relative(it->path(), mediapath).lexically_normal();

        std::string stem = rp.stem().generic_string();

        // The numbered-sequence collapsing below is an OBJ animation-frame
        // feature and hardcodes a ".obj" display name, so it must not see other
        // formats: a .gltf whose name ends in a digit would have that digit read
        // as a frame index and be relabeled ".obj", hiding the real file.
        bool const isObj = rp.extension() == ".obj";
        if (std::string_view seq = GetSequenceStr(stem); isObj && !seq.empty())
        {
            auto number = ToInt(seq);
            if (!number)
                continue;

            std::string name =
                (rp.parent_path() / std::string_view(stem.data(), seq.data()))
                .generic_string();

            auto it = assets.find(name);

            if (it == assets.end())
            {
                it = InsertAsset(rp, name);
            }

            if (IsPadded(seq))
                it->second.padding = std::max(it->second.padding, (int)seq.size());
            it->second.type = MediaAsset::Type::OBJ_SEQUENCE;
            it->second.GrowFrameRange(*number);
        }
        else
        {
            InsertAsset(rp);
        }
    }

    for (auto it = assets.begin(); it != assets.end();)
    {
        auto& asset = *it;

        if (asset.second.IsSequence())
        {
            char buf[1024];
            if (asset.second.frameRange.x < asset.second.frameRange.y)
            {
                std::snprintf(buf, std::size(buf), "%s[%d-%d].obj", asset.first.c_str(),
                    asset.second.frameRange.x, asset.second.frameRange.y);
                asset.second.sequenceName = buf;

                if (asset.second.padding > 0)
                    std::snprintf(buf, std::size(buf), "%s%%0%dd.obj",
                        asset.first.c_str(), asset.second.padding);
                else
                    std::snprintf(buf, std::size(buf), "%s%%d.obj", asset.first.c_str());
                asset.second.sequenceFormat = buf;

                asset.second.frameRate = 24.f;

                // check for a wave audio file
                std::snprintf(buf, std::size(buf), "%s%d.wav", asset.first.c_str(),
                    asset.second.frameRange.x);
                if (fs::is_regular_file(mediapath / buf))
                    it->second.wavePath = buf;

                it = std::next(it);
            }
            else // WAR for a single obj file whose name ends in a number being
                // mistaken for an animation keyframe
            {
                std::snprintf(buf, std::size(buf), "%s%d.obj", asset.first.c_str(),
                    asset.second.frameRange.x);
                MediaAsset asset = { .frameRange = {0, 0}, .frameRate = 0.f };
                std::swap(assets[buf], asset);
                it = assets.erase(it);
            }
        }
        else
        {
            // this could be a json file, which the GUI doesn't parse
            // so we need to set it to an invalid frame range
            it->second.frameRange = { std::numeric_limits<int>::max(), std::numeric_limits<int>::min() };
            it->second.frameRate = 0.f;
            it = std::next(it);
        }
    }

    postProcessMediaAssets(assets);

    return assets;
}

ImFont* UserInterface::AddFontFromMemoryCompressedBase85TTF(
    const char* data, float fontSize, const uint16_t* range)
{
    ImFontConfig fontConfig;
    fontConfig.MergeMode = false;
    fontConfig.FontDataOwnedByAtlas = false;
    ImFont* imFont = ImGui::GetCurrentContext()
        ->IO.Fonts->AddFontFromMemoryCompressedBase85TTF(
            data, fontSize, &fontConfig, (const ImWchar*)range);

    return imFont;
}

bool UserInterface::FolderDialog(std::string& m_filepath)
{
#ifdef _WIN32
    IFileOpenDialog* dlg;
    wchar_t* path = NULL;

    // Create the FileOpenDialog object.
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
        IID_IFileOpenDialog, (LPVOID*)&dlg);
    if (SUCCEEDED(hr))
    {
        FILEOPENDIALOGOPTIONS options;
        if (SUCCEEDED(dlg->GetOptions(&options)))
        {
            options |= FOS_PICKFOLDERS | FOS_PATHMUSTEXIST;
            dlg->SetOptions(options);
        }

        if (SUCCEEDED(dlg->Show(NULL)))
        {
            IShellItem* pItem;
            if (SUCCEEDED(dlg->GetResult(&pItem)))
            {
                hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &path);
                std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
                m_filepath = converter.to_bytes(path);

                pItem->Release();
            }
        }
        dlg->Release();
    }
    return true;
#else  // _WIN32
    // minimal implementation avoiding a GUI library, ignores filters for now,
    // and relies on external 'zenity' program commonly available on linuxoids
    char chars[PATH_MAX] = { 0 };
    std::string app = "zenity --file-selection --directory";
    FILE* f = popen(app.c_str(), "r");
    bool gotname = (nullptr != fgets(chars, PATH_MAX, f));
    pclose(f);

    if (gotname && chars[0] != '\0')
    {
        filepath = chars;

        // trim newline at end that zenity inserts
        filepath.erase(filepath.find_last_not_of(" \n\r\t") + 1);

        return true;
    }
    return false;
#endif // _WIN32
}

bool UserInterface::FileDialog(bool bOpen, char const* filters,
    std::string& m_filepath)
{
#ifdef _WIN32
    IFileOpenDialog* dlg;
    wchar_t* path = NULL;
    // Create the FileOpenDialog object.
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
        IID_IFileOpenDialog, (LPVOID*)&dlg);
    if (SUCCEEDED(hr))
    {
        auto parseTokens = [](char const* str)
            {
                std::vector<std::wstring> tokens;
                while (*str)
                {
                    if (size_t len = strlen(str); len > 0)
                    {
                        tokens.push_back(std::wstring(str, str + len));
                        str += len + 1;
                    }
                    else
                        break;
                }
                return tokens;
            };

        auto createFilterSpecs = [](std::vector<std::wstring> const& tokens)
            {
                assert((tokens.size() % 2) == 0);

                std::vector<COMDLG_FILTERSPEC> filterSpecs(tokens.size() / 2);
                for (uint8_t i = 0; i < tokens.size() / 2; ++i)
                    filterSpecs[i] = { .pszName = tokens[i * 2].c_str(),
                                      .pszSpec = tokens[i * 2 + 1].c_str() };
                return filterSpecs;
            };

        if (auto const& tokens = parseTokens(filters); !tokens.empty())
        {
            auto filterSpecs = createFilterSpecs(tokens);
            dlg->SetFileTypes((uint32_t)filterSpecs.size(), filterSpecs.data());
        }

        hr = dlg->Show(NULL);
        if (SUCCEEDED(hr))
        {
            IShellItem* pItem;
            hr = dlg->GetResult(&pItem);
            if (SUCCEEDED(hr))
            {
                hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &path);

                std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;

                m_filepath = converter.to_bytes(path);

                pItem->Release();
            }
        }
        dlg->Release();
    }
    return true;
#else  // _WIN32
    // minimal implementation avoiding a GUI library, ignores filters for now,
    // and relies on external 'zenity' program commonly available on linuxoids
    char chars[PATH_MAX] = { 0 };
    std::string app = "zenity --file-selection";
    if (!bOpen)
    {
        app += " --save --confirm-overwrite";
    }
    FILE* f = popen(app.c_str(), "r");
    bool gotname = (nullptr != fgets(chars, PATH_MAX, f));
    pclose(f);

    if (gotname && chars[0] != '\0')
    {
        filepath = chars;

        // trim newline at end that zenity inserts
        filepath.erase(filepath.find_last_not_of(" \n\r\t") + 1);

        return true;
    }
    return false;
#endif // _WIN32
}

void TimeLineEditorState::Update(float elapsedTime)
{
    currentTime += elapsedTime;

    if (currentTime > endTime)
    {
        if (loop)
        {
            currentTime = startTime;
            if (setTimeCallback)
                setTimeCallback(*this);
        }
        else
        {
            currentTime = endTime;
            mode = Playback::Pause;
            if (pauseCallback)
                pauseCallback(*this);
        }
    }
}

#ifdef AUDIO_ENGINE_ENABLED
void UserInterface::SetupAudioEngine()
{
    m_audioEngine = audio::Engine::create();

    GetApp().GetUIData().timeLineEditorState.playCallback = [this](TimeLineEditorState const& state)
        {
            if (m_voice)
                m_voice->start();
        };
    GetApp().GetUIData().timeLineEditorState.pauseCallback = [this](TimeLineEditorState const& state)
        {
            if (m_voice)
                m_voice->stop();
        };
    GetApp().GetUIData().timeLineEditorState.setTimeCallback = [this](TimeLineEditorState const& state)
        {
            if (m_voice)
            {
                m_voice->stop();
                m_voice->setStart(*m_audioEngine, state.currentTime - GetApp().GetUIData().audioStartTime);
                if (state.IsPlaying())
                    m_voice->start();
            }
        };
}

void UserInterface::SetupAudioVoice(const std::string& wavepath, float startTime)
{
    auto& state = GetApp().GetUIData().timeLineEditorState;

    if (!m_audioEngine)
        return;

    if (wavepath.empty())
    {
        if (m_voice)
        {
            m_voice->stop();
            m_voice.reset();
        }
        return;
    }

    const fs::path& mediapath = m_app.GetMediaPath();

    // Resolve like 'models' and 'envmap' do: scene-file-relative first, then the
    // media root (the media-browser fallback path is media-root-relative).
    const fs::path sceneDir = fs::path(m_app.GetScene().GetInputPath()).parent_path();
    fs::path filepath = rtxmg::ResolveMediapath(sceneDir / wavepath, mediapath);
    if (filepath.empty())
        filepath = rtxmg::ResolveMediapath(wavepath, mediapath);
    if (filepath.empty())
    {
        donut::log::warning("Audio '%s' not found relative to the scene file or the media path '%s'.",
                            wavepath.c_str(), mediapath.generic_string().c_str());
        return;
    }

    std::shared_ptr<audio::WaveFile> wavefile = audio::WaveFile::read(filepath);
    if (!wavefile)
        return;

    m_voice = audio::Voice::create(*m_audioEngine, wavefile);
    if (!m_voice)
        return;

    float offset = state.currentTime - startTime;
    m_voice->setStart(*m_audioEngine, offset);
}

void UserInterface::MuteAudio(bool mute)
{
    GetApp().GetUIData().audioMuted = mute;
    if (m_audioEngine)
        m_audioEngine->mute(GetApp().GetUIData().audioMuted);
}

void UserInterface::BuildGridInstancingWindow()
{
    UIData& uiData = m_app.GetUIData();
    if (!uiData.showGridInstancingWindow)
    {
        m_gridInstancingWindowOpen = false;
        return;
    }

    // Rising-edge: initialize staged values from the current live values.
    if (!m_gridInstancingWindowOpen)
    {
        m_stagedGridCopies    = m_app.GetLodGridCopies();
        m_stagedGridGap       = m_app.GetLodGridGap();
        m_stagedGridRandomize = m_app.GetLodGridRandomize();
        m_gridInstancingWindowOpen = true;
    }

    ImGui::SetNextWindowSize(ImVec2(300, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(220, 80), ImGuiCond_FirstUseEver);
    bool open = true;
    if (!ImGui::Begin("Grid Instancing", &open))
    {
        ImGui::End();
        if (!open) uiData.showGridInstancingWindow = false;
        return;
    }
    if (!open)
    {
        uiData.showGridInstancingWindow = false;
        ImGui::End();
        return;
    }

    ImGui::TextDisabled("Changes require a scene reload (Apply).");
    ImGui::Spacing();

    ImGui::PushItemWidth(120.f);
    int copies = int(m_stagedGridCopies);
    if (ImGui::InputInt("Grid Copies", &copies, 1, 64))
        m_stagedGridCopies = uint32_t(std::max(1, copies));
    if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
        ImGui::SetTooltip("Number of instance copies to spawn on the grid\n"
                          "(cluster-LoD and subd meshes).");

    if (ImGui::SliderFloat("Grid Gap", &m_stagedGridGap, 0.0f, 4.0f, "%.3f"))
        m_stagedGridGap = std::max(0.0f, m_stagedGridGap);
    if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
        ImGui::SetTooltip("Grid spacing as a multiple of the model AABB extent.");
    ImGui::PopItemWidth();

    ImGui::Checkbox("Randomize Rotations", &m_stagedGridRandomize);
    if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
        ImGui::SetTooltip("Apply a random rotation to each grid copy.");

    ImGui::Spacing();
    if (ImGui::Button("Apply"))
    {
        m_app.SetLodGridCopies(m_stagedGridCopies);
        m_app.SetLodGridGap(m_stagedGridGap);
        m_app.SetLodGridRandomize(m_stagedGridRandomize);
        m_app.ReloadCurrentScene();
        uiData.showGridInstancingWindow = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel"))
        uiData.showGridInstancingWindow = false;

    ImGui::End();
}

#else
void UserInterface::SetupAudioEngine() {}
void UserInterface::SetupAudioVoice(const std::string& wavepath, float startTime) {}
void UserInterface::MuteAudio(bool) {}
#endif
