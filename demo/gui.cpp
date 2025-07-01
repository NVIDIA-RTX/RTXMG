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
#include "implot.h"

#include <charconv>
#include <filesystem>
#include <filesystem>
#include <algorithm>
#include <string>
#include <sstream>

#include <donut/app/imgui_renderer.h>
#include "rtxmg/scene/scene.h"
#include "rtxmg/utils/constants.h"
#include "rtxmg/utils/formatters.h"
#include "rtxmg/cluster_builder/tessellator_constants.h"

#include "korgi.h"
#include "rtxmg/profiler/gui.h"
#include "rtxmg/profiler/statistics.h"

#include <donut/app/StreamlineInterface.h>

namespace fs = std::filesystem;

using namespace donut;

#define UI_RED ImVec4(1.f, 0.f, 0.f, 1.f)
#define UI_SAGE ImVec4(.3f, .4f, .35f, 1.f)
#define UI_PLUM ImVec4(.4f, .3f, .35f, 1.f)

constexpr float kItemWidth = 200.0f;

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

        bool displayStats = false;

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

            gui->MuteAudio(settings.audioMuted);

            UIData& ui = app->GetUIData();
            ui.includeJsonAssets = settings.jsonAssetsFilter;
            ui.includeObjAssets = settings.objAssetsFilter;

            const fs::path& mediaPath = app->GetMediaPath();
            if (!mediaPath.empty())
            {
                assert(ui.mediaAssets.empty());
                auto folder_filters = ui.folderFilters();
                auto format_filters = ui.formatFilters();
                ui.mediaAssets = FindMediaAssets(mediaPath, folder_filters.data(), format_filters.data());
            }

            app->SetWindowState(settings.windowState);

            settings.wantApply = false;
        }
    };


    ini_handler.ReadLineFn = [](ImGuiContext*, ImGuiSettingsHandler* handler, void* entry, const char* line) {

        int audioMuted = 0;
        if (std::sscanf(line, "AudioMuted=%d", &audioMuted) == 1)
            settings.audioMuted = audioMuted;

        uint32_t json = 0, obj = 0;
        if (std::sscanf(line, "FormatFilters={ json=%d, obj=%d }", &json, &obj) == 2)
        {
            settings.jsonAssetsFilter = (bool)json;
            settings.objAssetsFilter = (bool)obj;
        }

        int displayStats = false;
        if (std::sscanf(line, "DisplayStatistics=%d", &displayStats) == 1)
            settings.displayStats = displayStats != 0;

        donut::math::int2 windowSize{};
        if (std::sscanf(line, "WindowSize=%d,%d", &windowSize.x, &windowSize.y) == 2)
        {
            settings.windowState.windowSize = windowSize;
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
        settings.objAssetsFilter = ui.includeObjAssets;
        settings.displayStats = gui->GetProfilerGUI().displayGraphWindow;
        
        settings.windowState = app->GetWindowState();
        
        buf->reserve(buf->size() + 2);  // ballpark reserve
        buf->appendf("[%s][%s]\n", handler->TypeName, "Settings");
        buf->appendf("AudioMuted=%d\n", settings.audioMuted);
        buf->appendf("FormatFilters={ json=%d, obj=%d }\n", settings.jsonAssetsFilter, settings.objAssetsFilter);
        buf->appendf("DisplayStatistics=%d\n", settings.displayStats);
        buf->appendf("WindowSize=%d,%d\n", settings.windowState.windowSize.x, settings.windowState.windowSize.y);
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

    BuildUIMain({ width, height });

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

void UserInterface::BuildUIMain(int2 screenLayoutSize)
{
    UIData& uiData = GetApp().GetUIData();

    auto& renderer = m_app.GetRenderer();
#ifdef _WIN32
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
#endif
    ImVec2 itemSize = ImGui::GetItemRectSize();

    const char* kWindowName = "Settings";
    SetConstrainedWindowPos(kWindowName, ImVec2(10, 10), ImVec2(0.0f, 0.0f), MakeImVec2(screenLayoutSize));
    ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSizeConstraints(ImVec2(100.f, 200.f), ImVec2(float(screenLayoutSize.x), screenLayoutSize.y - 70.0f));
    ImGui::Begin(kWindowName, nullptr, ImGuiWindowFlags_None);

    ImGui::PushItemWidth(kItemWidth);

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

    ImGui::SameLine();
    bool showMicroTriangles = renderer.GetShowMicroTriangles();

    float buttonWidth = ImGui::GetTextLineHeightWithSpacing() + ImGui::CalcTextSize("Visualize Micro Triangles").x
        + ImGui::GetStyle().FramePadding.x * 2.0f;
    ImGui::SetNextItemWidth(buttonWidth);

    if (showMicroTriangles)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.0f, 0.0f, 1.0f));
    if (ImGui::Button("Micro Triangles"))
    {
        renderer.SetShowMicroTriangles(!showMicroTriangles);
    }
    if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
        ImGui::SetTooltip("Toggle micro triangle visualization mode with a unique color per triangle id.");
    if (showMicroTriangles)
        ImGui::PopStyleColor();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_Header, UI_SAGE);
    if (ImGui::CollapsingHeader("Scene Loading", ImGuiTreeNodeFlags_DefaultOpen))
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

        if (ImGui::Checkbox("Json", &uiData.includeJsonAssets))
            objFilesNeedUpdate = true;
        ImGui::SameLine();
        if (ImGui::Checkbox("Obj", &uiData.includeObjAssets))
            objFilesNeedUpdate = true;

        if (objFilesNeedUpdate || uiData.mediaAssets.empty())
        {
            auto folderFilters = uiData.folderFilters();
            auto formatFilters = uiData.formatFilters();

            uiData.mediaAssets = FindMediaAssets(mediapath, folderFilters.data(),
                formatFilters.data());
        }

        char const* currentAssetName =
            uiData.currentAsset ? uiData.currentAsset->GetName() : nullptr;
        if (ImGui::BeginCombo("Scene", currentAssetName,
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
            uiData.currentAsset->name)
        {
            ImGui::SetTooltip("%s", uiData.currentAsset->name);
        }
    }

#if RTXMG_DEV_FEATURES
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

        ImGui::SameLine();
        bool accelBuildLoggingEnabled = m_app.GetAccelBuildLoggingEnabled();
        if (ImGui::Checkbox("AS Log", &accelBuildLoggingEnabled))
        {
            m_app.SetAccelBuildLoggingEnabled(accelBuildLoggingEnabled);
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Log accel build buffers (Syncs GPU, Slow!)");

        ImGui::Checkbox("Build AS", &uiData.forceRebuildAccelStruct);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip(
                "Re-tessellate and rebuild acceleration structures every frame.\n"
                "Disable to freeze tessellation and move the camera around.\n"
                "Animation will always force a rebuild.");

        ImGui::SameLine();
        ImGui::Checkbox("Monolithic ClusterBuild", &uiData.enableMonolithicClusterBuild);
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip(
                "Use a single shader for compute cluster tiling and fill clusters.\n"
                "Instead of splitting the dispatches by surface type");

#if ENABLE_PIXEL_DEBUG
        int2& debugPixel = renderer.GetDebugPixel();
        ImGui::InputInt2("DebugPixel (Right-click)", debugPixel.data());
#endif
    }
#endif

    if (!showMicroTriangles && ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
    {   
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
        if (ImGuiComboFromArray("Color Mode", &colorMode, kColorModeNames))
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
            if (ImGui::SliderFloat("Exposure", &exposure, 0.f, 1000.f, "%.3f", ImGuiSliderFlags_Logarithmic))
            {
                renderer.SetExposure(exposure);
            }

            TonemapOperator tonemap = renderer.GetTonemapOperator();
            if (ImGuiComboFromArray("Tonemapping Operator", &tonemap, kToneMapOperatorNames))
            {
                renderer.SetTonemapOperator(tonemap);
            }
        }

        BuildUIEnvmap(itemSize);
    }
    ImGui::PopStyleColor();
    ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_Header, UI_SAGE);
    if (ImGui::CollapsingHeader("Tessellation", ImGuiTreeNodeFlags_DefaultOpen))
    {
        TessellatorConfig::MemorySettings memSettings = m_app.GetTessMemSettings();

        int32_t maxKClusters = memSettings.maxClusters >> 10u;
        int vertexMB = int32_t(memSettings.vertexBufferBytes >> 20ull);
        int clasMB = int32_t(memSettings.clasBufferBytes >> 20ull);

        auto& stats = GetApp().m_BuildStats;

        bool memSettingsChanged = false;

        const float kFlashWarningPeriod = 1.0f;
        float t = fmodf(float(ImGui::GetTime()), kFlashWarningPeriod);  // Flashes every second (0.5s on, 0.5s off)
        bool shouldHighlight = (t < 0.5f);

        bool highlightMaxClusters = shouldHighlight && stats.desired.m_numClusters > stats.allocated.m_numClusters;
        bool highlightVertexMemory = shouldHighlight && stats.desired.m_vertexBufferSize > stats.allocated.m_vertexBufferSize;
        bool highlightClasMemory = shouldHighlight && stats.desired.m_clasSize > stats.allocated.m_clasSize;

        const ImVec4 kHighlightColor = ImVec4(0.5f, 0.0f, 0.0f, 1.0f); // Red highlight
        if (highlightMaxClusters)
            ImGui::PushStyleColor(ImGuiCol_FrameBg, kHighlightColor);
        if (ImGui::InputInt("Max Clusters (K)", &maxKClusters, 64, 256, ImGuiInputTextFlags_EnterReturnsTrue))
        {
            memSettings.maxClusters = std::min((uint32_t(std::max(maxKClusters, 64)) << 10u), kMaxApiClusterCount);
            memSettingsChanged = true;
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Max clusters in a scene which affects the size of cluster data buffer and BLAS memory");
        if (highlightMaxClusters)
            ImGui::PopStyleColor();

        if (highlightVertexMemory)
            ImGui::PushStyleColor(ImGuiCol_FrameBg, kHighlightColor);
        if (ImGui::InputInt("Vertex Memory (MB)", &vertexMB, 128, 512, ImGuiInputTextFlags_EnterReturnsTrue))
        {
            memSettings.vertexBufferBytes = std::min(size_t(std::max(vertexMB, 128)), size_t(8192)) << 20ull;
            memSettingsChanged = true;
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Max memory in MB allocated for tessellated vertices");
        if (highlightVertexMemory)
            ImGui::PopStyleColor();

        if (highlightClasMemory)
            ImGui::PushStyleColor(ImGuiCol_FrameBg, kHighlightColor);
        if (ImGui::InputInt("CLAS Memory (MB)", &clasMB, 128, 512, ImGuiInputTextFlags_EnterReturnsTrue))
        {
            memSettings.clasBufferBytes = std::min(size_t(std::max(clasMB, 128)), size_t(8192)) << 20ull;
            memSettingsChanged = true;
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Max memory in megabytes allocated for cluster acceleration structures CLAS");
        if (highlightClasMemory)
            ImGui::PopStyleColor();

        if (memSettingsChanged)
        {
            m_app.SetTessMemSettings(memSettings);
        }

        int   clusterPattern = static_cast<int>(m_app.GetClusterTessellationPattern());
        float comboBoxWidth = ImGui::GetTextLineHeightWithSpacing() + ImGui::CalcTextSize("Slanted  ").x
            + ImGui::GetStyle().FramePadding.x * 2.0f;
        ImGui::SetNextItemWidth(comboBoxWidth);
        if (ImGui::Combo("Tess Pattern", &clusterPattern, "Regular\0Slanted\0"))
        {
            m_app.SetClusterTessellationPattern(ClusterPattern(clusterPattern));
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip(
                "Toggles the 'slanted' grid pattern on. 'Slanted' grids\n"
                "allow for a smoother transition when the number of edge segments on\n"
                "opposite sides of a quad don't match.\n\n");

        ImGui::SameLine();
        bool updateTessCamera = m_app.GetUpdateTessellationCamera();
        if (ImGui::Checkbox("Update Tess Camera", &updateTessCamera))
        {
            m_app.SetUpdateTessellationCamera(updateTessCamera);
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Keep tessellation camera in sync. Uncheck to lock the camera.");

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

        {
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
        }

        float displacementScale = m_app.GetDisplacementScale();
        if (ImGui::SliderFloat("Displacement Scale", &displacementScale, 0.0f, 3.0f))
        {
            m_app.SetDisplacementScale(displacementScale);
        }
        if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Scaling factor for displacement maps");
    }
    ImGui::PopStyleColor();

    ImGui::Spacing();

#if DONUT_WITH_STREAMLINE
    ImGui::PushStyleColor(ImGuiCol_Header, UI_SAGE);
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
                    std::array<const char*, 7> kDlssPresetNames = {
                        "Default",
                        "Preset A",
                        "Preset B",
                        "Preset C",
                        "Preset D",
                        "Preset E",
                        "Preset F"
                    };

                    std::array<const char*, 7> kDlssRRPresetNames = {
                        "Default",
                        "Preset A",
                        "Preset B",
                        "Preset C",
                        "Preset D",
                        "Preset E",
                        "Preset G"
                    };

                    if (denoiserMode == DenoiserMode::DlssSr)
                    {
                        if (ImGui::BeginCombo("DLSS SR Preset", kDlssPresetNames[(int)uiData.dlssPreset]))
                        {
                            for (int i = 0; i < kDlssPresetNames.size(); ++i)
                            {
                                bool isSelected = i == static_cast<int>(uiData.dlssPreset);

                                if (ImGui::Selectable(kDlssPresetNames[i], isSelected)) uiData.dlssPreset = (StreamlineInterface::DLSSPreset)i;
                                if (isSelected) ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }
                    }
                    else
                    {
                        if (ImGui::BeginCombo("DLSS RR Preset", kDlssRRPresetNames[(int)uiData.dlssRRPreset]))
                        {
                            for (int i = 0; i < kDlssRRPresetNames.size(); ++i)
                            {
                                bool isSelected = i == static_cast<int>(uiData.dlssRRPreset);

                                if (ImGui::Selectable(kDlssRRPresetNames[i], isSelected)) uiData.dlssRRPreset = (StreamlineInterface::DLSSRRPreset)i;
                                if (isSelected) ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }
                    }
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
    ImGui::PopStyleColor();
    ImGui::Spacing();
#endif
    ImGui::PopItemWidth();
    ImGui::End();

    m_profiler.fps = (int)(1000.f / (float)m_app.GetCPUFrameTime());
    m_profiler.desiredTris = stats::clusterAccelSamplers.numTriangles.latest;
    m_profiler.allocatedTris = stats::clusterAccelSamplers.numTriangles.max;
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

    m_profiler.BuildUI<stats::FrameSamplers, stats::ClusterAccelSamplers, stats::EvaluatorSamplers, stats::MemUsageSamplers>(m_iconicFont, m_implot,
        stats::frameSamplers, stats::clusterAccelSamplers, stats::evaluatorSamplers, stats::memUsageSamplers);

    if (stats::evaluatorSamplers.m_topologyQualityButtonPressed)
    {
        renderer.SetColorMode(ColorMode::COLOR_BY_TOPOLOGY);
    }

    float profilerWidth = m_profiler.controllerWindow.size.x;
    float timelineWidth = float(screenLayoutSize.x) - 30.f - profilerWidth;

    BuildUITimeline(screenLayoutSize, timelineWidth);

    BuildMemoryWarning(screenLayoutSize);
}

void UserInterface::BuildMemoryWarning(int2 screenLayoutSize)
{
    auto& stats = GetApp().m_BuildStats;
    bool clusterCountExceeded = stats.desired.m_numClusters > stats.allocated.m_numClusters;
    bool clasMemoryExceeded = stats.desired.m_clasSize > stats.allocated.m_clasSize;
    bool vertexMemoryExceeded = stats.desired.m_vertexBufferSize > stats.allocated.m_vertexBufferSize;

    if (!clusterCountExceeded && !clasMemoryExceeded && !vertexMemoryExceeded)
        return;

    ImVec2 overlayPos(screenLayoutSize.x * 0.5f, 10.0f); // Center X, 10px from the top
    ImVec2 overlayPivot(0.5f, 0.0f); // Center horizontally, stick to the top

    ImGui::SetNextWindowPos(overlayPos, ImGuiCond_Always, overlayPivot);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, IM_COL32(128, 0, 0, 200));  // Dark red
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 255));    // White text

    ImGui::Begin("Memory Exceeded", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                  ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::PushFont(m_nvidiaBldFont);
    ImGui::Text("Tessellation memory budget exceeded");
    ImGui::PopFont();
    ImGui::Text("Expect flickering. Increase memory budget in settings");

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
    // guaranteed to be called after first scene is loaded, so the attributes are available.


    const RTXMGScene::Attributes& attrs = m_app.GetScene().GetAttributes();

    SetAnimationRange(attrs.frameRange, attrs.frameRate);

    if (!attrs.audio.empty())
        SetupAudioVoice(attrs.audio, GetApp().GetUIData().audioStartTime = attrs.audioStartTime);

    return Init(shaderFactory);
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

    m_app.HandleSceneLoad(shapePath, mediapath.generic_string(), frameRange);

    const RTXMGScene::Attributes& attrs = m_app.GetScene().GetAttributes();

    SetAnimationRange(attrs.frameRange, attrs.frameRate);

    if (!attrs.audio.empty())
        SetupAudioVoice(attrs.audio, GetApp().GetUIData().audioStartTime = attrs.audioStartTime);
    else
        SetupAudioVoice(asset.wavePath, GetApp().GetUIData().audioStartTime = asset.waveStartTime);

    GetApp().GetUIData().showRecommendationWindow = true;

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

std::array<char const*, 4> UIData::formatFilters() const
{
    std::array<char const*, 4> filters;

    std::fill(filters.begin(), filters.end(), nullptr);

    int idx = 0;
    if (includeJsonAssets)
        filters[idx++] = ".json";
    if (includeObjAssets)
        filters[idx++] = ".obj";

    //Future: include unstructured assets
    //if (includeEddAssets)
    //    filters[idx++] = ".eddbin";

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

static inline bool isFormatFiltered(fs::path const& ext,
    char const* const* filters)
{
    for (char const* const* filter = filters; *filter != nullptr; ++filter)
        if (ext == *filter)
            return true;
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
            it->second.name = it->first.c_str();
            return it;
        };

    auto opts = std::filesystem::directory_options::follow_directory_symlink;
    for (auto it = fs::recursive_directory_iterator(mediapath, opts);
        it != fs::recursive_directory_iterator(); ++it)
    {
        if (it->is_directory() && isFolderFiltered(it->path(), folderFilters))
            it.disable_recursion_pending();

        if (!isFormatFiltered(it->path().extension(), formatFilters))
        {
            continue;
        }

        fs::path rp = fs::relative(it->path(), mediapath).lexically_normal();

        std::string stem = rp.stem().generic_string();

        if (std::string_view seq = GetSequenceStr(stem); !seq.empty())
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

bool UserInterface::FolderDialog(std::string& filepath)
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
                filepath = converter.to_bytes(path);

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
    std::string& filepath)
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

                filepath = converter.to_bytes(path);

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

    std::shared_ptr<audio::WaveFile> wavefile = audio::WaveFile::read(mediapath / wavepath);
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

#else
void UserInterface::SetupAudioEngine() {}
void UserInterface::SetupAudioVoice(const std::string& wavepath, float startTime) {}
void UserInterface::MuteAudio(bool) {}
#endif
