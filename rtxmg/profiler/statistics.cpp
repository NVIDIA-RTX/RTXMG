/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <numeric>
#include <opensubdiv/tmr/topologyMap.h>

#include <cmath>
#include <locale>
#include <string>
#include <sstream>

#include <imgui_internal.h>
#include <implot.h>

#include "rtxmg/cluster_lod/pass.h"
#include "rtxmg/profiler/gui.h"
#include "rtxmg/profiler/statistics.h"
#include "rtxmg/utils/buffer.h"
#include "rtxmg/utils/formatters.h"

namespace stats {

    constexpr ImPlotAxisFlags flags = ImPlotAxisFlags_AutoFit;
    constexpr double          yref = 0;  // std::numeric_limits<double>::lowest();
    constexpr double          xscale = 1.0;
    constexpr double          xstart = 0;

    FrameSamplers        frameSamplers;

    void ProfileFrameTimers()
    {
        frameSamplers.gpuFrameTime.Profile();
        frameSamplers.gpuRenderTime.Profile();
        frameSamplers.gpuDenoiserTime.Profile();
        frameSamplers.blitTime.Profile();
    }

    ClusterAccelSamplers clusterAccelSamplers;
    ClusterTessTab       clusterTessTab;
    ClusterLodTab        clusterLodTab;
    MemUsageSamplers     memUsageSamplers;
    BakeStats            bakeStats;
    EvaluatorSamplers evaluatorSamplers;
    StreamingSamplers    streamingSamplers;
    VramBreakdown        vramBreakdown;

    void FrameSamplers::BuildUI(ImFont *iconicFont, ImPlotContext *plotContext) const
    {
        constexpr int stride = (int)sizeof(float);

        const float fontScale = ImGui::GetIO().FontGlobalScale;

        enum class GraphMode : int {
            Overview = 0,
            HiZ
        };
        static GraphMode mode = GraphMode::Overview;
        ImGui::PushItemWidth(125);
        ImGui::Combo("Graph modes", reinterpret_cast<int*>(&mode), "Overview\0Hierarchical-Z\0");
        ImGui::PopItemWidth();

        std::array<GPUTimer*, 3> timers = { nullptr, nullptr, nullptr };

        // Accel-build running-average totals for the per-stage text breakdown
        // below; only filled in Overview mode.
        float tessAvgMs = 0.f, lodAvgMs = 0.f, tlasAvgMs = 0.f;
        bool  hasAccel  = false;

        switch (mode)
        {
        case GraphMode::Overview: {
            timers[0] = &gpuFrameTime.Profile();
            timers[1] = &gpuRenderTime.Profile();
            timers[2] = &gpuDenoiserTime.Profile();
            computeMotionVectorsTimer.Profile();
            // Resolved in this mode too so the breakdown below accounts for every pass.
            zRenderPassTime.Profile();
            hiZRenderTime.Profile();
            blitTime.Profile();

            // Cluster Tess and Cluster LOD build sequentially each frame, so their
            // phase sub-timers sum into one accel-build series.
            auto& cas = clusterAccelSamplers;
            hasAccel  = cas.hasClusterTess || cas.hasClusterLod;
            auto last   = [](GPUTimer& t) { return t.Profile().latest; };
            auto avgNan = [](GPUTimer& t) { float v = t.RunningAverage(); return std::isnan(v) ? 0.f : v; };
            // Only sum a path that actually ran: a path that never dispatched also
            // never Start/Stop'd its timers, so Profile() resolves garbage queries.
            float tessLast = 0.f, lodLast = 0.f;
            if (cas.hasClusterTess)
            {
                tessLast  = last(cas.clusterTilingTime) + last(cas.fillClustersTime)
                          + last(cas.buildClasTime) + last(cas.buildBlasTime);
                tessAvgMs = avgNan(cas.clusterTilingTime) + avgNan(cas.fillClustersTime)
                          + avgNan(cas.buildClasTime) + avgNan(cas.buildBlasTime);
            }
            if (cas.hasClusterLod)
            {
                lodLast  = last(cas.clusterLodTraversalTime) + last(cas.clusterLodClasBuildTime)
                         + last(cas.clusterLodClasMovePersistentTime) + last(cas.clusterLodClasMoveCompactionTime)
                         + last(cas.clusterLodAllocUnloadUpdateTime) + last(cas.clusterLodAllocFreegapsTime)
                         + last(cas.clusterLodAllocAgeTime) + last(cas.clusterLodAllocLoadTime)
                         + last(cas.clusterLodAllocStatusTime) + last(cas.clusterLodBlasBuildTime)
                         + last(cas.clusterLodUploadTime);
                lodAvgMs = avgNan(cas.clusterLodTraversalTime) + avgNan(cas.clusterLodClasBuildTime)
                         + avgNan(cas.clusterLodClasMovePersistentTime) + avgNan(cas.clusterLodClasMoveCompactionTime)
                         + avgNan(cas.clusterLodAllocUnloadUpdateTime) + avgNan(cas.clusterLodAllocFreegapsTime)
                         + avgNan(cas.clusterLodAllocAgeTime) + avgNan(cas.clusterLodAllocLoadTime)
                         + avgNan(cas.clusterLodAllocStatusTime) + avgNan(cas.clusterLodBlasBuildTime)
                         + avgNan(cas.clusterLodUploadTime);
            }
            // TLAS refresh + instance-desc fill runs for both paths.
            const float tlasLast = hasAccel ? last(cas.tlasBuildTime) : 0.f;
            tlasAvgMs = hasAccel ? avgNan(cas.tlasBuildTime) : 0.f;
            if (cas.hasClusterLod)
            {
                cas.clusterLodSubmitIdleTime.Profile();
                cas.clusterLodHostTime.Profile();
            }
            if (Profiler::Get().IsRecording())
                accelBuildTime.PushBack(tessLast + lodLast + tlasLast);
        } break;
        case GraphMode::HiZ: {
            timers[0] = &zRenderPassTime.Profile();
            timers[1] = &zReprojectionTime.Profile();
            timers[2] = &hiZRenderTime.Profile();
        } break;
        default:
            return;
        }

        // Implot appears to be having issues if the arrays of data have different sizes & offsets
        assert(timers[0]->size() == timers[1]->size() && timers[0]->size() == timers[2]->size());

        if (ImPlot::BeginPlot("##timers2", ImVec2(-1, 150 * fontScale)))
        {
            ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(timers[0]->size()), ImGuiCond_Always);

            constexpr bool autofit = false;

            if constexpr (autofit)
            {
                // ImPlot autofit is a little wiggly - exploring alternatives below
                ImPlot::SetupAxis(ImAxis_Y1, timers[0]->name.c_str(), ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_AuxDefault);
                ImPlot::SetupAxis(ImAxis_Y2, timers[1]->name.c_str(), ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_AuxDefault);
                ImPlot::SetupAxis(ImAxis_Y3, timers[2]->name.c_str(), ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_AuxDefault);
            }
            else
            {
                ImPlot::SetupAxis(ImAxis_Y1, "Time (ms)");
                ImPlot::SetupAxis(ImAxis_Y2, "##hidden1", ImPlotAxisFlags_NoDecorations);
                ImPlot::SetupAxis(ImAxis_Y3, "##hidden2", ImPlotAxisFlags_NoDecorations);

                // Scale to the larger of GPU/CPU frame so the CPU line stays on
                // screen exactly when it matters (CPU-bound).
                float vmax = std::max(timers[0]->RunningAverage(),
                                      mode == GraphMode::Overview ? cpuFrameTime.RunningAverage() : 0.f) * 1.75f;
                if (vmax < 1e-6)
                    vmax = timers[1]->RunningAverage() * 1.75f;

                // Constrain both Y axes to the same range for better readability
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0., vmax, ImPlotCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y2, 0., vmax, ImPlotCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y3, 0., vmax, ImPlotCond_Always);
            }


            for (uint8_t i = 0; i < timers.size(); ++i)
            {
                if (!timers[i])
                    continue;

                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1 + i);

                ImPlot::PlotLine(timers[i]->name.c_str(), timers[i]->data(), (int)timers[i]->size(),
                    xscale, xstart, ImPlotShadedFlags_None, timers[i]->Offset(), stride);
            }

            if (mode == GraphMode::Overview)
            {
                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
                if (hasAccel)
                    ImPlot::PlotLine(accelBuildTime.name.c_str(), accelBuildTime.data(), (int)accelBuildTime.size(),
                        xscale, xstart, ImPlotShadedFlags_None, (int)accelBuildTime.Offset(), stride);
                ImPlot::PlotLine(cpuFrameTime.name.c_str(), cpuFrameTime.data(), (int)cpuFrameTime.size(),
                    xscale, xstart, ImPlotShadedFlags_None, (int)cpuFrameTime.Offset(), stride);
            }

            ImPlot::EndPlot();
            if (ImGui::IsItemHovered())
            {
                switch (mode)
                {
                case GraphMode::Overview:
                    ImGui::SetTooltip(
                        "CPU frame: %.4fms (above GPU frame = CPU-bound)\n\n"
                        "GPU timers:\n"
                        "  - Frame: %.4fms\n"
                        "  - Accel Build: %.4fms\n"
                        "  - Pathtrace: %.4fms\n"
                        "  - Motion Vectors: %.4fms\n"
                        "  - Denoiser: %.4fms\n",
                        cpuFrameTime.RunningAverage(),
                        gpuFrameTime.RunningAverage(),
                        tessAvgMs + lodAvgMs + tlasAvgMs,
                        gpuRenderTime.RunningAverage(),
                        computeMotionVectorsTimer.RunningAverage(),
                        gpuDenoiserTime.RunningAverage());
                break;
                default:
                    return;
                }
            }

        }

        // Per-stage GPU time (running average), in frame execution order.
        if (mode == GraphMode::Overview)
        {
            auto avg = [](GPUTimer& t) { float v = t.RunningAverage(); return std::isnan(v) ? 0.f : v; };
            auto& cas = clusterAccelSamplers;

            auto avgCpu = [](CPUTimer& t) { float v = t.RunningAverage(); return std::isnan(v) ? 0.f : v; };

            const float hiZMs   = avg(zRenderPassTime) + avg(hiZRenderTime);
            const float idleMs  = cas.hasClusterLod ? avg(cas.clusterLodSubmitIdleTime) : 0.f;
            const float knownMs = tessAvgMs + lodAvgMs + tlasAvgMs + avg(gpuRenderTime)
                                + avg(computeMotionVectorsTimer) + avg(gpuDenoiserTime)
                                + hiZMs + avg(blitTime) + idleMs;

            ImGui::Spacing();
            ImGui::SeparatorText("Frame time (avg)");
            ImGui::Text("CPU frame:      %.3f ms", cpuFrameTime.RunningAverage());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Wall-clock per frame.  Above the GPU frame time below,\n"
                                  "the frame is CPU-bound and the GPU is idling.");
            ImGui::Text("    UI build:   %.3f ms", uiBuildTime.RunningAverage());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Time inside BuildUIMain, included in CPU frame above.\n"
                                  "A panel that walks scene data can dominate this.");
            ImGui::Text("GPU frame:      %.3f ms", avg(gpuFrameTime));
            if (hasAccel)
            {
                ImGui::Text("Accel Build:    %.3f ms", tessAvgMs + lodAvgMs + tlasAvgMs);
                if (cas.hasClusterTess) ImGui::Text("    Cluster Tess: %.3f ms", tessAvgMs);
                if (cas.hasClusterLod)  ImGui::Text("    Cluster LOD:  %.3f ms", lodAvgMs);
                ImGui::Text("    TLAS:         %.3f ms", tlasAvgMs);
            }
            ImGui::Text("Hi-Z prepass:   %.3f ms", hiZMs);
            ImGui::Text("Path Tracing:   %.3f ms", avg(gpuRenderTime));
            ImGui::Text("Motion Vectors: %.3f ms", avg(computeMotionVectorsTimer));
            ImGui::Text("Denoiser:       %.3f ms", avg(gpuDenoiserTime));
            ImGui::Text("Blit:           %.3f ms", avg(blitTime));
            if (cas.hasClusterLod)
            {
                ImGui::Text("Stream idle:    %.3f ms", idleMs);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "GPU idle, not work.  The streaming path submits the accel half of\n"
                        "the frame early so the GPU runs it while the host records the rest;\n"
                        "this is the part of that recording the GPU could not cover, and it\n"
                        "counts toward \"GPU frame\" while belonging to no pass.  It grows\n"
                        "with the host cost below, so a frame-time rise can land here with\n"
                        "every build timer flat.");
                ImGui::Text("Host streaming: %.3f ms (CPU)", avgCpu(cas.clusterLodHostTime));
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Host time in StageResidencyUpdate — request handling, group-data staging,\n"
                        "BLAS-cache bookkeeping.  Counts toward CPU frame, not GPU frame.");
            }
            ImGui::Text("Unaccounted:    %.3f ms", std::max(0.f, avg(gpuFrameTime) - knownMs));
        }
    }

    void EvaluatorSamplers::BuildUI(ImFont *iconicFont, ImPlotContext *plotContext)
    {
        if (hasBadTopology)
        {
            ImGui::PushFont(iconicFont);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.223f, 0.325f, 0.447f, 1.f));
            ImGui::SeparatorText((char const*)(u8"\ue08F" "## recommendations"));
            ImGui::PopStyleColor();
            ImGui::PopFont();

            ImGui::TextWrapped("Switch to the 'Topology Quality' color mode in the settings "
                "window to visualize problem areas in the mesh. Areas in red are in need of attention");

            m_topologyQualityButtonPressed = ImGui::Button("Topology Quality");
            if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
                ImGui::SetTooltip("Switches the color mode to 'Toplogy Quality'.");
            ImGui::Spacing();
        }
        char buf[32];

        const float fontScale = ImGui::GetIO().FontGlobalScale;

        auto buildRow = [&buf]<typename T>(char const* name, T value, char const* tooltip = nullptr, bool displayMB = false)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", name);
            ImGui::TableSetColumnIndex(1);
            if constexpr (std::is_same_v<T, size_t>)
            {
                if (displayMB)
                    MegabytesFormatter(static_cast<double>(value), buf, (int) std::size(buf));
                else
                    MemoryFormatter(static_cast<double>(value), buf, (int) std::size(buf));
                ImGui::Text("%s", buf);
            }
            else if constexpr (std::is_same_v<T, float>)
                ImGui::Text("%.1f", value);
            else if constexpr (std::is_integral_v<T>)
                ImGui::Text("%lld", (long long)value);
            if (tooltip && ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tooltip);
        };

        ImGui::SeparatorText("Per-geometry data");
        ImGui::TextWrapped("To see per-geometry subdivision data, click the \"Inspector\" button.");
        if (ImGui::Button("Inspector"))
            m_openInspectorRequested = true;

        ImGui::Spacing();

        // Scene-wide topology map: the subdivision-plan hashmap.
        ImGui::SeparatorText("TopologyMap");
        {
            ImGui::BeginTable("Topology Map", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX);
            {
                static const float kColWidth0 = 200 * fontScale;
                static const float kColWidth1 = 80 * fontScale;

                ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, kColWidth0);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, kColWidth1);
                ImGui::TableHeadersRow();

                buildRow("PSL mean", topologyMapStats.pslMean);
                buildRow("Hash count", (int64_t)topologyMapStats.hashCount);
                buildRow("Address space", (int64_t)topologyMapStats.addressCount);
                buildRow("load factor", topologyMapStats.loadFactor);
                ImGui::TableNextRow(ImGuiTableRowFlags_Headers, 2.5f);
                //buildRow( "Patch-points max", (int64_t)topologyMap.patchPointsMax );
                buildRow("Subdivision plans count", (int64_t)topologyMapStats.plansCount);
                buildRow("Stencil matrix row count min", (int64_t)topologyMapStats.stencilCountMin);
                buildRow("Stencil matrix row count max", (int64_t)topologyMapStats.stencilCountMax);
                buildRow("Stencil matrix row count avg", (int64_t)topologyMapStats.stencilCountAvg);
                buildRow("Memory use", topologyMapStats.plansByteSize);
            }
            ImGui::EndTable();

            ImGui::SameLine();

            if (ImPlot::BeginPlot("##TopomapStencilHistogram", ImVec2(-1, 174 * fontScale), ImPlotFlags_NoMouseText))
            {
                auto const& values = topologyMapStats.stencilCountHistogram;

                ImPlotFormatter formatter = [](double value, char* buff, int size, void* user_data) -> int
                    {
                        auto const* samplers = reinterpret_cast<EvaluatorSamplers const*>(user_data);
                        uint32_t    min = samplers->topologyMapStats.stencilCountMin;
                        uint32_t    max = samplers->topologyMapStats.stencilCountMax;
                        uint32_t    count = (uint32_t)samplers->topologyMapStats.stencilCountHistogram.size();
                        if (uint32_t range = max - min; range > 0 && count > 0)
                            value = min + (value / count) * range;
                        else
                            value = min;
                        return snprintf(buff, size_t(size), "%d", (int)value);
                    };

                ImPlot::SetupAxis(ImAxis_X1, "Num patch points", ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisFormat(ImAxis_X1, formatter, (void*)this);
                ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_AutoFit);
                ImPlot::PlotBars("Num Plans", values.data(), static_cast<int>(values.size()), 1.0, 0.5, ImPlotBarsFlags_None);

                ImPlot::EndPlot();
            }
        }
    }


    void ClusterAccelSamplers::BuildTessUI(ImFont *iconicFont, ImPlotContext *plotContext) const
    {
        constexpr int stride  = (int)sizeof(uint32_t);
        constexpr int fstride = (int)sizeof(float);

        const float fontScale = ImGui::GetIO().FontGlobalScale;

        // ============================= Cluster Tess ==========================
        if (hasClusterTess)
        {
            // Profiled once here, then reused by the timing and throughput plots.
            auto const& clusterTiling = clusterTilingTime.Profile();
            auto const& fillClusters  = fillClustersTime.Profile();
            auto const& buildClas     = buildClasTime.Profile();
            auto const& buildBlas     = buildBlasTime.Profile();

            if (ImPlot::BeginPlot("##accel_builder_tess", ImVec2(-1, 150 * fontScale)))
            {
                ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
                float vmax = 3.f;
                if (float ravg = std::max(buildClas.RunningAverage(),
                    std::max(buildBlas.RunningAverage(), std::max(clusterTiling.RunningAverage(), fillClusters.RunningAverage())));
                    ravg > (vmax * .01f))
                    vmax = ravg * 2.f;
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0., vmax, ImPlotCond_Always);

                ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(buildBlas.size()), ImGuiCond_Always);

                ImPlot::SetupAxis(ImAxis_Y1, "Time (ms)", ImPlotAxisFlags_AutoFit);
                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
                auto plot = [](auto& series)
                    {
                        ImPlot::PlotLine(series.name.c_str(), series.data(), (int)series.size(), xscale, xstart,
                                          ImPlotShadedFlags_None, static_cast<int>(series.Offset()), stride);
                    };
                plot(clusterTiling);
                plot(fillClusters);
                plot(buildClas);
                plot(buildBlas);
                ImPlot::EndPlot();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "GPU timers:\n\n"
                        "  - Tiling: %.4fms tessellation metric\n"
                        "    + limit surface evaluation prep\n\n"
                        "  - Fill: %.4fms subdivision surface\n"
                        "    limit evaluation + vertex writing.\n\n"
                        "  - CLAS Build: %.4fms CLAS build time.\n\n"
                        "  - BLAS Build: %.4fms BLAS from CLAS build time",
                        clusterTiling.RunningAverage(),
                        fillClusters.RunningAverage(),
                        buildClas.RunningAverage(),
                        buildBlas.RunningAverage());
            }
            ImGui::Spacing();

            auto const& nt = numTriangles;
            auto const& nc = numClusters;
            if (ImPlot::BeginPlot("##accel_builder_geo", ImVec2(-1, 150 * fontScale)))
            {
                ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(nt.size()), ImGuiCond_Always);
                ImPlot::SetupAxis(ImAxis_Y1, nt.name.c_str(), ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisFormat(ImAxis_Y1, HumanFormatter, nullptr);

                ImPlot::SetupAxis(ImAxis_X2, nullptr, ImPlotAxisFlags_NoDecorations);
                ImPlot::SetupAxisLimits(ImAxis_X2, 0, static_cast<double>(nc.size()), ImGuiCond_Always);
                ImPlot::SetupAxis(ImAxis_Y2, nc.name.c_str(), ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_AuxDefault);
                ImPlot::SetupAxisFormat(ImAxis_Y2, HumanFormatter, nullptr);

                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);

                ImPlot::PlotShaded(nt.name.c_str(), nt.data(), (int)nt.size(), yref, xscale, xstart, ImPlotShadedFlags_None,
                                    static_cast<int>(nt.Offset()), stride);

                ImPlot::SetAxes(ImAxis_X2, ImAxis_Y2);
                ImPlot::PlotLine(nc.name.c_str(), nc.data(), (int)nc.size(), xscale, xstart, ImPlotShadedFlags_None,
                                  static_cast<int>(nc.Offset()), stride);

                ImPlot::EndPlot();
            }
            ImGui::Spacing();

            // Tessellated triangles/sec through the tiling -> fill -> CLAS -> BLAS
            // pipeline: a build-rate metric, not an overall-frame one.
            if (Profiler::Get().IsRecording())
            {
                float sumTime = clusterTiling.latest + fillClusters.latest + buildClas.latest + buildBlas.latest;
                uint32_t ntris = numTriangles.latest;
                tessTrisPerSec.PushBack(static_cast<float>(1000. * double(ntris) / double(std::max(sumTime, 1e-6f))));
            }
            if (ImPlot::BeginPlot("BVH Throughput", ImVec2(-1, 150 * fontScale)))
            {
                ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, (double)tessTrisPerSec.size(), ImGuiCond_Always);
                ImPlot::SetupAxis(ImAxis_Y1, "Tris / Sec", ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisFormat(ImAxis_Y1, HumanFormatter, nullptr);
                ImPlot::PlotShaded(tessTrisPerSec.name.c_str(), tessTrisPerSec.data(), (int)tessTrisPerSec.size(), 0.f, xscale, xstart,
                                   ImPlotShadedFlags_None, tessTrisPerSec.Offset(), fstride);
                ImPlot::EndPlot();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Tessellated triangles processed per second:\n"
                        "  - surface edge-metric evaluation\n"
                        "  - Catmull-Clark limit surface evaluation\n"
                        "  - displacement\n"
                        "  - tessellation\n"
                        "  - cluster fill\n"
                        "  - BVH build\n");
            }
            ImGui::Spacing();
        }
    }

    void ClusterAccelSamplers::BuildLodUI(ImFont *iconicFont, ImPlotContext *plotContext) const
    {
        constexpr int stride  = (int)sizeof(uint32_t);
        constexpr int fstride = (int)sizeof(float);
        const float fontScale = ImGui::GetIO().FontGlobalScale;

        // ============================= Cluster Lod ===========================
        // Per-frame LOD traversal + streaming CLAS/BLAS work.  Spikier than the
        // tessellation path: most of the cost lands on frames with residency changes.
        if (hasClusterLod)
        {

            ImGui::SeparatorText("Cluster Lod");

            if (ImPlot::BeginPlot("##accel_builder_cluster_lod", ImVec2(-1, 150 * fontScale)))
            {
                ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(clusterLodTraversal.size()), ImGuiCond_Always);
                float vmax = 3.f;
                if (float ravg = std::max(std::max(clusterLodTraversal.RunningAverage(), clusterLodAllocation.RunningAverage()),
                                          std::max(clusterLodClasBuild.RunningAverage(), clusterLodBlasBuild.RunningAverage()));
                    ravg > (vmax * .01f))
                    vmax = ravg * 2.f;
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0., vmax, ImPlotCond_Always);

                ImPlot::SetupAxis(ImAxis_Y1, "Time (ms)", ImPlotAxisFlags_AutoFit);
                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
                auto plotf = [](auto& series)
                    {
                        ImPlot::PlotLine(series.name.c_str(), series.data(), (int)series.size(), xscale, xstart,
                                          ImPlotShadedFlags_None, static_cast<int>(series.Offset()), fstride);
                    };
                plotf(clusterLodTraversal);
                plotf(clusterLodAllocation);
                plotf(clusterLodClasBuild);
                plotf(clusterLodBlasBuild);
                plotf(clusterLodUpload);
                ImPlot::EndPlot();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Cluster-LOD GPU timers (spikier than Cluster Tess — LOD\n"
                        "geometry is persistent, so work lands on residency changes):\n\n"
                        "  - Traversal:  %.4fms LOD traversal.\n\n"
                        "  - Allocation: %.4fms CLAS pool alloc / freegaps /\n"
                        "    compaction (incl. Streaming Compact allocator CLAS moves).\n\n"
                        "  - CLAS Build: %.4fms implicit CLAS build + persistent\n"
                        "    allocator CLAS moves.\n\n"
                        "  - BLAS Build: %.4fms cluster-LOD BLAS build.\n\n"
                        "  - Upload:     %.4fms group-data / resident / update copies.",
                        clusterLodTraversal.RunningAverage(),
                        clusterLodAllocation.RunningAverage(),
                        clusterLodClasBuild.RunningAverage(),
                        clusterLodBlasBuild.RunningAverage(),
                        clusterLodUpload.RunningAverage());
            }
            ImGui::Spacing();

            // Per-frame TLAS geometry.  Unique (CLAS-deduped footprint = BLAS memory)
            // and Total (instanced = ray-traced) differ by orders of magnitude, so
            // they get separate plots — a shared axis would hide the smaller one.
            auto geoPlot = [&](const char* id, auto const& tris, auto const& clusters)
            {
                if (ImPlot::BeginPlot(id, ImVec2(-1, 150 * fontScale)))
                {
                    ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
                    ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(tris.size()), ImGuiCond_Always);
                    ImPlot::SetupAxis(ImAxis_Y1, "Triangles", ImPlotAxisFlags_AutoFit);
                    ImPlot::SetupAxisFormat(ImAxis_Y1, HumanFormatter, nullptr);

                    ImPlot::SetupAxis(ImAxis_X2, nullptr, ImPlotAxisFlags_NoDecorations);
                    ImPlot::SetupAxisLimits(ImAxis_X2, 0, static_cast<double>(clusters.size()), ImGuiCond_Always);
                    ImPlot::SetupAxis(ImAxis_Y2, "Clusters", ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_AuxDefault);
                    ImPlot::SetupAxisFormat(ImAxis_Y2, HumanFormatter, nullptr);

                    ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
                    ImPlot::SetNextLineStyle(ImVec4(0.99f, 0.55f, 0.10f, 1.0f));
                    ImPlot::PlotLine(tris.name.c_str(), tris.data(), (int)tris.size(), xscale, xstart,
                                     ImPlotLineFlags_None, static_cast<int>(tris.Offset()), stride);
                    ImPlot::SetAxes(ImAxis_X2, ImAxis_Y2);
                    ImPlot::SetNextFillStyle(ImVec4(0.20f, 0.50f, 0.80f, 1.0f), 0.6f);
                    ImPlot::PlotShaded(clusters.name.c_str(), clusters.data(), (int)clusters.size(), yref, xscale, xstart,
                                       ImPlotShadedFlags_None, static_cast<int>(clusters.Offset()), stride);
                    ImPlot::EndPlot();
                }
            };
            if (ImGui::CollapsingHeader("Unique (CLAS footprint)", ImGuiTreeNodeFlags_DefaultOpen))
                geoPlot("##accel_builder_cluster_lod_geo_unique", clusterLodUniqueTriangles, clusterLodUniqueClusters);
            if (ImGui::CollapsingHeader("Total (instanced)"))
                geoPlot("##accel_builder_cluster_lod_geo_total", clusterLodTotalTriangles, clusterLodTotalClusters);
            ImGui::Spacing();

            // ---- BLAS reuse + per-frame geometry stats ------------------------
            const auto& ss = stats::streamingSamplers;
            const shaderio::SceneBuildingCounters& bc = ss.latestCounters;

            ImGui::SeparatorText("BLAS reuse");

            if (ImGui::BeginTable("##cluster_lod_geo_stats", 2,
                                  ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX))
            {
                ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, 240 * fontScale);
                ImGui::TableSetupColumn("Value",  ImGuiTableColumnFlags_WidthFixed, 150 * fontScale);
                ImGui::TableHeadersRow();
                char mb[32];
                auto cnt = [&](const char* n, uint64_t v)
                {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(n);
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%llu", (unsigned long long)v);
                };
                // Cached subset drawn as the fill, the unique total as the track.
                auto cachedBar = [&](const char* n, uint64_t cached, uint64_t total)
                {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(n);
                    ImGui::TableSetColumnIndex(1);
                    const float frac = total ? float(double(cached) / double(total)) : 0.f;
                    char hc[24], ht[24], ov[56];
                    HumanFormatter(double(cached), hc, (int)std::size(hc));
                    HumanFormatter(double(total),  ht, (int)std::size(ht));
                    snprintf(ov, std::size(ov), "%s / %s (%.0f%%)", hc, ht, frac * 100.f);
                    ImGui::ProgressBar(frac, ImVec2(-1.f, 0.f), ov);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s cached / %s unique (%.1f%%)", hc, ht, frac * 100.f);
                };
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("BLAS built");
                MegabytesFormatter(double(ss.latestBlasActualBytes), mb, (int)std::size(mb));
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(mb);
                cnt("BLAS builds",       bc.blasBuildCounter);
                cnt("Sharing providers", bc.numSharingProviders);
                cnt("Sharing consumers", bc.numSharingConsumers);
                cnt("Merged BLASes",     bc.numMergedBlas);
                cnt("Cached BLASes",     ss.latest.cachedBlasCount);
                cachedBar("Cached clusters / Unique Clusters",   bc.cachedUniqueClusters,  bc.uniqueClusters);
                cachedBar("Cached triangles / Unique Triangles", bc.cachedUniqueTriangles, bc.uniqueTriangles);
                cnt("Rendered clusters", bc.numRenderedClusters);
                cnt("Unique clusters",   bc.uniqueClusters);
                cnt("Total clusters",    bc.totalClusters);
                cnt("Unique triangles",  bc.uniqueTriangles);
                cnt("Total triangles",   bc.totalTriangles);
                ImGui::EndTable();
            }
            ImGui::Spacing();
        }
    }

    float ClusterAccelSamplers::ProfileClusterLodPhases() const
    {
        if (!hasClusterLod || !Profiler::Get().IsRecording())
            return 0.f;

        // Profile() drains the sub-timer ring, so exactly one caller per frame can
        // read it -- hence all five phase series are summed here rather than in
        // the UI, which would leave them empty on a headless run.  A region that
        // did not dispatch this frame contributes .latest == 0.
        const float traversal  = clusterLodTraversalTime.Profile().latest;
        const float clasBuild  = clusterLodClasBuildTime.Profile().latest
                               + clusterLodClasMovePersistentTime.Profile().latest;
        const float allocation = clusterLodAllocUnloadUpdateTime.Profile().latest
                               + clusterLodAllocFreegapsTime.Profile().latest
                               + clusterLodAllocAgeTime.Profile().latest
                               + clusterLodAllocLoadTime.Profile().latest
                               + clusterLodAllocStatusTime.Profile().latest
                               + clusterLodClasMoveCompactionTime.Profile().latest;
        const float blasBuild  = clusterLodBlasBuildTime.Profile().latest;
        const float upload     = clusterLodUploadTime.Profile().latest;

        clusterLodTraversal.PushBack(traversal);
        clusterLodClasBuild.PushBack(clasBuild);
        clusterLodAllocation.PushBack(allocation);
        clusterLodBlasBuild.PushBack(blasBuild);
        clusterLodUpload.PushBack(upload);

        return clasBuild + allocation;
    }

    // ---- Memory-tab shared widgets -----------------------------------------

    // Kept dark so the white bar-overlay text stays legible.
    static ImVec4 SatColor(float frac)
    {
        const ImVec4 satGreen (0.16f, 0.40f, 0.18f, 1.f);
        const ImVec4 satYellow(0.50f, 0.42f, 0.10f, 1.f);
        const ImVec4 satRed   (0.55f, 0.15f, 0.15f, 1.f);
        return frac >= 0.90f ? satRed : (frac >= 0.80f ? satYellow : satGreen);
    }

    // cap == 0 means "unbudgeted" -> print the used value alone.
    static void MemoryBar(uint64_t used, uint64_t cap)
    {
        char u[24];
        MemoryFormatter(static_cast<double>(used), u, (int)std::size(u));
        if (cap == 0)
        {
            ImGui::TextUnformatted(u);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%llu bytes", (unsigned long long)used);
            return;
        }
        char c[24], ov[64];
        MemoryFormatter(static_cast<double>(cap), c, (int)std::size(c));
        const float frac = float(double(used) / double(cap));
        snprintf(ov, std::size(ov), "%s / %s (%.0f%%)", u, c, frac * 100.f);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, SatColor(frac));
        ImGui::ProgressBar(frac, ImVec2(-1.0f, 0.f), ov);
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%llu / %llu bytes (%.1f%%)",
                              (unsigned long long)used, (unsigned long long)cap, frac * 100.f);
    }

    // Slot-count analogue of MemoryBar: at 100% nothing more can stream in even
    // if the byte pools still have room.
    static void CountBar(uint64_t used, uint64_t cap)
    {
        char u[24];
        HumanFormatter(static_cast<double>(used), u, (int)std::size(u));
        if (cap == 0) { ImGui::TextUnformatted(u); return; }
        char c[24], ov[64];
        HumanFormatter(static_cast<double>(cap), c, (int)std::size(c));
        const float frac = float(double(used) / double(cap));
        snprintf(ov, std::size(ov), "%s / %s (%.0f%%)", u, c, frac * 100.f);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, SatColor(frac));
        ImGui::ProgressBar(frac, ImVec2(-1.0f, 0.f), ov);
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%llu / %llu (%.1f%%)%s",
                              (unsigned long long)used, (unsigned long long)cap, frac * 100.f,
                              frac >= 1.f ? " - EXHAUSTED: nothing more can stream" : "");
    }

    // Shared by the Cluster Tess and Cluster LOD sections so they read identically.
    static bool BeginMemTable(const char* id, float fontScale)
    {
        if (!ImGui::BeginTable(id, 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX))
            return false;
        ImGui::TableSetupColumn("Name",            ImGuiTableColumnFlags_WidthFixed, 200 * fontScale);
        ImGui::TableSetupColumn("Used (Blocks) / Max", ImGuiTableColumnFlags_WidthFixed, 170 * fontScale);
        ImGui::TableSetupColumn("Per-uTri",        ImGuiTableColumnFlags_WidthFixed, 80 * fontScale);
        ImGui::TableSetupColumn("Per Pixel",       ImGuiTableColumnFlags_WidthFixed, 80 * fontScale);
        ImGui::TableSetupColumn("Per Cluster",     ImGuiTableColumnFlags_WidthFixed, 80 * fontScale);
        ImGui::TableHeadersRow();
        return true;
    }

    static void MetricRow(const char* name, uint64_t used, uint64_t cap,
                          uint64_t tris, uint32_t pixels, uint32_t clusters)
    {
        char b[32];
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(name);
        ImGui::TableSetColumnIndex(1); MemoryBar(used, cap);
        ImGui::TableSetColumnIndex(2);
        if (tris)     ImGui::Text("%.1f bits", double(used) / double(tris) * 8.0); else ImGui::TextUnformatted("n/a");
        ImGui::TableSetColumnIndex(3);
        if (pixels)   ImGui::Text("%.2f B", double(used) / double(pixels));        else ImGui::TextUnformatted("n/a");
        ImGui::TableSetColumnIndex(4);
        if (clusters) { MemoryFormatter(double(used) / double(clusters), b, (int)std::size(b)); ImGui::TextUnformatted(b); }
        else          ImGui::TextUnformatted("n/a");
    }

    // Residency slot counts share the byte-pool table so the caps read alongside
    // Geometry/CLAS; the per-byte metric columns don't apply.
    static void CountRow(const char* name, uint64_t used, uint64_t cap)
    {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(name);
        ImGui::TableSetColumnIndex(1); CountBar(used, cap);
        ImGui::TableSetColumnIndex(2); ImGui::TextUnformatted("n/a");
        ImGui::TableSetColumnIndex(3); ImGui::TextUnformatted("n/a");
        ImGui::TableSetColumnIndex(4); ImGui::TextUnformatted("n/a");
    }

    // ---- Material textures: resident GPU bytes vs the scene's full-resolution
    //      footprint.  The environment map is excluded (it isn't budgeted). ----
    static void BuildTextureMemUI(const TextureMemStats& t, float fontScale)
    {
        const uint64_t fullBytes = t.FullBytes();
        char loaded[24], full[24], budget[24], overlay[96];
        MemoryFormatter(double(t.loadedBytes), loaded, (int)std::size(loaded));
        MemoryFormatter(double(fullBytes),     full,   (int)std::size(full));

        const float frac = fullBytes ? float(double(t.loadedBytes) / double(fullBytes)) : 0.f;
        // The budget binds only if the scene has KTX2 to drop mips from; quoting
        // a Max on an all-.jpg scene would imply a cap that can't be enforced.
        const bool budgetApplies = t.budgetableCount != 0;
        if (budgetApplies && t.budgetBytes)
        {
            MemoryFormatter(double(t.budgetBytes), budget, (int)std::size(budget));
            snprintf(overlay, std::size(overlay), "%s / %s (%.0f%%)  (Max: %s)", loaded, full, frac * 100.f, budget);
        }
        else if (budgetApplies)
            snprintf(overlay, std::size(overlay), "%s / %s (%.0f%%)  (Max: unlimited)", loaded, full, frac * 100.f);
        else
            snprintf(overlay, std::size(overlay), "%s / %s (%.0f%%)", loaded, full, frac * 100.f);

        // Below 100% just means mips were dropped or are still streaming, so
        // this bar takes a flat fill instead of MemoryBar's saturation colors.
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.20f, 0.34f, 0.50f, 1.f));
        ImGui::ProgressBar(std::min(frac, 1.f), ImVec2(-1.0f, 0.f), overlay);
        ImGui::PopStyleColor();
        const ImVec2 barMin = ImGui::GetItemRectMin(), barMax = ImGui::GetItemRectMax();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%llu / %llu bytes resident (%.1f%%).\n"
                              "Denominator is the scene's FULL-resolution GPU footprint (every mip of\n"
                              "every material texture), so the budget dropping high-res mips shows as\n"
                              "a bar short of 100%%.  Measured up front for KTX2; other formats have no\n"
                              "footprint to read before decoding, but nothing drops their mips, so a\n"
                              "resident one counts as its own full size.  The environment map is not\n"
                              "counted (it isn't budgeted).",
                              (unsigned long long)t.loadedBytes, (unsigned long long)fullBytes,
                              frac * 100.f);

        // Budget waterline, drawn only below the full footprint — above it the
        // whole set fits and the marker would just pin to the right edge.
        if (budgetApplies && t.budgetBytes && fullBytes && t.budgetBytes < fullBytes)
        {
            const float x = barMin.x + (barMax.x - barMin.x) * float(double(t.budgetBytes) / double(fullBytes));
            ImGui::GetWindowDrawList()->AddLine(ImVec2(x, barMin.y), ImVec2(x, barMax.y),
                                                ImGui::GetColorU32(ImVec4(1.f, 0.85f, 0.25f, 0.9f)), 2.f);
        }

        if (ImGui::BeginTable("##tex_mem", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX))
        {
            ImGui::TableSetupColumn("Textures", ImGuiTableColumnFlags_WidthFixed, 200 * fontScale);
            ImGui::TableSetupColumn("Value",    ImGuiTableColumnFlags_WidthFixed, 150 * fontScale);
            ImGui::TableHeadersRow();
            char b[32];
            auto texRow = [&](const char* name, const char* value, const char* tooltip)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(name);
                if (tooltip && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(value);
            };
            auto texMemRow = [&](const char* name, uint64_t bytes, const char* tooltip)
            {
                MemoryFormatter(double(bytes), b, (int)std::size(b));
                texRow(name, b, tooltip);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%llu bytes", (unsigned long long)bytes);
            };
            char counts[64];
            snprintf(counts, std::size(counts), "%u / %u", t.loadedCount, t.textureCount);
            texRow("Resident / scene", counts,
                   "Textures finalized onto the GPU vs the unique images the scene references.\n"
                   "Short of the total mid-load = still decoding; short of it afterwards = a\n"
                   "texture failed to load (see the log).");
            texMemRow("Resident (GPU)", t.loadedBytes,
                      "GPU bytes the resident textures occupy right now - the bar's numerator.");
            texMemRow("On disk", t.diskBytes,
                      "Sum of the texture files as stored. Well under the GPU footprint for\n"
                      "KTX2 (the GPU holds the BCn payload inflated, not the zstd stream) and\n"
                      "for .jpg/.png (which decode to uncompressed RGBA).");
            texMemRow("Full res (GPU)", fullBytes,
                      "GPU bytes if every mip of every texture were resident - the bar's\n"
                      "denominator. Read from the file headers for KTX2/DDS; the other\n"
                      "formats never drop mips, so this counts the ones already decoded\n"
                      "at their resident size (see the bar's tooltip).");

            // KTX2-only rows: on a .jpg/.png scene these would all read zero.
            if (t.budgetableCount)
            {
                if (t.budgetableCount != t.textureCount)
                {
                    snprintf(counts, std::size(counts), "%u", t.budgetableCount);
                    texRow("Budgetable (KTX2/DDS)", counts,
                           "Of the scene's textures, how many are KTX2 or DDS - the formats whose\n"
                           "mips the budget can drop without decoding, because their headers give\n"
                           "the per-level sizes up front. The rest always load in full, whatever\n"
                           "the budget.");
                }
                // The kept size only says something once the budget actually binds;
                // below it, kept == full and the row would just restate Full res.
                texMemRow("Dropped to fit budget", t.budgetableFullBytes - t.keptBytes,
                          "GPU bytes of high-res mips the budget solver discarded from the\n"
                          "KTX2/DDS subset. 0 = the whole set fits and every mip is loaded.");
                if (t.budgetBytes)
                {
                    MemoryFormatter(double(t.budgetBytes), b, (int)std::size(b));
                    texRow("Budget", b,
                           "The memory target mips are dropped to fit (--texture-budget-mb).\n"
                           "0 disables budgeting.");
                }
                else
                    texRow("Budget", "unlimited",
                           "Budgeting disabled (--texture-budget-mb 0): every mip is loaded.");
                snprintf(counts, std::size(counts), "%u", t.droppedCount);
                texRow("Textures with mips dropped", counts,
                       "How many textures gave up high-resolution mips to fit the budget.\n"
                       "0 = the whole set fit at full resolution.");
            }
            ImGui::EndTable();
        }
        ImGui::Spacing();
    }

    // ---- Cluster Tess + subdivision build memory ----------------------------
    static void BuildTessMemUI(const MemUsageSamplers& mem, const ClusterAccelSamplers& cas, float fontScale)
    {
        const uint32_t desiredTris       = cas.numTriangles.latest;
        const uint32_t desiredClusters   = cas.numClusters.latest;
        const uint32_t allocatedClusters = cas.numClusters.max;
        const uint32_t numPixels         = cas.renderSize.x * cas.renderSize.y;

        ImGui::Text("Render Resolution: %d x %d", cas.renderSize.x, cas.renderSize.y);
        ImGui::Text("Micro-triangles: %u (%.2f per pixel)", desiredTris, numPixels ? desiredTris / float(numPixels) : 0.f);
        ImGui::Text("Clusters: %u / %u", desiredClusters, allocatedClusters);
        ImGui::Spacing();

        if (BeginMemTable("Memory Usage", fontScale))
        {
            MetricRow("Vertex buffer",          mem.vertexBufferSize.latest,        mem.vertexBufferSize.max,        desiredTris, numPixels, desiredClusters);
            MetricRow("Vertex normals buffer",  mem.vertexNormalsBufferSize.latest, mem.vertexNormalsBufferSize.max, desiredTris, numPixels, desiredClusters);
            MetricRow("Cluster AS (CLAS)",      mem.clasSize.latest,                mem.clasSize.max,                desiredTris, numPixels, desiredClusters);
            MetricRow("Cluster Data buffer",    mem.clusterShadingDataSize.latest,  mem.clusterShadingDataSize.max,  0, 0, desiredClusters);
            // BLAS and its scratch are just-sized and unbudgetable, so pass cap 0
            // to print the value instead of an always-100% bar.
            MetricRow("Bottom Level AS (BLAS)", mem.blasSize.latest,                0,                           0, 0, allocatedClusters);
            MetricRow("BLAS scratch buffer",    mem.blasScratchSize.latest,         0,                           0, 0, allocatedClusters);

            const size_t total = mem.blasSize.latest + mem.blasScratchSize.latest + mem.clasSize.latest
                + mem.vertexBufferSize.latest + mem.vertexNormalsBufferSize.latest + mem.clusterShadingDataSize.latest;
            const size_t totalMax = mem.blasSize.max + mem.blasScratchSize.max + mem.clasSize.max
                + mem.vertexBufferSize.max + mem.vertexNormalsBufferSize.max + mem.clusterShadingDataSize.max;
            MetricRow("Total ", total, totalMax, desiredTris, numPixels, desiredClusters);
            ImGui::EndTable();
        }

        ImGui::Spacing();
        ImGui::TextUnformatted("Topology & Subdivision");
        ImGui::Spacing();
        if (ImGui::BeginTable("Subdivision", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX))
        {
            ImGui::TableSetupColumn("Name",   ImGuiTableColumnFlags_WidthFixed, 200 * fontScale);
            ImGui::TableSetupColumn("Memory", ImGuiTableColumnFlags_WidthFixed, 80 * fontScale);
            ImGui::TableHeadersRow();
            char b[32];
            auto subdRow = [&](char const* name, size_t sz, char const* tooltip)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(name);
                if (tooltip && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
                ImGui::TableSetColumnIndex(1);
                MemoryFormatter(static_cast<double>(sz), b, (int)std::size(b));
                ImGui::TextUnformatted(b);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%zu bytes.", sz);
            };
            subdRow("Topology map", evaluatorSamplers.topologyMapStats.plansByteSize,
                "Total size of the topology map (one shared by all sub-d meshes).");
            subdRow("Surface tables", evaluatorSamplers.surfaceTablesByteSizeTotal,
                "Total size of vertex surface tables (replace the index buffer; ~3-5x cage size).");
            subdRow("Total ", evaluatorSamplers.topologyMapStats.plansByteSize
                            + evaluatorSamplers.surfaceTablesByteSizeTotal, nullptr);
            ImGui::EndTable();
        }
    }

    // ---- Scene totals: static bake counts, not live residency ---------------
    static void BuildClodSceneTotalsUI(const rtxmg::StreamingStats& s, float fontScale)
    {
        if (s.geometryCount)
        {
            ImGui::SeparatorText("Static Bake Totals");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Bake-time scene totals, NOT live residency.\n"
                                  "Unique = each geometry counted once (asset cost).\n"
                                  "Instanced = weighted by instance references (a mesh instanced N\n"
                                  "times counts N x).");
            if (ImGui::BeginTable("##scene_cluster_lod", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX))
            {
            // Deliberately blank header on the metric-name column, so the header
            // row reads "Instanced | Unique".
            ImGui::TableSetupColumn("##metric", ImGuiTableColumnFlags_WidthFixed, 200 * fontScale);
            ImGui::TableSetupColumn("Instanced", ImGuiTableColumnFlags_WidthFixed, 125 * fontScale);
            ImGui::TableSetupColumn("Unique",    ImGuiTableColumnFlags_WidthFixed, 125 * fontScale);
            ImGui::TableHeadersRow();
            char a[24], b[24];
            auto sceneRow = [&](const char* name, uint64_t instanced, uint64_t unique,
                                const char* tooltip)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(name);
                if (tooltip && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
                ImGui::TableSetColumnIndex(1);
                if (instanced) { HumanFormatter(double(instanced), a, (int)std::size(a)); ImGui::TextUnformatted(a); }
                else           ImGui::TextUnformatted("-");
                ImGui::TableSetColumnIndex(2);
                if (unique) { HumanFormatter(double(unique), b, (int)std::size(b)); ImGui::TextUnformatted(b); }
                else        ImGui::TextUnformatted("-");
            };
            sceneRow("Triangles", s.sceneTriangles, s.modelTriangles,
                     "Full-detail (LOD0) triangles. Instanced = weighted by instance references.");
            sceneRow("Clusters",  s.sceneClusters,  s.modelClusters,
                     "Full-detail (LOD0) clusters. Instanced = weighted by instance\n"
                     "references.");
            sceneRow("Clusters (all LODs)", s.sceneClustersAllLods, s.modelClustersAllLods,
                     "Baked clusters across the WHOLE LOD hierarchy (LOD0 + every coarser\n"
                     "level), vs the \"Clusters\" row above which is LOD0 only. Instanced =\n"
                     "weighted by instance references.");
            sceneRow("Groups",    s.sceneGroups,    s.modelGroups,
                     "Baked cluster groups across ALL LOD levels (streaming granularity).\n"
                     "Instanced = weighted by instance references.");
            sceneRow("Meshes", s.instanceCount,     s.geometryCount,
                     "Instanced = mesh placements in the scene (each references one geometry).\n"
                     "Unique = geometries after mesh/primitive dedup.");
            ImGui::EndTable();
            }
        }
    }

    // ---- Traversal, from a 1-frame-latency readback.  A saturated row means
    //      traversal wanted more than the buffers allow this frame. -----------
    static void BuildClodTraversalUI(const shaderio::SceneBuildingCounters& bc,
                                     const rtxmg::StreamingStats& s, float fontScale)
    {
        if (ImGui::BeginTable("##trav_cluster_lod", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX))
        {
            ImGui::TableSetupColumn("Per Frame",    ImGuiTableColumnFlags_WidthFixed, 200 * fontScale);
            ImGui::TableSetupColumn("Used / Limit", ImGuiTableColumnFlags_WidthFixed, 250 * fontScale);
            ImGui::TableHeadersRow();
            char a[24], b[24];
            // `limit` names the cap the bar divides by, so the row says which
            // budget to raise rather than just that it is full.
            auto travRow = [&](const char* name, uint64_t requested, uint64_t reserved,
                               const char* limit, const char* tooltip)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(name);
                if (tooltip && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
                ImGui::TableSetColumnIndex(1);
                HumanFormatter(double(requested), a, (int)std::size(a));
                if (!reserved)
                {
                    ImGui::TextUnformatted("n/a");
                    return;
                }
                HumanFormatter(double(reserved), b, (int)std::size(b));
                const float frac = float(double(requested) / double(reserved));
                char ov[64];
                snprintf(ov, sizeof(ov), "%s / %s  (%.0f%%)", a, b, frac * 100.f);
                // Saturated means work was dropped this frame, so it gets the
                // same warning red the overflow banner uses.
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                                      frac >= 1.f ? ImVec4(0.62f, 0.16f, 0.11f, 1.f)
                                                  : ImVec4(0.20f, 0.34f, 0.50f, 1.f));
                ImGui::ProgressBar(std::min(frac, 1.f), ImVec2(-FLT_MIN, 0.f), ov);
                ImGui::PopStyleColor();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%llu / %llu\nLimit: %s",
                                      (unsigned long long)requested,
                                      (unsigned long long)reserved, limit);
            };
            // The desired* pair, not the write counters: traversal_setup clamps
            // those in place, so an overflowed frame reads exactly 100%.
            travRow("Tasks (nodes)", bc.desiredTraversalNodes, ClusterLodPass::kMaxTraversalInfos,
                    "kMaxTraversalInfos (1 << 20), fixed at compile time",
                    "Interior LOD-hierarchy nodes traversal pushed onto the node queue this\n"
                    "frame, against that queue's fixed capacity. Full = nodes dropped this\n"
                    "frame, which shows up as missing or coarse geometry.");
            travRow("Groups",        bc.desiredTraversalGroups, ClusterLodPass::kMaxTraversalInfos,
                    "kMaxTraversalInfos (1 << 20), a separate queue of the same size",
                    "Leaf cluster-groups traversal pushed onto the group queue this frame,\n"
                    "against that queue's fixed capacity. Full = groups dropped this frame.\n\n"
                    "NOT the same as \"Groups (resident)\" above: that is the streaming working\n"
                    "set held across frames against the residency budget, while this is one\n"
                    "frame's queue traffic against a fixed 1M-entry queue. A few hundred\n"
                    "groups is normal here and reads as 0%.");
            travRow("Clusters (for BLAS build)", bc.desiredRenderClusters,
                    bc.effectiveMaxRenderClusters,
                    "effectiveMaxRenderClusters = (1 << renderClusterBits) minus the\n"
                    "cached-BLAS reservation; raise \"Render cluster bits\"",
                    "Clusters traversal selected and emitted for BLAS building this frame,\n"
                    "before the cap is applied. Full = clusters dropped this frame (flicker).\n\n"
                    "Not what is on screen: with BLAS caching or sharing a visible cluster can\n"
                    "be served by an existing BLAS and never counted here, so this legitimately\n"
                    "falls toward 0 once a static view converges.");
            travRow("BLAS builds",   bc.blasBuildCounter, s.instanceCount,
                    "scene instance count (not a budget - reuse drives it down)",
                    "Per-instance BLAS actually built this frame, against the scene's total\n"
                    "instance count. This is a reuse metric rather than a budget: BLAS\n"
                    "sharing, caching and merging all push it down, and low is good.");
            ImGui::EndTable();
        }
    }

    // ---- Resident pools, with the per-something metrics over the UNIQUE
    //      resident footprint --------------------------------------------------
    static void BuildClodPoolUI(const rtxmg::StreamingStats& s, uint64_t uTris,
                                uint32_t numPixels, uint32_t uClusters, float fontScale)
    {
        if (BeginMemTable("##mem_cluster_lod", fontScale))
        {
            // The resident geometry pool holds quantized texcoords always and
            // per-vertex normals only when the Vertex Normals toggle is on
            // (positions are fetched from the AS, so they are not pooled).
            MetricRow(s.residentNormals ? "Geometry (normals, texcoord)" : "Geometry (texcoord)",
                                          s.usedDataBytes,   s.maxDataBytes,      uTris, numPixels, uClusters);
            MetricRow("CLAS",             s.usedClasBytes,   s.reservedClasBytes, uTris, numPixels, uClusters);
            MetricRow("CLAS wasted",      s.wastedClasBytes, 0,                   uTris, numPixels, uClusters);
            MetricRow("Cached BLAS pool", s.cachedBlasBytes, 0,                   0, 0, 0);
            // Residency slot caps, tuned via --maxresidentgroups / the UI budgets.
            CountRow("Groups (resident)",   s.residentGroups,   s.maxGroups);
            CountRow("Clusters (resident)", s.residentClusters, s.maxClusters);
            ImGui::EndTable();
        }
        const bool groupsFull   = s.maxGroups   && s.residentGroups   >= s.maxGroups;
        const bool clustersFull = s.maxClusters && s.residentClusters >= s.maxClusters;
        if (groupsFull || clustersFull)
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.45f, 0.30f, 1.f));
            ImGui::TextWrapped("Max resident %s exhausted (%s) - geometry cannot fully "
                               "stream in. Raise --maxresidentgroups or the \"Max resident "
                               "groups\" budget.",
                               groupsFull ? "groups" : "clusters",
                               groupsFull ? "groups slot pool full" : "clusters slot pool full");
            ImGui::PopStyleColor();
        }
    }

    void MemUsageSamplers::BuildUI(ImFont *iconicFont, ImPlotContext *plotContext) const
    {
        const float fontScale = ImGui::GetIO().FontGlobalScale;
        const auto& cas = stats::clusterAccelSamplers;

        if (textures.Valid() && ImGui::CollapsingHeader("Textures", ImGuiTreeNodeFlags_DefaultOpen))
        {
            BuildTextureMemUI(textures, fontScale);
        }

        if (cas.hasClusterTess && ImGui::CollapsingHeader("Cluster Tess", ImGuiTreeNodeFlags_DefaultOpen))
        {
            BuildTessMemUI(*this, cas, fontScale);
        }

        if (cas.hasClusterLod && ImGui::CollapsingHeader("Cluster LOD", ImGuiTreeNodeFlags_DefaultOpen))
        {
            const rtxmg::StreamingStats&           s  = stats::streamingSamplers.latest;
            const shaderio::SceneBuildingCounters& bc = stats::streamingSamplers.latestCounters;
            const uint32_t numPixels = cas.renderSize.x * cas.renderSize.y;

            ImGui::SeparatorText("Used / Budget");
            // uniqueTriangles/uniqueClusters are the CLAS-deduped resident footprint.
            BuildClodPoolUI(s, bc.uniqueTriangles, numPixels, bc.uniqueClusters, fontScale);
            ImGui::Spacing();
            ImGui::SeparatorText("Traversal");
            BuildClodTraversalUI(bc, s, fontScale);
            ImGui::Spacing();
            // Last: bake-time constants, well below the live figures above.
            BuildClodSceneTotalsUI(s, fontScale);
            ImGui::Spacing();
        }
    }

    void SurfaceTableStats::BuildTopologyRecommendations()
    {
        float ratio = 0.f;
        if (!IsCatmarkTopology(&ratio))
        {
            std::stringstream ss;
            ss.setf(std::ios::fixed);
            ss.precision(1);
            ss << "High number of irregular (non-quad) faces detected (" << ratio * 100.f << " %). ";
            ss << "Irregular faces impact both performance and memory. Catmark subdivision ";
            ss << "meshes should use mostly quads.";
            topologyRecommendations.push_back(ss.str());
        }

        // note: TopologyRefiner::GetMaxValence() accumulates both face and vertex valence
        // into the same max variable ; on the rare occasion where a model has both a high
        // valence vertex and a face of equal or greater valence, this recommendation will
        // not be triggered. The assumption is that once the high valence faces are removed
        // from the topology, if high valence vertices remain, this recommendation will 
        // then trigger as intended.
        if ((maxValence > 8) && (maxValence > maxFaceSize))
        {
            std::stringstream ss;
            ss << "Some vertices have up to " << maxValence << " incident edges. Ideally max ";
            ss << "valence should be <= 8.";
            topologyRecommendations.push_back(ss.str());
        }

        if (maxFaceSize > 5)
        {
            std::stringstream ss;
            ss << "Some polygons faces have up to " << maxFaceSize << " edges. It is recommended ";
            ss << "to use quads with a few triangles and pentagons in delicate areas.";
            topologyRecommendations.push_back(ss.str());
        }

        if (sharpnessMax > 8.f)
        {
            std::stringstream ss;
            ss << "Some creased edges or vertices have a very high sharpness value (found up ";
            ss << "to " << sharpnessMax << "). Consider replacing those with 'infinitely sharp' ";
            ss << "creases of value 10 for better performance.";
            topologyRecommendations.push_back(ss.str());
        }
        else if (sharpnessMax > 4.f && sharpnessMax <= 8.f)
        {
            std::stringstream ss;
            ss << "Some creased edges or vertices have a high sharpness value (found up to ";
            ss << sharpnessMax << "). Consider adding edge-loops and reducing sharpness creases ";
            ss << "to values <= 4.0 for better control over the surface and better performance.";
            topologyRecommendations.push_back(ss.str());
        }
    }
    
    void SurfaceTableStats::BuildRecommendationsUI(ImFont *iconicFont) const
    {
        for (auto const& rec : topologyRecommendations)
        {
            ImGui::Spacing();

            ImGui::PushFont(iconicFont);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 0.f, 1.f));
            ImGui::SeparatorText((char const*)(u8"\ue0D8" "## env map"));
            ImGui::PopStyleColor();
            ImGui::PopFont();

            ImGui::TextWrapped("%s", rec.c_str());

            ImGui::Spacing();
        }
    }


    void SurfaceTableStats::BuildDetailUI(ImFont* iconicFont, ImPlotContext* plotContext, uint32_t imguiID) const
    {
        const float fontScale = ImGui::GetIO().FontGlobalScale;

        char buf[128];

        auto buildRow = [&buf]<typename T>(char const* name, T value, char const* tooltip = nullptr, bool displayMB = false)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", name);
            ImGui::TableSetColumnIndex(1);
            if constexpr (std::is_same_v<T, size_t>)
            {
                if (displayMB)
                    MegabytesFormatter(static_cast<double>(value), buf, (int) std::size(buf));
                else
                    MemoryFormatter(static_cast<double>(value), buf, (int) std::size(buf));
                ImGui::Text("%s", buf);
            }
            else if constexpr (std::is_same_v<T, float>)
                ImGui::Text("%.1f", value);
            else if constexpr (std::is_integral_v<T>)
                ImGui::Text("%lld", (long long)value);
            if (tooltip && ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tooltip);
        };

        // Per-geometry detail, rendered full width below the Inspector's expandable
        // summary row (which is what shows/hides it, so no header here).
        {
            BuildRecommendationsUI(iconicFont);

            ImGui::Spacing();

            snprintf(buf, std::size(buf), "##Surface_Table_%d", imguiID);

            ImGui::BeginTable(buf, 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX);
            {
                static const float kColWidth0 = 200 * fontScale;
                static const float kColWidth1 = 80 * fontScale;

                ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, kColWidth0);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, kColWidth1);
                ImGui::TableHeadersRow();

                buildRow("Memory use", byteSize,
                    "Total memory use for Tmr::SurfaceTable.\n"
                    "note: does not account for the texcoord surface table.\n");

                buildRow("Surfaces count", (int64_t)surfaceCount,
                    "Number of surfaces in the table.\n");

                buildRow("Pure regular surface count", bsplineSurfaceCount,
                            "Number of 'pure' regular surfaces in the table that can\n"
                            "be resolved with a single b-spline patch (ie. surfaces\n"
                            "with 16 control points, no boundaries and a subdivision\n"
                            "plan with no stencil matrix.\n");

                buildRow("Irregular face count", irregularFaceCount,
                    "Number of irregular faces in the table (ie. non-quads)\n"
                    "for Catmark subdivision scheme.\n");

                buildRow("Valence max", maxValence,
                    "Maximum vertex valence in the control cage.\n");

                buildRow("Face m_size max", maxFaceSize,
                    "Maximum number of vertices in a face in the control cage.\n");

                buildRow("Sharpness max", sharpnessMax,
                    "Highest sharpness value for edge or vertex in the control cage.\n");

                buildRow("Inf sharp", infSharpCreases,
                    "Number of edges or vertices with an 'infinitely' sharp crease tag.\n");

                buildRow("Stencil matrix row count avg", stencilCountAvg,
                    "Average number of patch-points per surface across the table.\n"
                    "Obtained by iterating over each surface in the table and\n"
                    "summing up the number of patch-points (aka rows in the stencil\n"
                    "matrix) in the subdivision plan associated with that surface.\n"
                    "The aveage is given by dividing this sum by the number of surfaces\n"
                    "in the table.\n"
                    "This average is a proxy measure of the global amount of computations\n"
                    "required to obtain the limit surface. Use of sharp edges or high\n"
                    "valence vertices will increase this average, while prevalance of\n"
                    "'regular' topology lowers this average.\n");
            }
            ImGui::EndTable();

            ImGui::SameLine();

            if (bsplineSurfaceCount > 0)
            {
                snprintf(buf, std::size(buf), "##Surface_Stencil_Pie_Chart_%d", imguiID);

                if (ImPlot::BeginPlot(buf, ImVec2(235 * fontScale, 153 * fontScale), ImPlotFlags_NoMouseText))
                {
                    float count = float(surfaceCount);
                    float bspline_ratio = float(bsplineSurfaceCount) / count;
                    float regular_ratio = float(regularSurfaceCount) / count;
                    float isolation_ratio = float(isolationSurfaceCount) / count;
                    float sharp_ratio = float(sharpSurfaceCount) / count;
                    float holes_ratio = float(holesCount) / count;

                    float values[5] = { bspline_ratio, regular_ratio, isolation_ratio, sharp_ratio, holes_ratio };

                    static char const* labels[std::size(values)] = {
                        "BSpline",
                        "Regular",
                        "Smooth",
                        "Sharp",
                        "Holes",
                    };

                    ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
                    ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_NoDecorations);

                    ImPlot::SetupLegend(ImPlotLocation_West, ImPlotLegendFlags_Outside);

                    ImPlot::PlotPieChart(labels, values, (int) std::size(values), 0, 0, 0.4, "%.2f", ImPlotFlags_NoLegend);

                    ImPlot::EndPlot();
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Distribution of surfaces in the table:\n"
                            " - 'B-Spline' surfaces are 'pure' regular surfaces\n"
                            "   (16 control points in 1-ring, no boundaries, no\n"
                            "   patch-points, no stencil matrix).\n"
                            "   The limit of these surfaces can be evaluated\n"
                            "   through a dedicated fast-path.\n"
                            "\n"
                            " - 'Regular' surfaces are still b-spline surfaces, but\n"
                            "   with boundaries (9 or 12 control points in 1-ring,\n"
                            "   no patch-points, no stencil matrix).\n"
                            "\n"
                            " - 'Smooth' surfaces are areas that require feature\n"
                            "    isolation (a stencil matrix is required, but with a\n"
                            "    lower isolation level).\n"
                            "\n"
                            " - 'Sharp' surfaces are surfaces with semi-sharp\n"
                            "   creases (full feature isolation and stencil matrix).\n");
                }
            }
            ImGui::SameLine();

            snprintf(buf, std::size(buf), "##Surface_Stencil_Hisogram_%d", imguiID);

            if (ImPlot::BeginPlot(buf, ImVec2(-1, 153 * fontScale), ImPlotFlags_NoMouseText))
            {
                auto const& values = stencilCountHistogram;

                ImPlotFormatter formatter = [](double value, char* buff, int size, void* user_data) -> int
                    {
                        auto const* samplers = reinterpret_cast<EvaluatorSamplers const*>(user_data);
                        uint32_t    min = samplers->topologyMapStats.stencilCountMin;
                        uint32_t    max = samplers->topologyMapStats.stencilCountMax;
                        uint32_t    count = (uint32_t)samplers->topologyMapStats.stencilCountHistogram.size();
                        if (uint32_t range = max - min; range > 0 && count > 0)
                            value = min + (value / count) * range;
                        else
                            value = min;
                        return snprintf(buff, size_t(size), "%d", (int)value);
                    };

                ImPlot::SetupAxis(ImAxis_X1, "Num patch points", ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisFormat(ImAxis_X1, formatter, reinterpret_cast<void*>(&evaluatorSamplers));
                ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_AutoFit);
                ImPlot::PlotBars("Num Surfaces", values.data(), static_cast<int>(values.size()), 1.0, 0.5, ImPlotBarsFlags_None);

                ImPlot::EndPlot();
            }
        }

        ImGui::Spacing();
    }

    void StreamingSamplers::BuildUI(ImFont* /*iconicFont*/, ImPlotContext* /*plotContext*/) const
    {
        const float   fontScale = ImGui::GetIO().FontGlobalScale;
        constexpr int fstride   = (int)sizeof(float);

        // This tab is the streaming *dynamics*: transfer/load rates and residency.

        // Hold the plot back until stats start flowing — an empty AutoFit plot draws
        // with an empty scissor, which is benign but noisy.
        if (transferRate.samples_count == 0)
        {
            ImGui::TextDisabled("Waiting for streaming data...");
            ImGui::Spacing();
        }
        else
        {

        // ---- streamed-from-disk throughput + group load/unload rate on a second Y
        //      axis.  Both are per-frame deltas, so they read 0 while idle. -------
        if (ImPlot::BeginPlot("##stream_throughput", ImVec2(-1, 150 * fontScale)))
        {
            ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(transferRate.size()), ImGuiCond_Always);
            ImPlot::SetupAxis(ImAxis_Y1, "Transfer/s", ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisFormat(ImAxis_Y1, HumanFormatter, nullptr);
            ImPlot::SetupAxis(ImAxis_Y2, "groups / s", ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_AuxDefault);

            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
            ImPlot::PlotLine(transferRate.name.c_str(), transferRate.data(), (int)transferRate.size(), xscale, xstart,
                             ImPlotLineFlags_None, (int)transferRate.Offset(), fstride);

            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine(loadsPerSec.name.c_str(), loadsPerSec.data(), (int)loadsPerSec.size(), xscale, xstart,
                             ImPlotLineFlags_None, (int)loadsPerSec.Offset(), fstride);
            ImPlot::PlotLine(unloadsPerSec.name.c_str(), unloadsPerSec.data(), (int)unloadsPerSec.size(), xscale, xstart,
                             ImPlotLineFlags_None, (int)unloadsPerSec.Offset(), fstride);
            ImPlot::EndPlot();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Geometry streamed from disk: %.2f MB/s\n"
                                  "Loads %.0f / Unloads %.0f groups per second",
                                  transferRate.latest / (1024.0f * 1024.0f), loadsPerSec.latest, unloadsPerSec.latest);
        }
        ImGui::Spacing();

        }  // end if (samples_count != 0) — throughput plot

        // ---- snapshot table (always shown; reads zero-initialized `latest`) ----
        const rtxmg::StreamingStats& s = latest;

        if (ImGui::BeginTable("Streaming stats", 3,
                              ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX))
        {
            const float kCol0 = 180 * fontScale;
            const float kCol1 = 90 * fontScale;
            ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, kCol0);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, kCol1);
            ImGui::TableSetupColumn("Max / Reserved", ImGuiTableColumnFlags_WidthFixed, kCol1);
            ImGui::TableHeadersRow();

            char b0[32], b1[32];
            auto memRow = [&](const char* name, uint64_t used, uint64_t cap)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%s", name);
                MegabytesFormatter(static_cast<double>(used), b0, (int)std::size(b0));
                ImGui::TableSetColumnIndex(1); ImGui::Text("%s", b0);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%llu bytes", (unsigned long long)used);
                ImGui::TableSetColumnIndex(2);
                if (cap) { MegabytesFormatter(static_cast<double>(cap), b1, (int)std::size(b1)); ImGui::Text("%s", b1); }
                else     { ImGui::TextDisabled("-"); }
            };
            // `human` switches to K/M/B/T suffixes, for counts that run to the
            // millions on large scenes.
            auto cntRow = [&](const char* name, uint64_t v, uint64_t cap, bool warn = false, bool human = false)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%s", name);
                if (warn) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
                ImGui::TableSetColumnIndex(1);
                if (human) { HumanFormatter(double(v), b0, (int)std::size(b0)); ImGui::Text("%s", b0); }
                else       ImGui::Text("%llu", (unsigned long long)v);
                if (human && ImGui::IsItemHovered()) ImGui::SetTooltip("%llu", (unsigned long long)v);
                ImGui::TableSetColumnIndex(2);
                if (cap)
                {
                    if (human) { HumanFormatter(double(cap), b1, (int)std::size(b1)); ImGui::Text("%s", b1); }
                    else       ImGui::Text("%llu", (unsigned long long)cap);
                }
                else ImGui::TextDisabled("-");
                if (warn) ImGui::PopStyleColor();
            };
            auto msRow = [&](const char* name, float ms, const char* tooltip)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%s", name);
                if (tooltip && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f ms", ms);
                ImGui::TableSetColumnIndex(2); ImGui::TextDisabled("-");
            };

            // Residency + this-frame load/unload activity.
            cntRow("Resident groups",   s.residentGroups,   s.maxGroups,
                   s.maxGroups   && s.residentGroups   >= s.maxGroups);   // red when slot pool exhausted
            cntRow("Resident clusters", s.residentClusters, s.maxClusters,
                   s.maxClusters && s.residentClusters >= s.maxClusters, /*human*/true);
            cntRow("Resident triangles", s.residentTriangles, 0, false, /*human*/true);
            memRow("Last transfer",     s.transferBytes,  s.maxTransferBytes);
            cntRow("Last loads",        s.loadCount,   s.maxLoadCount,   s.maxLoadCount   && s.loadCount   >= s.maxLoadCount);
            cntRow("Last unloads",      s.unloadCount, s.maxUnloadCount, s.maxUnloadCount && s.unloadCount >= s.maxUnloadCount);
            cntRow("Uncompleted loads", s.uncompletedLoadCount, 0, s.uncompletedLoadCount > 0);

            // ---- Streaming impact.  Accumulated over frames with streaming activity
            //      only, so it reads as the peak hitch, not the steady-state cost. --
            const float avgClasBuildMs = streamFrameCount
                ? float(sumStreamClasBuildMs / double(streamFrameCount)) : 0.f;
            msRow("CLAS build/frame (max)", maxStreamClasBuildMs,
                  "Peak single-frame cluster-LOD CLAS build time INCLUDING\n"
                  "allocation (freegaps / age filter / load / status / moves).\n"
                  "Tracked every frame (idle frames cost ~0), so it is the\n"
                  "worst-case streaming hitch.");
            msRow("CLAS build/frame (avg)", avgClasBuildMs,
                  "Average of the same CLAS-build-incl-allocation time, taken\n"
                  "only over frames with streaming activity (loads / unloads /\n"
                  "transfer) so idle frames don't dilute it.");
            memRow("Transfer/frame (max)", maxStreamTransferBytes, 0);

            ImGui::EndTable();
        }

        // Peaks span the current scene; a manual reset lets the user zero them
        // before a measured fly-through (e.g. --dolly) to capture that run only.
        ImGui::TextDisabled("Averaged over %llu streaming frame(s) since scene load / reset.",
                            (unsigned long long)streamFrameCount);
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset peaks"))
        {
            maxStreamClasBuildMs   = 0.f;
            sumStreamClasBuildMs   = 0.0;
            streamFrameCount       = 0;
            maxStreamTransferBytes = 0;
        }

        ImGui::Spacing();
    }

}  // end namespace stats
