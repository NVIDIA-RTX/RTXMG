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

// clang-format off

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// clang-format on

// Machine-readable export of everything the Profiler and the stats:: samplers
// collected during a run (--dump-stats).  A regression harness can consume it
// as both the perf baseline and the counter golden.

namespace stats
{

// Run identity the profiler cannot know on its own.  Recorded in the dump so a
// baseline is only ever compared against a run from a comparable configuration.
struct RunIdentity
{
    std::string commandLine;
    std::string buildConfig;
    std::string graphicsApi;
    std::string gpuName;
    std::string sceneFile;
    uint32_t    renderedFrames = 0;  // post-load frames, i.e. what -nf counts
    int32_t     renderWidth    = 0;
    int32_t     renderHeight   = 0;
};

// One row per rendered post-load frame.  These are integers derived from work
// counts, so they are exactly reproducible across runs of the same build and
// are compared with zero tolerance -- unlike the timings, which are not.
struct FrameRecord
{
    uint32_t frame            = 0;
    uint32_t renderedClusters = 0;
    uint32_t desiredClusters  = 0;
    uint32_t uniqueClusters   = 0;
    uint32_t totalClusters    = 0;
    uint32_t residentGroups   = 0;
    uint32_t residentClusters = 0;
    uint32_t blasBuilds       = 0;

    // Each timer's latest sample when the row is logged; written to `frameTimes`,
    // not `frameLog`.  GPU values lag the frame by the query read-back latency.
    float    cpuFrameMs       = 0.f;
    float    gpuFrameMs       = 0.f;
    float    hostStreamingMs  = 0.f;
};

// Off unless --dump-stats asked for it: a long run would grow this unboundedly.
struct FrameLog
{
    bool                     enabled = false;
    std::vector<FrameRecord> records;

    void Push(const FrameRecord& r)
    {
        if (enabled)
            records.push_back(r);
    }
};
extern FrameLog frameLog;

// One per image --shot-list wrote.  Recording the settle cost alongside the
// counters is what lets a later "why did this golden move?" separate a render
// change from the shot having been captured before streaming converged.
struct ShotRecord
{
    std::string label;
    std::string mode;
    std::string file;
    uint32_t    settleFrames     = 0;
    uint32_t    accumFrames      = 0;
    // Subframes the path tracer had accumulated at the capture.  The image is a
    // pure function of this (seed = TEA(16, pixel, subframe), weight
    // 1/(subframe+1)), so two runs that disagree here cannot match.
    uint32_t    subframeIndex    = 0;
    float       exposure         = 0.f;
    uint32_t    residentGroups   = 0;
    uint32_t    residentClusters = 0;
    uint32_t    uniqueClusters   = 0;
    uint64_t    totalTriangles   = 0;

    // Settings in force at the capture, so a sweep's images are attributable
    // without reconstructing them from the shot list.
    std::string dlssMode;
    int32_t     renderWidth    = 0;   // what the path tracer traced at
    int32_t     renderHeight   = 0;
    int32_t     outputWidth    = 0;   // what was presented, and what lodPixelError
    int32_t     outputHeight   = 0;   // is measured against
    float       lodPixelError  = 0.f;
    // What traversal actually used: the adaptive controller raises the static
    // error under budget pressure, so the two differ exactly when it kicked in.
    float       adaptiveLodPixelError = 0.f;
    // The shader permutation that ran, not what was asked for: false whenever
    // the run loaded no normal maps to shade with.
    bool        normalMapShading = false;

    // The Memory tab's two cluster-LoD pools, used vs budget.  Percentages are
    // redundant with the pairs but are the number the tab reads out.
    uint32_t    renderedClusters    = 0;
    uint64_t    geometryBytes       = 0;
    uint64_t    geometryBudgetBytes = 0;
    float       geometryPercent     = 0.f;
    uint64_t    clasBytes           = 0;
    uint64_t    clasBudgetBytes     = 0;
    float       clasPercent         = 0.f;

    // Whole-process video memory on the local adapter, vs what the OS is willing
    // to give this process.  Unlike the pools above this covers render targets,
    // DLSS internals, TLAS/scratch and driver overhead.  0 = not measured (VK).
    uint64_t    processVramBytes    = 0;
    uint64_t    vramBudgetBytes     = 0;
};
extern std::vector<ShotRecord> shotRecords;

// Writes the timers, samplers, streaming/memory blocks and the frame log to
// `path`.  Logs and returns false on failure; never throws into the shutdown
// path that calls it.
bool DumpToJson(const std::filesystem::path& path, const RunIdentity& run);

// Clears every timing window, leaving the timers' query state alone.  The
// samplers retain BENCH_FRAME_COUNT (400) frames -- more than a --shot-list
// capture point costs -- so without this each point's timings are blended with
// the two or three before it.
void ResetTimingSamplers();

}  // end namespace stats
