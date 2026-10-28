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

#include "rtxmg/profiler/stopwatch.h"
#include "rtxmg/profiler/profiler.h"

#include <nvrhi/utils.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/core/math/math.h>

#include <string>
#include <map>
#include <mutex>

#include "rtxmg/cluster_lod/shaderio.h"        // for shaderio::SceneBuildingCounters (BLAS rows)
#include "rtxmg/profiler/streaming_stats.h"    // for rtxmg::StreamingStats (Streaming tab)

// Only ever used as pointers here; including <imgui.h>/<implot.h> would make
// every consumer of this header link them.
struct ImFont;
struct ImPlotContext;

class UserInterface;

namespace stats
{
    using CPUTimer = Profiler::CPUTimer;
    using GPUTimer = Profiler::GPUTimer;
    //
    // SubD stats
    //
    struct TopologyMapStats
    {
        // hashmap
        float  pslMean = 0.f;
        size_t hashCount = 0;
        size_t addressCount = 0;
        float  loadFactor = 0.f;

        // plans
        uint32_t plansCount = 0;
        size_t   plansByteSize = 0;

        uint32_t regularFacePlansCount = 0;

        uint32_t maxFaceSize = 0;
        uint32_t sharpnessCount = 0;
        float    sharpnessMax;

        // patch points
        uint32_t stencilCountMin = 0;
        uint32_t stencilCountMax = 0;
        float stencilCountAvg = 0;
        std::vector<uint32_t> stencilCountHistogram;
    };

    struct SurfaceTableStats
    {
        std::string name;

        size_t indexBufferSize = 0;
        size_t vertCountBufferSize = 0;

        size_t byteSize = 0;
        size_t surfaceCount = 0;

        uint32_t irregularFaceCount = 0;
        uint32_t maxValence = 0;
        uint32_t maxFaceSize = 0;
        // | boundaries | stencils  | creases |
        uint32_t holesCount = 0;            // |            |           |         |
        uint32_t bsplineSurfaceCount = 0;   // |            |           |         |
        uint32_t regularSurfaceCount = 0;   // |     x      |           |         |
        uint32_t isolationSurfaceCount = 0; // |     X      |     X     |         |
        uint32_t sharpSurfaceCount = 0;     // |     X      |     X     |    X    |

        float sharpnessMax = 0.f;
        uint32_t infSharpCreases = 0;

        uint32_t stencilCountMin = ~uint32_t(0);
        uint32_t stencilCountMax = 0;
        float stencilCountAvg = 0;
        std::vector<uint32_t> stencilCountHistogram;

        std::vector<std::string> topologyRecommendations;

        bool IsCatmarkTopology(float* ratio = nullptr) const
        {
            // guess if the user passed a triangles mesh (ie. not a subd model)
            float _ratio = float(irregularFaceCount) / float(surfaceCount);
            if (ratio)
                *ratio = _ratio;
            return _ratio < .25f;
        }
        void BuildTopologyRecommendations();

        void BuildRecommendationsUI(ImFont *iconicFont) const;

        // Per-geometry detail, embedded by the Inspector under an expandable
        // summary row — which is why it renders no header of its own.
        void BuildDetailUI(ImFont *iconicFont, ImPlotContext *plotContext, uint32_t imguiID) const;
    };

    //
    // General stats
    //

    struct FrameSamplers
    {
        std::string name = "Frame";

        Sampler<float, Profiler::BENCH_FRAME_COUNT> cpuFrameTime = { .name = "CPU/frame (ms)" };

        // Wall time spent building the ImGui windows, on the render thread.  Split
        // out from cpuFrameTime because one expensive panel is easy to miss in the
        // frame total -- the Inspector's first-open sweep cost 2.2 s here.
        Sampler<float, Profiler::BENCH_FRAME_COUNT> uiBuildTime = { .name = "CPU/UI build (ms)" };

        GPUTimer& gpuFrameTime = Profiler::InitTimer<GPUTimer>("GPU/frame (ms)");
        GPUTimer& gpuRenderTime = Profiler::InitTimer<GPUTimer>("GPU/trace (ms)");
        GPUTimer& gpuDenoiserTime = Profiler::InitTimer<GPUTimer>("GPU/denoiser (ms)");
        
        GPUTimer& hiZRenderTime = Profiler::InitTimer<GPUTimer>("GPU/hi-z (ms)");
        GPUTimer& zReprojectionTime = Profiler::InitTimer<GPUTimer>("GPU/zReprojection (ms)");
        GPUTimer& zRenderPassTime = Profiler::InitTimer<GPUTimer>("GPU/zRenderPass (ms)");
        GPUTimer& computeMotionVectorsTimer = Profiler::InitTimer<GPUTimer>("GPU/motion vectors (ms)");
        GPUTimer& blitTime = Profiler::InitTimer<GPUTimer>("GPU/blit (ms)");

        // Per-frame Cluster Tess + Cluster LOD total, host-summed from their phase
        // sub-timers.  mutable because the const BuildUI is what pushes to it.
        mutable Sampler<float, Profiler::BENCH_FRAME_COUNT> accelBuildTime = { .name = "GPU/accel build (ms)" };

        void BuildUI(ImFont *iconicFont, ImPlotContext *plotContext) const;
        // A tab is hidden when the scene cannot produce its data; frame timings always can.
        bool TabEnabled() const { return true; }
    };
    extern FrameSamplers frameSamplers;

    // Push this frame's GPU frame / path-trace / DLSS durations into their
    // samplers.  Normally the profiler's Overview graph does it, so anything
    // running with that window closed has to call this itself.
    void ProfileFrameTimers();


    struct ClusterAccelSamplers
    {
        std::string name = "AccelBuilder";

        // ---- Cluster Tess (tessellation path) GPU timers --------------------
        GPUTimer& clusterTilingTime = Profiler::InitTimer<GPUTimer>("Cluster Tess/Tiling");
        GPUTimer& fillClustersTime  = Profiler::InitTimer<GPUTimer>("Cluster Tess/Fill");
        GPUTimer& buildClasTime     = Profiler::InitTimer<GPUTimer>("Cluster Tess/CLAS Build");
        GPUTimer& buildBlasTime     = Profiler::InitTimer<GPUTimer>("Cluster Tess/BLAS Build");

        // ---- Cluster LOD path GPU timers ------------------------------------
        // A GPUTimer spans one contiguous Start/Stop region, but the logical phases
        // are dispatched in disjoint interleaved regions — hence a sub-timer per
        // region, host-summed into the phase series below (see BuildLodUI).
        GPUTimer& clusterLodTraversalTime          = Profiler::InitTimer<GPUTimer>("Cluster Lod/Traversal (region)");
        GPUTimer& clusterLodClasBuildTime          = Profiler::InitTimer<GPUTimer>("Cluster Lod/CLAS Build (region)");
        GPUTimer& clusterLodClasMovePersistentTime = Profiler::InitTimer<GPUTimer>("Cluster Lod/CLAS Move persistent (region)");
        GPUTimer& clusterLodClasMoveCompactionTime = Profiler::InitTimer<GPUTimer>("Cluster Lod/CLAS Move compaction (region)");
        GPUTimer& clusterLodAllocUnloadUpdateTime  = Profiler::InitTimer<GPUTimer>("Cluster Lod/Alloc Unload+Update (region)");
        GPUTimer& clusterLodAllocFreegapsTime      = Profiler::InitTimer<GPUTimer>("Cluster Lod/Alloc Freegaps (region)");
        GPUTimer& clusterLodAllocAgeTime           = Profiler::InitTimer<GPUTimer>("Cluster Lod/Alloc Age Filter (region)");
        GPUTimer& clusterLodAllocLoadTime          = Profiler::InitTimer<GPUTimer>("Cluster Lod/Alloc Load Groups (region)");
        GPUTimer& clusterLodAllocStatusTime        = Profiler::InitTimer<GPUTimer>("Cluster Lod/Alloc Status (region)");
        GPUTimer& clusterLodBlasBuildTime          = Profiler::InitTimer<GPUTimer>("Cluster Lod/BLAS Build (region)");
        // Group-data / resident / update uploads recorded by StageResidencyUpdate, i.e.
        // the copies that feed a load.  Scales with camera motion, not residency.
        GPUTimer& clusterLodUploadTime             = Profiler::InitTimer<GPUTimer>("Cluster Lod/Upload (region)");
        // Scene refresh + instance-desc fill + TLAS build.  Shared with the
        // tessellated path, but still per-frame accel work.
        GPUTimer& tlasBuildTime              = Profiler::InitTimer<GPUTimer>("Accel/TLAS Build (region)");
        // GPU idle across the streaming path's mid-frame submit: how much of the
        // host's render-half recording the accel half could not cover.
        GPUTimer& clusterLodSubmitIdleTime         = Profiler::InitTimer<GPUTimer>("Cluster Lod/Submit idle (region)");
        // Host time in StageResidencyUpdate — request readback handling, group-data
        // staging, BLAS-cache bookkeeping.
        CPUTimer& clusterLodHostTime               = Profiler::InitTimer<CPUTimer>("Cluster Lod/Host streaming (ms)");

        // Per-frame phase series the AccelBuilder tab plots, summed from the
        // sub-timers above.  mutable because the const BuildUI pushes to them.
        mutable Sampler<float, Profiler::BENCH_FRAME_COUNT> clusterLodTraversal  = { .name = "Cluster Lod/Traversal" };
        mutable Sampler<float, Profiler::BENCH_FRAME_COUNT> clusterLodAllocation = { .name = "Cluster Lod/Allocation" };
        mutable Sampler<float, Profiler::BENCH_FRAME_COUNT> clusterLodClasBuild  = { .name = "Cluster Lod/CLAS Build" };
        mutable Sampler<float, Profiler::BENCH_FRAME_COUNT> clusterLodBlasBuild  = { .name = "Cluster Lod/BLAS Build" };
        mutable Sampler<float, Profiler::BENCH_FRAME_COUNT> clusterLodUpload     = { .name = "Cluster Lod/Upload" };

        // Cluster Tess geometry counts (one BLAS per instance, so unique == total).
        Sampler<uint32_t> numClusters = { .name = "Clusters", };
        Sampler<uint32_t> numTriangles = { .name = "Triangles", };

        // Tessellation build rate; mutable because the const BuildTessUI pushes to
        // it.  Left unnamed -- the empty name is the plot's series label.
        mutable Sampler<float> tessTrisPerSec;

        // Cluster LOD per-frame TLAS geometry, from SceneBuildingCounters
        // (TRACK_RENDER_STATS).  Unique = CLAS-deduped; Total = instanced.
        Sampler<uint32_t> clusterLodUniqueTriangles = { .name = "Unique Tris", };
        Sampler<uint32_t> clusterLodTotalTriangles  = { .name = "Total Tris", };
        Sampler<uint32_t> clusterLodUniqueClusters  = { .name = "Unique Clusters", };
        Sampler<uint32_t> clusterLodTotalClusters   = { .name = "Total Clusters", };

        donut::math::int2 renderSize = {};

        // Which accel paths ran this frame; drives which plot groups are shown.
        bool hasClusterTess = false;
        bool hasClusterLod  = false;

        // Called by ClusterTessTab / ClusterLodTab, which are separate Profiler tabs.
        void BuildTessUI(ImFont* iconicFont, ImPlotContext* plotContext) const;
        void BuildLodUI(ImFont* iconicFont, ImPlotContext* plotContext) const;

        // Host-sums the region sub-timers into the five phase series above, and
        // returns this frame's CLAS-build + allocation GPU time (ms), i.e. the
        // streaming cost.  Call exactly once per frame: Profile() drains the
        // sub-timer ring, so a second caller would read zeros.
        float ProfileClusterLodPhases() const;
    };
    extern ClusterAccelSamplers clusterAccelSamplers;

    // Thin wrappers that split the tessellation and cluster-LOD BVH stats into two
    // independent Profiler tabs, each hidden when its path isn't in the scene.
    struct ClusterTessTab
    {
        std::string name = "ClusterTess BVH";
        void BuildUI(ImFont* f, ImPlotContext* p) const { clusterAccelSamplers.BuildTessUI(f, p); }
        bool TabEnabled() const { return clusterAccelSamplers.hasClusterTess; }
    };
    extern ClusterTessTab clusterTessTab;

    struct ClusterLodTab
    {
        std::string name = "ClusterLOD BVH";
        void BuildUI(ImFont* f, ImPlotContext* p) const { clusterAccelSamplers.BuildLodUI(f, p); }
        bool TabEnabled() const { return clusterAccelSamplers.hasClusterLod; }
    };
    extern ClusterLodTab clusterLodTab;

    struct EvaluatorSamplers
    {
        std::string name = "Subdivision Evaluator";

        TopologyMapStats topologyMapStats;

        bool hasBadTopology = false;
        bool m_topologyQualityButtonPressed = false;
        // Set when the tab's "Inspector" button is clicked; the UI polls it to
        // open the Inspector window (per-geometry subdivision data lives there).
        mutable bool m_openInspectorRequested = false;
        size_t surfaceTablesByteSizeTotal = 0;

        std::vector<SurfaceTableStats> surfaceTableStats;


        // run-time evaluation

        Sampler<uint32_t> numLimitSamples = { .name = "Limit evaluations", };

        void BuildUI(ImFont* iconicFont, ImPlotContext* plotContext);
        // Only subdivision meshes (i.e. the tessellation path) produce this data.
        bool TabEnabled() const { return clusterAccelSamplers.hasClusterTess; }
    };
    extern EvaluatorSamplers evaluatorSamplers;

    // Material-texture memory for the Memory tab (the environment map is excluded,
    // it isn't budgeted).  Scene totals are measured once by ApplyKtxTextureBudget,
    // the loaded* fields refreshed as textures finalize.  Only KTX2 can be
    // mip-budgeted without decoding, so budgetable* covers just that subset.
    struct TextureMemStats
    {
        uint32_t textureCount    = 0;  // unique images the scene references, any format
        uint32_t budgetableCount = 0;  // of those, how many are KTX2
        uint32_t droppedCount    = 0;  // of the KTX2 ones, how many lost high-res mips
        uint32_t loadedCount     = 0;  // how many are resident on the GPU
        uint64_t diskBytes       = 0;  // their file bytes as stored on disk
        uint64_t budgetableFullBytes = 0;  // full-res GPU footprint of the KTX2 subset
        uint64_t keptBytes       = 0;  // GPU footprint the solver planned for that subset
        uint64_t budgetBytes     = 0;  // --texture-budget-mb (0 = unlimited)
        uint64_t loadedBytes     = 0;  // resident GPU bytes, any format
        uint64_t loadedOtherBytes = 0; // of loadedBytes, the non-KTX2 part
        // What the resident set occupies in VRAM: loadedBytes plus page and mip-tail
        // padding.  Everything else here stays packed, the unit the budget solves in.
        uint64_t vramBytes       = 0;

        // Padding the allocator adds on top of the packed sizes; carried onto
        // budget projections so they compare with the VRAM window's allocation.
        uint64_t VramPaddingBytes() const { return vramBytes > loadedBytes ? vramBytes - loadedBytes : 0; }

        // Full-resolution GPU footprint, the residency bar's denominator.  Only the
        // KTX2 half is known before decoding; other formats never drop mips, so a
        // resident one is already full size and tracks the numerator.
        uint64_t FullBytes() const { return budgetableFullBytes + loadedOtherBytes; }

        bool Valid() const { return textureCount != 0 || loadedCount != 0; }
    };

    // Baked cluster-LoD geometry totals, reduced from GeometryView::lodStats at
    // scene load.  This is the bake's own accounting, and the only way a test can
    // tell a --compress run that actually compressed from one whose flag never
    // reached BakerConfig: both render identically.
    struct BakeStats
    {
        uint64_t bakedBytes  = 0;   // sum of GroupInfo::sizeBytes, as stored
        uint64_t deviceBytes = 0;   // sum of GetDeviceSize(), unstripped
        uint64_t posBytes    = 0;
        uint64_t nrmBytes    = 0;
        uint64_t uvBytes     = 0;
        uint64_t triangles   = 0;
        uint32_t groups      = 0;
        uint32_t clusters    = 0;
        uint32_t geometries  = 0;
        bool     compressed  = false;  // any LoD arithmetic-packed on disk
        bool     quantizedUv = false;  // any cluster po2-grid quantized UVs
    };
    extern BakeStats bakeStats;

    struct MemUsageSamplers
    {
        std::string name = "Memory";

        Sampler<size_t> blasSize = { .name = "BLAS size", };
        Sampler<size_t> blasScratchSize = { .name = "BLAS Scratch", };
        Sampler<size_t> clasSize = { .name = "CLAS size", };
        Sampler<size_t> vertexBufferSize = { .name = "Vertex Buffer", };
        Sampler<size_t> vertexNormalsBufferSize = { .name = "Vertex Normals Buffer", };
        Sampler<size_t> clusterShadingDataSize = { .name = "Cluster Data Buffer", };

        TextureMemStats textures;

        void BuildUI(ImFont* iconicFont, ImPlotContext* plotContext) const;
        bool TabEnabled() const { return true; }
    };
    extern MemUsageSamplers memUsageSamplers;

    // Cluster-LOD streaming stats.  Everything here comes from the host-side
    // rtxmg::StreamingStats (ClusterLodStreaming::getStats), so no GPU readback.
    struct StreamingSamplers
    {
        std::string name = "Streaming";

        // Pool occupancy (levels), MB.
        Sampler<float> geometryMB = { .name = "Geometry (MB)" };
        Sampler<float> clasMB     = { .name = "CLAS (MB)" };

        // Disk-streaming rates, from per-frame deltas of the CUMULATIVE totals — the
        // latched last-batch counters never fall back to 0 when streaming goes idle.
        Sampler<float> transferRate = { .name = "Transfer/s" };
        Sampler<float> loadsPerSec  = { .name = "Loads/s" };
        Sampler<float> unloadsPerSec = { .name = "Unloads/s" };

        // Residency (levels).
        Sampler<uint32_t> residentGroups   = { .name = "Resident groups" };
        Sampler<uint32_t> residentClusters = { .name = "Resident clusters" };

        // BLAS effectiveness: per-instance BLAS built this frame, which drops as
        // sharing/caching/merging reuse them.  latestCounters backs the BLAS rows.
        Sampler<uint32_t>               blasBuilds = { .name = "BLAS builds" };
        shaderio::SceneBuildingCounters latestCounters = {};
        uint64_t                        latestBlasActualBytes = 0;

        // Previous cumulative totals, for the per-frame delta -> rate above.
        uint64_t prevTotalTransferBytes = 0;
        uint64_t prevTotalLoads         = 0;
        uint64_t prevTotalUnloads       = 0;

        // Streaming-impact peaks, accumulated only over frames with streaming
        // activity.  mutable so the const BuildUI's "Reset peaks" button can zero them.
        mutable float    maxStreamClasBuildMs   = 0.f;  // CLAS build incl. allocation
        mutable double   sumStreamClasBuildMs    = 0.0;  // running sum for the average
        mutable uint64_t streamFrameCount        = 0;    // frames the sum/max span
        mutable uint64_t maxStreamTransferBytes  = 0;    // max bytes moved in one frame

        // Latest full snapshot — backs the stats table rows + saturation warnings.
        rtxmg::StreamingStats latest = {};

        void BuildUI(ImFont* iconicFont, ImPlotContext* plotContext) const;
        // Streaming tab is only relevant for cluster-LOD scenes.
        bool TabEnabled() const { return clusterAccelSamplers.hasClusterLod; }
    };
    extern StreamingSamplers streamingSamplers;

    // Everything the sample allocates on the device, bucketed the way the VRAM
    // Budget window plots it.  Refreshed once a frame from the samplers above, so
    // it is a view of them rather than a second source of truth.
    //
    // Deliberately NOT a process total: nvrhi's descriptor heaps and the driver
    // allocate outside anything the sample can see.  RTXMGDemoApp::QueryDriverVram
    // supplies that, and the difference is the unaccounted gap.
    struct VramBreakdown
    {
        uint64_t textures      = 0;  // material textures resident on the GPU
        uint64_t envmap        = 0;  // envmap texture + importance-sampling tables
        // Cluster-LoD pools, as VRAM actually committed: the geometry and
        // cached-BLAS pools grow in blocks, so these are block multiples rather
        // than the sub-allocated totals the Memory tab reports.
        uint64_t clodGeometry  = 0;  // cluster-LoD resident geometry pool
        uint64_t clodClas      = 0;  // cluster-LoD CLAS pool
        uint64_t clodCachedBlas = 0;
        uint64_t clodCachedBlasBudget = 0;
        // Everything cluster-LoD outside the three pools: residency and
        // per-frame tables, the LOD hierarchies, and the pinned low-detail data.
        uint64_t clodMetadata  = 0;
        uint64_t tessVertices  = 0;  // cluster_tess vertex + normals buffers
        uint64_t tessClas      = 0;
        uint64_t tessClusterData = 0;
        // Subd meshes, keyframes, poses, topology maps and the tessellator's
        // per-instance / per-patch buffers.
        uint64_t tessSurfaces  = 0;

        // Occupancy inside the allocation above, for the pools that track it —
        // the same used/allocated pairs the profiler's Memory tab plots.  Never
        // summed into Accounted(): the allocation is what holds the VRAM.
        uint64_t clodGeometryUsed    = 0;
        uint64_t clodClasUsed        = 0;
        uint64_t tessVerticesUsed    = 0;
        uint64_t tessClasUsed        = 0;
        uint64_t tessClusterDataUsed = 0;
        uint64_t blasUsed            = 0;
        // BLAS storage of both paths at its allocated size, plus the TLAS and
        // its instance descs.
        uint64_t blas          = 0;
        // nvrhi's per-command-list AS build scratch pools; they never shrink.
        uint64_t asScratch     = 0;
        uint64_t dlss          = 0;  // Streamline's estimate for DLSS / DLSS-RR
        uint64_t renderTargets = 0;  // output textures, NRD set, HiZ chain, z-prepass

        // Driver-reported, when the API exposes it (D3D12 always, VK with
        // VK_EXT_memory_budget).  driverUsage covers everything, including what
        // the buckets above cannot see.
        bool     driverValid   = false;
        uint64_t driverUsage   = 0;
        uint64_t driverBudget  = 0;   // what the OS is currently willing to grant

        uint64_t Accounted() const
        {
            return textures + envmap + clodGeometry + clodClas + clodCachedBlas
                 + clodMetadata + tessVertices + tessClas + tessClusterData + tessSurfaces
                 + blas + asScratch + dlss + renderTargets;
        }
        // Nonzero only when the driver reading is available and above the stack.
        uint64_t Unaccounted() const
        {
            const uint64_t acc = Accounted();
            return (driverValid && driverUsage > acc) ? driverUsage - acc : 0;
        }
    };
    extern VramBreakdown vramBreakdown;

}  // end namespace stats
