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

// clang-format off

#pragma once

#include "rtxmg_demo.h"
#include "cluster_lod_budgets.h"
#include "rtxmg/cluster_tess/tessellator_config.h"
#include "rtxmg/scene/scene.h"

#include <array>
#include <string>
#include <json/json.h>

#include <donut/core/math/math.h>
#include <donut/app/StreamlineInterface.h>

using namespace donut::math;

// clang-format on

// All state & values controllable from the command-line
// used to initialize the application on launch.
//
// note: much of this state should be broken up into sub-sections
// note: currently the application owns this struct and can override
//       some of its values interactively (which is less than ideal).

// Things that can be modified by a scene.json file
struct SceneArgs
{
    std::array<std::string, TextureType::TEXTURE_TYPE_COUNT> textures;

    float envmapAzimuth = 0.f;
    float envmapElevation = 0.f;
    float envmapIntensity = 1.f;

    ColorMode colorMode = ColorMode::BASE_COLOR;
    ShadingMode shadingMode = ShadingMode::PT;
    TonemapOperator tonemapOperator = TonemapOperator::Aces;

    float wireframeThickness = .1f;

    int spp = 1;
    int ptMaxBounces = 2;

    float firefliesClamp = 0.f;
    float exposure = 1.f;
    // Auto-exposure (donut histogram eye-adaptation).  ON by default; exposure is
    // driven from average scene luminance and `exposure` above is compensation.
    // Disable with --no-auto-exposure.
    bool autoExposure = true;

    float dispScale = 1.f;
    float dispBias = 0.f;

    float roughnessOverride = 0.f;

    bool enableWireframe = false;
    bool enableDenoiser = true;
};

struct Args : SceneArgs
{
    // convenience accessors for legacy code
    using SceneArgs::colorMode;
    using SceneArgs::dispBias;
    using SceneArgs::dispScale;
    using SceneArgs::enableWireframe;
    using SceneArgs::envmapAzimuth;
    using SceneArgs::envmapElevation;
    using SceneArgs::envmapIntensity;
    using SceneArgs::exposure;
    using SceneArgs::firefliesClamp;
    using SceneArgs::ptMaxBounces;
    using SceneArgs::shadingMode;
    using SceneArgs::spp;
    using SceneArgs::wireframeThickness;

    SceneArgs& sceneArgs() { return *this; }

    int width = 1920;
    int height = 1080;
    bool resolutionSetByCmdLine = false;

    std::string outfile;
    std::string outfileWithUI;
    int exitAfterFrame = -1;

    // --dump-stats <file.json>: write the profiler timers, streaming/memory
    // blocks and the per-frame counter ladder at exit.  Consumed by the
    // regression harness as both the perf baseline and the counter golden.
    std::string dumpStatsFile;

    // argv rejoined, recorded in the --dump-stats run identity so a baseline is
    // never compared against a run configured differently.
    std::string commandLine;

    // --shot-list <file.json>: walk a list of capture points in one process,
    // writing a golden image per point per mode.  See demo/shot_list.h.
    std::string shotListFile;
    // --shot-out <dir>: the report directory, holding an images/ and a stats/
    // subdirectory.  Defaults to the shot list's own directory.
    std::string shotOutDir;

    // --texture-budget-mb <N>: cap total KTX2 texture GPU memory to N MB by
    // dropping the highest-resolution mips (greedy waterline); 0 = unlimited.
    // The default leaves headroom for the renderer's other fixed pools (CLAS,
    // cluster geometry, BLAS, render targets, DLSS); scenes that already fit
    // are unaffected.  Raise it for full-res texture runs on large-VRAM cards.
    int textureBudgetMB = 4096;

    // --normalmap-sweep [frames]: capture, enable normal maps, capture, disable,
    // capture -- each state settled for [frames] post-load frames.  Both edges go
    // through ApplyTextureSettingsAndReload, so this is the headless equivalent
    // of ticking the Normal Maps checkbox twice.
    bool     normalMapSweep = false;
    uint32_t normalMapSweepSettleFrames = 60;

    // --vram-mb <N>: pretend the card has N MB, for testing the VRAM Budget
    // window's clamps and plots on a machine that has more.  Budgeting only --
    // no allocation is capped by it.  0 = report the real adapter VRAM.
    int vramOverrideMB = 0;

    // -debugPixel <x> <y>: fire the viewport pick and the SHADER_DEBUG
    // predicates for that pixel and read both buffers back on the last frame
    // of -nf <N>.  (-1,-1) = disabled.
    int2 debugPixel = int2(-1, -1);

    // --dlssMode <m>: optional override for the default DLSS quality.
    // eDLAA gives 1:1 render-to-display ratio so -dp pixel coords match
    // the saved screenshot's pixel grid exactly. eUnknown = leave default.
    donut::app::StreamlineInterface::DLSSMode dlssMode =
        donut::app::StreamlineInterface::DLSSMode::eMaxQuality;
    bool dlssModeSet = false;
    std::string meshInputFile = std::string{};
    // Media/asset root for resolving relative scene/texture/envmap paths + the UI
    // asset browser.  Empty = auto: an explicit --media wins, else an ABSOLUTE
    // -mf implicitly sets it to that file's parent directory, else the "assets"
    // folder next to the executable.
    std::string mediaPath;
    std::string camString;

    uint4 edgeSegments = { 8, 8, 8, 8 };

    unsigned char quantNBits{ 0 };

    bool enableFrustumVisibility = true;
    bool enableBackfaceVisibility = true;
    bool enableHiZVisibility = true;
    // Camera every view-dependent geometry decision runs against: cluster-LoD
    // detail selection, cluster_tess rate, culling, HiZ occlusion + its pyramid.
    bool updateLodCamera = true;
    bool enableVertexNormals = false;     // tessellation (subd) per-vertex normals
    // Cluster-LoD baked per-vertex normals.  ON by default: keeps normals in the
    // resident group blobs (StreamingConfig::stripResidentNormals false), which
    // costs geometry-pool bytes -- see the 2 GB maxGeometryMegaBytes default.
    bool enableClusterLodVertexNormals = true;
    // Load cluster-LoD normal maps.  They are 42% of the flagship set's texture
    // bytes, so --no-normalmaps buys that back -- at the cost of a scene reload,
    // since what is read off disk can only change on one.
    bool enableNormalMaps = true;
    // Shade with them, once loaded.  A live toggle -- it only picks the path-tracer
    // permutation -- and only meaningful on top of the baked vertex normals above.
    bool normalMapShading = true;

    TessellatorConfig::MemorySettings tessMemorySettings;
    TessellatorConfig::VisibilityMode visMode = TessellatorConfig::VisibilityMode::VIS_LIMIT_EDGES;
    TessellatorConfig::AdaptiveTessellationMode tessMode = TessellatorConfig::AdaptiveTessellationMode::SPHERICAL_PROJECTION;

    // Note: the defaults here are intended for TMR
    int isoLevelSharp = 6;
    int isoLevelSmooth = 3;
    uint32_t globalIsolationLevel = TessellatorConfig::kMaxIsolationLevel;

    float fineTessellationRate = TessellatorConfig::kDefaultFineTessellationRate;
    float coarseTessellationRate = TessellatorConfig::kDefaultCoarseTessellationRate;

    // Adaptive tessellation rate: divide the tessellation rates while the
    // cluster_tess budgets run hot (same deadband as adaptiveLodError; never
    // finer than the rates above).  --no-adaptive-tess-rate opts out.
    bool adaptiveTessRate = true;
    ClusterTessPattern clusterPattern = ClusterTessPattern::SLANTED;

    // Cluster-LoD: target screen-space error (in pixels) at which the LoD
    // traversal stops descending. Threshold = 2*tan(fov/2) * lodPixelError / height.
    float lodPixelError = 1.0f;

    // Adaptive LoD error: raise the effective lodPixelError while the streaming
    // pools run hot (>85% smoothed load: +2%/frame; <70%: slow recovery; never
    // below lodPixelError) so the resident set fits the budgets instead of
    // churning.  --no-adaptive-error opts out.  Streaming mode only.
    bool adaptiveLodError = true;

    // Instance grid.
    // numCopies > 1 replicates the original instances — both cluster-LoD and
    // subdivision-mesh instances — with spacing lodGridGap * model_extent,
    // fanning out from the original (corner-anchored).
    //   lodGridBits:
    //     bits 0..2 = grid axes XYZ (default 0|2 = XZ plane)
    //     bits 3..5 = random rotation axes XYZ (default 3|5 = random axis in XZ)
    uint32_t lodGridCopies = 1;
    float    lodGridGap    = 1.0f;
    uint32_t lodGridBits   = 0x05u | 0x28u; // 0b101101 = 45: XZ grid + XZ rotation

    // --preload: opt into ClusterLodPreloaded, which uploads the full LoD
    // hierarchy at scene load.  Renders identically to the default streaming
    // path at steady state, but skips request-emit / staging / allocation.
    bool usePreload = false;

    // --linearalloc: replace the persistent CLAS allocator's 5-shader pipeline
    // with the compaction allocator (defrags resident CLAS to the pool base
    // each update frame).  Diagnostic: isolates Implicit→Move correctness from
    // persistent-allocator policy bugs.
    bool useLinearClasAllocator = false;

    // BLAS sharing: low-detail instances of a geometry reuse one
    // canonical instance's BLAS instead of each building their own.
    // On by default (keeps render-cluster demand under budget on
    // instance-heavy scenes); toggle off at runtime in the Cluster LODs UI.
    bool useBlasSharing = true;
    // Number of coarse tail LoD levels eligible for sharing
    // (sharingMinLevel = lodLevelsCount - this).  Higher = more aggressive
    // sharing (finer levels share too).  Optional arg to --blassharing;
    // default 8.
    uint32_t blasSharingEnabledLevels = 8;

    // BLAS caching: a fully-resident discrete-LoD-level BLAS per
    // geometry is built once into a persistent pool and reused across frames.
    // Requires --blassharing (caching ⊆ sharing); enabling --blascaching turns
    // sharing on implicitly.  On by default.
    bool useBlasCaching = true;
    // Coarse tail LoD levels eligible for caching (blasCacheMinLevel =
    // lodLevelsCount - this).  Optional arg to --blascaching; default 8.
    uint32_t blasCachingEnabledLevels = 8;

    // BLAS merging: all high-detail per-instance builds of a geometry are
    // collapsed into ONE merged BLAS per geometry, built at the highest resident
    // detail.  Requires --blassharing + streaming; enabling --blasmerging turns
    // sharing on implicitly.  Independent of --blascaching (both may be on).
    // On by default.
    bool useBlasMerging = true;

    // Frustum + HiZ culling (runtime).  Selected by the single `--culling <mode>`
    // option (off | soft | hard | invisible); each mode is a superset of the prior.
    //   off                   no culling.
    //   soft (default)        off-screen/occluded instances + nodes kept in the TLAS
    //                         but coarsened via culledErrorScale (RT-safe — no holes).
    //   hard                  off-screen instances not traversed → low-detail BLAS;
    //                         HiZ-occluded groups are hard-skipped in traversal.
    //   invisible             hard + remove: null the BLAS so the TLAS drops the
    //                         off-screen instances entirely (truly invisible).
    bool useCulling = true;
    bool useHardCull = false;
    bool hardCullForcesInvisible = false;

    // HiZ occlusion test inside the cull (--hiz-occlusion <on|off>).  Independent
    // of the frustum test, which keeps running when this is off — turn it off to
    // isolate occlusion-cull artifacts.
    bool useHizOcclusion = true;

    // --show-occlusion-depth: start with the HiZ occlusion-depth visualization as
    // the displayed output (same as the UI "Show Occlusion Depth" checkbox).  Lets
    // the HiZ be screenshotted headless.
    bool showOcclusionDepth = false;

    ClusterLodBudgets clodBudgets;

    // Mantissa bits the CLAS builder drops from each vertex position
    // (--claspositionbits). Shrinks the CLAS pool at the cost of position
    // precision; a --compress bake raises it to compressionPosDropBits for free.
    uint32_t clasPositionTruncateBits = 0;

    // Diagnostic: verbose Cluster-LoD streaming/traversal/BLAS readbacks.
    // Forces extra readbacks and, in some cases, wait-for-idle; slow.
    bool debugClusterLod = false;

    // --verbose / -v: per-geometry, per-LoD and per-texture logging.  Published
    // to rtxmg::g_verboseLogging so subsystems read it without plumbing.
    bool verboseLogging = false;

    // Bake cluster-LoD geometry with compressed (arithmetic-packed) vertex data
    // (--compress). Forwarded to the importer's BakerConfig. The runtime
    // decompresses per-group on load.
    bool compressClusterData = false;

    // Bake cluster size (--clustersize <tris> <verts>). 0 = keep the BakerConfig
    // default.  Smaller clusters multiply the group count and can push the
    // streaming working set past the CLAS pool.
    uint32_t clusterTriangles = 0;
    uint32_t clusterVertices  = 0;

    // Bake raw float2 texcoords instead of po2-grid-quantized (--no-uvquant).
    // A/B diagnostic for the UV quantizer.
    bool quantizeTexCoords = true;

    // Upload resident group blobs verbatim — no position/normal stripping
    // (--nostrip). Diagnostic: isolates the upload-time strip rewrite from
    // bake/shader-layout bugs (no rebake needed; upload-time only).
    bool stripResidentData = true;

    // Keep positions resident even where the AS position-fetch intrinsic is
    // available (--nostrippos), leaving the normals strip to the Vertex Normals
    // toggle.  The only way to exercise the keep-positions/drop-normals
    // combination — what a device without the intrinsic gets — on hardware that
    // does support position fetch.
    bool stripResidentPositions = true;

    // Bake simplifier attribute weights (--simplifyweights <normal> <texcoord>).
    // Negative = keep the BakerConfig defaults (0.5 / 0.5).  Attribute error is
    // folded into each group's baked maxQuadricError, which traversal tests
    // against lodPixelError — larger weights => finer LOD selection => bigger
    // streaming working set.
    float simplifyNormalWeight   = -1.f;
    float simplifyTexCoordWeight = -1.f;

    // Bake LOD error propagation (--loderrormerge <previous> <additive>).
    // Negative = keep the BakerConfig defaults.  meshoptimizer computes a
    // parent group's error as max(childError * previous, ownError) + additive *
    // ownError, and relies on it to be monotonically non-decreasing up the DAG,
    // so previous < 1 breaks that invariant.  Changes BakerConfig => full
    // rebake; pair with --cache-dir.
    float lodErrorMergePrevious = -1.f;
    float lodErrorMergeAdditive = -1.f;

    // Override the cluster-LoD shard cache directory (default: a shared
    // "_nvsngeocache" folder next to the gltf). Empty = default.
    // (--cache-dir <path>)
    //
    // The single place the bake flags' cache interaction is written down:
    // shards are named by INPUT hash only, so any flag that changes BakerConfig
    // (--clustersize, --simplifyweights, --compress, --no-uvquant) re-bakes them
    // in place and overwrites the copies baked at the old settings.  Point this
    // at a scratch dir to compare two configs; the load warns and names the
    // fields that changed either way.
    std::string clusterCacheDir;

    // --bake-workers: parallel bake worker count.  0 derives it from the core
    // count capped by physical RAM; lower it on a memory-constrained machine.
    uint32_t bakeWorkers = 0;

    // --nomat: force every cluster to opaque/single-sided/material 0 at runtime,
    // bypassing the alpha-mask geometry-index path so streaming-pressure races
    // can be isolated from it.  No re-bake required.
    bool enableMaterials = true;

    // --lpe-sweep: every lpeSweepFrameInterval frames, cycle lodPixelError
    // through 1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1 — a deterministic residency
    // add/evict stress driven by metric-scale changes alone.
    bool lpeSweep = false;
    uint32_t lpeSweepFrameInterval = 5;

    // --dolly: oscillate the camera along its look direction at the fly speed,
    // 1 s forward then 1 s back (triangle wave, looping).  Stresses the same
    // residency transitions as --lpe-sweep, but spatially.
    bool dolly = false;

    // --dolly-frames <N>: advance the dolly by a fixed 1/N of a leg per frame
    // instead of by elapsed wall-clock time.  0 = wall-clock (the default).
    // Wall-clock makes the camera path frame-rate dependent, so only the fixed
    // step can back an image or counter baseline.
    uint32_t dollyFrames = 0;

    // --anim-step <K>: advance subd animation by K keyframes per rendered frame
    // instead of by wall-clock time, so a run is reproducible.  0 = wall-clock.
    float animStep = 0.f;
    // --anim-dump <dir>: per frame, write render-res albedo + motion vectors as .npy.
    std::string animDumpDir;

    // --blas-toggle-sweep: cycle the BLAS sharing/caching toggles every N frames
    // once streaming has settled, reproducing the UI checkboxes headlessly.  The
    // settle delay matters: a toggle edge is only interesting against a populated
    // cached-BLAS pool and a converged resident set.  --blas-toggle-mode picks
    // which checkbox moves: `both` mirrors the UI (unchecking sharing force-clears
    // caching); `sharing`/`caching` hold the other one fixed to isolate an edge.
    enum class BlasToggleMode { Both, Sharing, Caching };
    bool blasToggleSweep = false;
    BlasToggleMode blasToggleMode = BlasToggleMode::Both;
    uint32_t blasToggleSweepFrameInterval = 10;
    uint32_t blasToggleSweepSettleFrames  = 60;

    // --tess-budget-sweep: cycle the cluster_tess Max Clusters budget every N
    // frames once the scene has settled, reproducing the Memory Settings sliders
    // headlessly.  A budget change reallocates the tessellator's BLAS/CLAS
    // storage, which the HiZ prepass may still be tracing.
    bool tessBudgetSweep = false;
    uint32_t tessBudgetSweepFrameInterval = 10;
    uint32_t tessBudgetSweepSettleFrames  = 30;

    // Camera fly speed in world units/second (WASD and --dolly).  0 = derive
    // from the scene's median instance size at load.
    float cameraSpeed = 0.f;
        
    float3 missColor = { .75, .75, .75 };

    // --test-rebake: after the initial load, trigger one GUI rebake with
    // compression on and compressionPosDropBits/TexDropBits bumped to 8.
    bool testRebake = false;

    bool debug = false;
    bool gpuValidation = false;
    bool aftermath = false;
    bool enableStreamlineLog = false;
    enum class ClusterApiRequest { Auto, Dxr, Nvapi };
    ClusterApiRequest clusterApi = ClusterApiRequest::Auto;
    bool enableAccelBuildLogging = false;
    bool enableTimeView = false;
    bool vsync = false;
    uint32_t maxFps = 0; // 0 = unlimited
    bool startMaximized = false;
    bool startFullscreen = false;

    void Parse(int argc, char const* const* argv);
};
Args& operator << (Args& args, const Json::Value& node);
// Re-assert command-line precedence over the fields a scene.json may set.
// Paired with operator<< — keep the two field lists in step.
void RestoreSceneOverridableFromCli(Args& args, const Args& cli);
