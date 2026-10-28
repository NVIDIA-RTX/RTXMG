/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "args.h"
#include "rtxmg_demo.h"

#include <assert.h>
#include <cstdio>
#include <functional>
#include <map>
#include <numbers>
#include <stdexcept>

#include "rtxmg/scene/json.h"
#include "rtxmg/utils/verbosity.h"

// clang-format on

static void printUsageAndExit(const char* argv0,
    const std::string& token = {})
{
    // clang-format off

    static char const* msg =
        "Usage  : %s [options]\n"
        "Options: \n"
        "  -h                      | --help                  Print this usage message\n"
        "  -at <mode>              | --adaptiveTessellation  Mode is: [uniform | world | sphere]\n"
        "                          | --aftermath             Enable Aftermath for GPU Crash debugging (not compatible with --debug)\n"
        "  -d3d12 | -dx12                                    Use the D3D12 backend (default)\n"
        "  -vk | -vulkan                                     Use the Vulkan backend\n"
        "  -ctr <f>                | --coarseTessellationRate Coarse adaptive edge sampling rate\n"
        "  -d                      | --debug                 Enable D3D12 Debug Layer\n"
        "  -gd                     | --gpudebug              Enable D3D12 GPU Validation + NVAPI ray tracing validation\n"
        "                          | --sllog                 Enable verbose Streamline (DLSS/denoiser) logging\n"
        "                          | --cluster-api <api>     D3D12 cluster operations: auto (default; DXR clusters when supported, else NVAPI) | dxr | nvapi\n"
        "  -v                      | --verbose               Log per-geometry / per-LoD / per-texture detail (~99% of the output on a large scene)\n"
        "  -ds <float>             | --displacementScale     Displacement scale along normal (relative to the largest extent of the object's AABB)\n"
        "  -ed                     | --enableDenoiser        Enable Denoiser\n"
        "  -envmap <envmap.exr>    |                         Specify the environment map to use\n"
        "  -es <e0> <e1> <e2> <e3> | --edgeSegments          Cluster edge m_size for all four edges (default 5 5 5 5)\n"
        "  -s <filename>           | --saveScreenshot        Save screenshot (no UI) to file upon exit\n"
        "  -sui <filename>         | --saveScreenshotWithUI  Save screenshot (with UI) to file upon exit\n"
        "  -dp <x> <y>             | --debugPixel            Set shader-debug predicate pixel (display coords; auto-scaled to render-target); also triggers debug-buffer readback on the last frame of -nf\n"
        "                          | --dlssMode <m>          DLSS quality mode: off | ultra | performance | balanced | quality | DLAA (1:1 render-to-display so -dp pixel coords match the screenshot exactly)\n"
        "  -ftr <f>                | --fineTessellationRate  Fine adaptive edge sampling rate\n"
        "                          | --adaptive-tess-rate    Cluster tess: scale the fine/coarse tessellation rates down on the GPU under tessellation-budget pressure (>85%% of max clusters/CLAS/vertex memory: one step down to 75%%, <60%%: slow recovery; never finer than -ftr/-ctr). ON BY DEFAULT; disable with --no-adaptive-tess-rate. Runtime-togglable in the UI\n"
        "                          | --auto-exposure         Auto-exposure: donut histogram eye-adaptation drives exposure from average scene luminance; the Exposure slider becomes an exposure-compensation multiplier. ON BY DEFAULT; disable with --no-auto-exposure. Runtime-togglable in the UI\n"
        "  -lpe <f>                | --lodPixelError         Cluster-LoD target screen-space error in pixels (default 1.0)\n"
        "                          | --adaptive-error        Cluster-LoD: raise the effective lodPixelError under streaming-budget pressure (>85%% load: +2%%/frame, <70%%: slow recovery; never below -lpe). ON BY DEFAULT; disable with --no-adaptive-error. Runtime-togglable in the UI\n"
        "  -gc <int>               | --gridCopies            Instance grid copy count, applies to cluster-LoD and subd meshes (default 1)\n"
        "  -gg <f>                 | --gridGap               Cluster-LoD instance grid spacing as multiple of model extent (default 1.0)\n"
        "                          | --preload               Cluster-LoD: use the preloaded path (upload the full hierarchy at scene load) instead of the default streaming path\n"
        "                          | --maxframeloadrequests <N>  Cluster-LoD streaming: max group load requests consumed per frame (default 256; higher converges faster after camera cuts but large bursts can stall frames)\n"
        "                          | --maxresidentgroups <N>  Cluster-LoD streaming: max simultaneously-resident groups (slot cap; default 131072). Raise for large scenes that wedge with byte pools under budget\n"
        "                          | --geometry-pool-mb <MB>  Cluster-LoD streaming: resident geometry pool size (default 2048). Sized at init, so it lands on a scene (re)load\n"
        "                          | --clas-pool-mb <MB>     Cluster-LoD streaming: resident CLAS pool size (default 2048). Sized at init, so it lands on a scene (re)load\n"
        "                          | --claspositionbits <N>  Cluster-LoD: mantissa bits the CLAS builder drops from vertex positions (default 0, clamp [0,22]). Shrinks the CLAS pool for coarser positions; --compress raises it to compressionPosDropBits at no precision cost\n"
        "                          | --render-cluster-bits <N>  Cluster-LoD per-frame render-cluster budget = 1<<N (default 20 = 1M, clamp [16,25]). Raise if 'Render cluster budget exceeded' flickers; ~16B/cluster. Init-time budget: changing it in the UI re-initializes the cluster-LoD resources\n"
        "  -bs                     | --blassharing [levels]  Cluster-LoD: BLAS sharing (low-detail instances reuse a canonical instance's BLAS). ON BY DEFAULT; disable with --no-blassharing. Optional [levels] = coarse tail LoD levels eligible for sharing (default 8; higher = more aggressive)\n"
        "  -bc                     | --blascaching [levels]  Cluster-LoD: BLAS caching (a per-geometry discrete-LoD BLAS is built once into a persistent pool and reused across frames). ON BY DEFAULT; disable with --no-blascaching. Implies --blassharing. Optional [levels] = coarse tail LoD levels eligible for caching (default 8)\n"
        "  -bm                     | --blasmerging           Cluster-LoD: BLAS merging (all high-detail per-instance builds of a geometry collapse into ONE merged BLAS per geometry). ON BY DEFAULT; disable with --no-blasmerging. Implies --blassharing; streaming-only. Composable with --blascaching\n"
        "                          | --no-blassharing        Cluster-LoD: disable BLAS sharing (also disables caching + merging, which are subsets of sharing)\n"
        "                          | --no-blascaching        Cluster-LoD: disable BLAS caching\n"
        "                          | --no-blasmerging        Cluster-LoD: disable BLAS merging\n"
        "                          | --culling <mode>        Cluster-LoD frustum+HiZ culling mode (default 'soft'):\n"
        "                          |                           off       = no culling\n"
        "                          |                           soft      = off-screen/occluded instances+nodes kept in TLAS but coarsened via culledErrorScale (DEFAULT; RT-safe)\n"
        "                          |                           hard      = off-screen instances not traversed → low-detail BLAS; HiZ-occluded groups hard-skipped\n"
        "                          |                           invisible = hard + REMOVE off-screen instances (null BLAS → dropped from the TLAS)\n"
        "                          | --hiz-occlusion <on|off> Cluster-LoD HiZ occlusion test inside the cull (default on; frustum test keeps running when off). --no-hiz-occlusion = off. Runtime-togglable in the UI\n"
        "                          | --show-occlusion-depth  Display the HiZ occlusion-depth visualization as output (headless screenshot of the HiZ; same as the UI checkbox)\n"
        "                          | --debug-clusterlod     Cluster-LoD: verbose streaming/traversal/BLAS readbacks (very slow)\n"
        "                          | --compress             Cluster-LoD: bake compressed (arithmetic-packed) vertex data; rebakes the .nvsngeo cache, decompressed per-group at runtime\n"
        "                          | --cache-dir <dir>      Cluster-LoD shard cache directory (alias --nvsngeocache). Without it the baker writes a '_nvsngeocache' folder next to the glTF, so a large scene cold-bakes there. A cache baked with different bake settings is silently re-baked in place\n"
        "                          | --bake-workers <n>     Cluster-LoD bake: parallel worker count (default 0 = core count capped by physical RAM). Each worker holds a decode scratch, the extracted geometry and the baker's working set at once, so lower it if a large cold bake exhausts memory\n"
        "                          | --simplifyweights <n> <t>  Cluster-LoD bake: simplifier normal and texcoord attribute weights (default 0.5 0.5). Changes the bake, so an existing cache is re-baked\n"
        "                          | --loderrormerge <prev> <add>  Cluster-LoD bake: LoD error propagation, previous-scale and additive terms (default 1.0 0.0). Changes the bake, so an existing cache is re-baked\n"
        "                          | --texture-budget-mb <MB>  Cap resident texture memory by dropping mips at load (0 = no cap)\n"
        "                          | --camera-speed <f>     Camera fly speed (WASD) in world units/second.  Default: derived from the scene's median instance size.\n"
        "  -isoLevelSharp <int>                              Max isolation level near sharp features such as creases (default 6)\n"
        "  -isoLevelSmooth <int>                             Max isolation level near smooth features such as extraordinary vertices (default 3)\n"
        "                          | --logAccelBuild         Log each buffer of the acceleration build (slow!)\n"
        "                          | --maxfps <int>          Limit rendering to N frames per second. 0 disables the cap.\n"
        "                          | --maxClusters <n>       Tessellation memory budget: max simultaneously live clusters\n"
        "                          | --vertBufferMB <MB>     Tessellation memory budget: vertex buffer size\n"
        "                          | --clasBufferMB <MB>     Tessellation memory budget: CLAS buffer size\n"
        "  -mrl <int>              | --maxRefinementLevel    Legacy: same as isoLevelSharp\n"
        "  -mc <r> <g> <b>         | --missColor             Miss color\n"
        "  -mf <filepath>          | --meshInputFile         Read .obj or scene file (an absolute path also sets the media folder to its parent dir)\n"
        "                          | --media <dir>           Media/asset root for relative scene/texture/envmap paths + UI browser (default: 'assets' next to the exe, or an absolute -mf's parent)\n"
        "  -p \"[eye][at][up]fov\"   | --cameraPos             Camera pose\n"
        "  -ptmb <n>               | --ptMaxBounces          Max PT bounces\n"
        "  -res <w> <h>            | --resolution            Set image dimensions to <w>x<h> (default 768 768)\n"
        "  -sm [prim_rays|ao|pt]   | --shadingMode           primary rays or AO or path tracing\n"
        "  -cm [base|uv|cid|tid|n|texcoord|mat|tArea|lod|group|blas|blascached]\n"
        "                          | --colorMode             color visualization mode (default: base)\n"
        "                                                    lod = colorize cluster-LOD hits by LOD level\n"
        "                                                    group = colorize by cluster group\n"
        "                                                    blas = red on coarsest cluster (low-detail BLAS), hashed otherwise\n"
        "                                                    blascached = highlight hits served by a cached BLAS\n"
        "  -spp <n>                                          Number of samples per pixel (need to use a perfect square number, default 1)\n"
        "  -nf <n>                 | --nframes               Exit after rendering N frames (default disabled)\n"
        "                          | --dump-stats <file>     Write profiler timings, streaming/memory stats and the per-frame counter ladder to a JSON file at exit\n"
        "                          | --shot-list <file>      Walk a JSON list of capture points (camera, exposure, render modes), settling streaming and accumulation at each, and write a screenshot per point per mode.  Exits when done\n"
        "                          | --shot-out <dir>        Report directory --shot-list writes into: images/ for the captures, stats/ for the per-location dumps.  Default: the shot list's own directory\n"
        "  -tv <true|false>        | --timeview              Enable Timeview (default false)\n"
        "  -vn <true|false>        | --vertexNormals         Enable tessellation (subd) vertex normals computation (default false)\n"
        "  -cvn <true|false>       | --clusterLodVertexNormals     Enable cluster-LoD baked vertex normals shading (default TRUE; keeps normals resident, sized for by the 2 GB geometry pool)\n"
        "                          | --normalmaps            Load cluster-LoD normal maps (the default). Needs -cvn. Loading is a scene-reload setting; shading with them is live\n"
        "                          | --no-normalmaps         Skip loading normal maps, buying back the ~42%% of a large scene's texture bytes they occupy\n"
        "                          | --normalmapshading <b>  Shade with the loaded normal maps, using a tangent frame derived from the hit triangle's du/dv (default on; no effect under --no-normalmaps)\n"
        "                          | --vsync <true|false>    Enable vertical sync (default false)\n"
        "  -wf <true|false>        | --wireframe             Set wireframe (default true)\n"
        "  -wm                     | --windowMaximized       Start window maximized (default false)\n"
        "  -fs                     | --startFullscreen       Start as borderless fullscreen window\n"
#if RTXMG_DEV_FEATURES
        "\n"
        "Developer options (RTXMG_DEV_FEATURES build only):\n"
        "                          | --linearalloc [MB]      Cluster-LoD streaming: replace the persistent CLAS allocator with the compaction allocator. Optional [MB] caps the CLAS pool (maxClasMegaBytes, default 2048) to test compaction under a tight budget\n"
        "                          | --nomat                 Cluster-LoD: force every cluster to opaque/single-sided/material 0 (bypasses the alpha-mask geometry-indices path)\n"
        "                          | --clustersize <t> <v>   Cluster-LoD bake: cluster granularity in triangles and vertices. Changes the bake, so an existing cache is re-baked\n"
        "                          | --nostrip               Cluster-LoD streaming: upload resident group blobs verbatim, with no position/normal stripping\n"
        "                          | --nostrippos            Cluster-LoD streaming: keep positions resident but let normals strip\n"
        "                          | --no-uvquant            Cluster-LoD bake: raw float2 texcoords instead of po2-grid-quantized. Changes the bake, so an existing cache is re-baked\n"
        "                          | --lpe-sweep [frames]    Cycle lodPixelError through 1,2,4,8,16,32,16,8,4,2,1 every N frames (default 5; loops)\n"
        "                          | --dolly                 Smooth linear dolly oscillating 1 s forward then 1 s back (loops) at the camera fly speed; reproduces spatial-LoD streaming load/evict transitions\n"
        "                          | --dolly-frames <n>      Like --dolly, but advances a fixed 1/n of a leg per frame instead of by wall-clock time, so the camera path is reproducible and can back a baseline\n"
        "                          | --anim-step <k>         Play subd animation at a fixed k keyframes per rendered frame instead of by wall-clock time, so a run is reproducible\n"
        "                          | --anim-dump <dir>       Each frame, write the render-res albedo (alb_NNNN.npy) and motion vectors (mv_NNNN.npy) to <dir>. Stalls the GPU every frame\n"
        "                          | --blas-toggle-sweep [frames]  Cycle the BLAS sharing/caching toggles every N frames (default 10; loops) once streaming has settled, as if clicking the UI checkboxes\n"
        "                          | --blas-toggle-settle <frames> Frames to let the scene settle after load before --blas-toggle-sweep starts toggling (default 60)\n"
        "                          | --blas-toggle-mode <mode>  Which checkbox --blas-toggle-sweep drives: 'both' (default; mirrors the UI, unchecking sharing clears caching), 'sharing' or 'caching' (holds the other fixed to isolate one edge)\n"
        "                          | --tess-budget-sweep [frames]  Cycle the cluster_tess Max Clusters budget every N frames (default 10; loops) once the scene has settled, as if dragging the Memory Settings slider\n"
        "                          | --tess-budget-settle <frames> Frames to let the scene settle after load before --tess-budget-sweep starts (default 30)\n"
        "                          | --vram-mb <MB>          Report <MB> as the card's VRAM in the VRAM Budget window, to test its clamps and plots on a bigger card. Budgeting only - no allocation is capped\n"
        "                          | --normalmap-sweep [frames]  Screenshot, enable normal maps (scene reload), screenshot, disable, screenshot -- settling [frames] post-load frames at each state (default 60). Writes nmsweep_<n>_<state>.png and exits\n"
#endif
        ;

    // clang-format on

    std::fprintf(stderr, msg, argv0);

    if (token.find("Unknown option") != std::string::npos)
        std::fprintf(stderr, "\n****** %s ******", token.c_str());
    else if (!token.empty())
        std::fprintf(stderr, "\n****** Invalid usage of '%s' ******",
            token.c_str());
    exit(1);
}

static bool parseBooleanArg(char const* token,
    char const* value, bool defaultValue)
{
    if (std::strncmp(value, "true", 4) == 0)
        return true;
    if (std::strncmp(value, "false", 5) == 0)
        return false;
    printUsageAndExit("rtxmg_demo", token);
    return defaultValue;
}

void Args::Parse(int argc, char const* const* argv)
{
    static std::map<std::string, ShadingMode> shadingModes{
        {"prim_rays", ShadingMode::PRIMARY_RAYS},
        {"ao", ShadingMode::AO},
        {"pt", ShadingMode::PT} };

    static std::map<std::string, ColorMode> colorModes{
        {"base", ColorMode::BASE_COLOR},
        {"uv", ColorMode::COLOR_BY_CLUSTER_UV},
        {"cid", ColorMode::COLOR_BY_CLUSTER_ID},
        {"tid", ColorMode::COLOR_BY_MICROTRI_ID},
        {"n", ColorMode::COLOR_BY_SHADING_NORMAL},
        {"texcoord", ColorMode::COLOR_BY_TEXCOORD},
        {"mat", ColorMode::COLOR_BY_MATERIAL},
        {"tArea", ColorMode::COLOR_BY_MICROTRI_AREA},
        {"lod", ColorMode::COLOR_BY_LOD_LEVEL},
        {"group", ColorMode::COLOR_BY_CLUSTER_GROUP},
        {"blas", ColorMode::COLOR_BY_BLAS_SOURCE},
        {"blascached", ColorMode::COLOR_BY_BLAS_CACHED} };

    static const std::map<std::string, TextureType> TextureTypes = {
        { "-envmap", TextureType::ENVMAP }
    };

    static std::map <std::string, TessellatorConfig::AdaptiveTessellationMode> adaptiveTessellationModes = {
        {"uniform", TessellatorConfig::AdaptiveTessellationMode::UNIFORM},
        {"sphere", TessellatorConfig::AdaptiveTessellationMode::SPHERICAL_PROJECTION},
        {"world", TessellatorConfig::AdaptiveTessellationMode::WORLD_SPACE_EDGE_LENGTH}
    };

    auto parseEnum = [&argv]<typename T>(std::string const& arg, std::map<std::string, T> const& enumModes)
    {
        if (const auto it = enumModes.find(arg); it != enumModes.end())
            return it->second;
        else
            printUsageAndExit(argv[0], arg);
        return T(0);
    };

    // Assigned, not appended: Parse runs again on scene load (G-2).
    commandLine.clear();
    for (int i = 0; i < argc; ++i)
        commandLine += (i ? " " : "") + std::string(argv[i]);

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);

        if (arg == "-d3d12" || arg == "-dx12" || arg == "--d3d12" || arg == "--dx12")
            continue;
        
        if (arg == "-vk" || arg == "-vulkan" || arg == "--vk" || arg == "--vulkan")
            continue;

        auto parseArgValues = [&argc, &argv, &i, &arg](int n,
            std::function<void()> func)
            {
                if (i >= argc - n)
                    printUsageAndExit(argv[0], argv[i]);
                func();
            };

        if (arg == "--debug" || arg == "-d")
        {
            debug = true;
        }
        else if (arg == "--gpudebug" || arg == "-gd")
        {
            gpuValidation = true;
        }
        else if (arg == "--aftermath")
        {
            aftermath = true;
        }
        else if (arg == "--sllog")
        {
            enableStreamlineLog = true;
        }
        else if (arg == "--cluster-api")
        {
            static const std::map<std::string, ClusterApiRequest> clusterApis{
                {"auto", ClusterApiRequest::Auto},
                {"dxr", ClusterApiRequest::Dxr},
                {"nvapi", ClusterApiRequest::Nvapi} };
            parseArgValues(1, [&]() { clusterApi = parseEnum(argv[++i], clusterApis); });
        }
        else if (arg == "--help" || arg == "-h")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--saveScreenshot" || arg == "-s")
        {
            parseArgValues(1, [&]() { outfile = argv[++i]; });
        }
        else if (arg == "--saveScreenshotWithUI" || arg == "-sui")
        {
            parseArgValues(1, [&]() { outfileWithUI = argv[++i]; });
        }
        else if (arg == "--nframes" || arg == "-nf")
        {
            parseArgValues(1, [&]() { exitAfterFrame = atoi(argv[++i]); });
        }
        else if (arg == "--dump-stats")
        {
            parseArgValues(1, [&]() { dumpStatsFile = argv[++i]; });
        }
        else if (arg == "--shot-list")
        {
            parseArgValues(1, [&]() { shotListFile = argv[++i]; });
        }
        else if (arg == "--shot-out")
        {
            parseArgValues(1, [&]() { shotOutDir = argv[++i]; });
        }
        else if (arg == "--texture-budget-mb")
        {
            parseArgValues(1, [&]() { textureBudgetMB = atoi(argv[++i]); });
        }
        else if (arg == "--debugPixel" || arg == "-dp")
        {
            parseArgValues(2, [&]()
                {
                    debugPixel.x = atoi(argv[++i]);
                    debugPixel.y = atoi(argv[++i]);
                });
        }
        else if (arg == "--dlssMode")
        {
            parseArgValues(1, [&]()
                {
                    static const std::map<std::string, donut::app::StreamlineInterface::DLSSMode> modes{
                        {"off",           donut::app::StreamlineInterface::DLSSMode::eOff},
                        {"ultra",         donut::app::StreamlineInterface::DLSSMode::eUltraPerformance},
                        {"performance",   donut::app::StreamlineInterface::DLSSMode::eMaxPerformance},
                        {"balanced",      donut::app::StreamlineInterface::DLSSMode::eBalanced},
                        {"quality",       donut::app::StreamlineInterface::DLSSMode::eMaxQuality},
                        {"DLAA",          donut::app::StreamlineInterface::DLSSMode::eDLAA},
                        {"dlaa",          donut::app::StreamlineInterface::DLSSMode::eDLAA},
                    };
                    dlssMode = parseEnum(argv[++i], modes);
                    dlssModeSet = true;
                });
        }
        else if (arg == "--timeview" || arg == "-tv")
        {
            parseArgValues(1, [&]() { enableTimeView = parseBooleanArg(arg.c_str(), argv[++i], enableTimeView); });
        }
        else if (arg == "-p" || arg == "--cameraPos")
        {
            parseArgValues(1, [&]() { camString = std::string(argv[++i]); });
        }
        else if (arg == "-res" || arg == "--resolution")
        {
            parseArgValues(2, [&]()
                {
                    width = atoi(argv[++i]);
                    height = atoi(argv[++i]);
                });
            resolutionSetByCmdLine = true;
        }
        else if (arg == "--wireframe" || arg == "-wf")
        {
            parseArgValues(1, [&]()
                {
                    enableWireframe =
                        parseBooleanArg(arg.c_str(), argv[++i], enableWireframe);
                });
        }
        else if (arg == "--displacementScale" || arg == "-ds")
        {
            parseArgValues(1, [&]() { dispScale = (float)atof(argv[++i]); });
        }
        else if (arg == "--adaptiveTessellation" || arg == "-at")
        {
            parseArgValues(
                1, [&]() { tessMode = parseEnum(argv[++i], adaptiveTessellationModes); });
        }
        else if (arg == "--edgeSegments" || arg == "-es")
        {
            parseArgValues(4, [&]()
            {
                edgeSegments.x = static_cast<uint32_t>(atoi(argv[++i]));
                edgeSegments.y = static_cast<uint32_t>(atoi(argv[++i]));
                edgeSegments.z = static_cast<uint32_t>(atoi(argv[++i]));
                edgeSegments.w = static_cast<uint32_t>(atoi(argv[++i]));
            });
        }
        else if (arg == "--missColor" || arg == "-mc")
        {
            parseArgValues(3, [&]()
                {
                    missColor = { (float)atof(argv[++i]), (float)atof(argv[++i]),
                                 (float)atof(argv[++i]) };
                });
        }
        else if (arg == "--ptMaxBounces" || arg == "-ptmb")
        {
            parseArgValues(1, [&]() { ptMaxBounces = atoi(argv[++i]); });
        }
        else if (arg == "--meshInputFile" || arg == "-mf")
        {
            parseArgValues(1, [&]() { meshInputFile = argv[++i]; });
        }
        else if (arg == "--media" || arg == "--mediapath" || arg == "--mediafolder")
        {
            parseArgValues(1, [&]() { mediaPath = argv[++i]; });
        }
        else if (arg == "--maxRefinementLevel" || arg == "-mrl" ||
            arg == "-isoLevelSharp")
        {
            parseArgValues(1, [&]() { isoLevelSharp = atoi(argv[++i]); });
        }
        else if (arg == "-isoLevelSmooth")
        {
            parseArgValues(1, [&]() { isoLevelSmooth = atoi(argv[++i]); });
        }
        else if (arg == "-envmap")
        {
            try
            {
                textures[parseEnum(arg, TextureTypes)] = argv[++i];
            }
            catch (std::exception& e)
            {
                std::fprintf(stderr, "Invalid option in '%s %s' : %s\n", arg.c_str(), argv[i], e.what());
                exit(1);
            }
        }
        else if (arg == "--shadingMode" || arg == "-sm")
        {
            parseArgValues(
                1, [&]() { shadingMode = parseEnum(argv[++i], shadingModes); });
        }
        else if (arg == "--colorMode" || arg == "-cm")
        {
            parseArgValues(
                1, [&]() { colorMode = parseEnum(argv[++i], colorModes); });
        }
        else if (arg == "-spp")
        {
            parseArgValues(1, [&]()
                {
                    spp = atoi(argv[++i]);
                    const int sqrt_spp =
                        static_cast<int>(std::sqrt(static_cast<float>(spp)));
                    if (sqrt_spp * sqrt_spp !=
                        spp) // check if spp is a perfect sqare number
                        printUsageAndExit(argv[0], arg);
                });
        }
        else if (arg == "--logAccelBuild")
        {
            enableAccelBuildLogging = true;
        }
        else if (arg == "--maxfps" || arg == "--max-fps")
        {
            parseArgValues(1, [&]() { maxFps = uint32_t(std::max(0, atoi(argv[++i]))); });
        }
        else if (arg == "--vertBufferMB")
        {
            parseArgValues(1, [&]() { tessMemorySettings.vertexBufferBytes = size_t(atoi(argv[++i])) << 20ull; });
        }
        else if (arg == "--maxClusters")
        {
            parseArgValues(1, [&]() { tessMemorySettings.maxClusters = atoi(argv[++i]); });
        }
        else if (arg == "--clasBufferMB")
        {
            parseArgValues(1, [&]() { tessMemorySettings.clasBufferBytes = size_t(atoi(argv[++i])) << 20ull; });
        }
        else if (arg == "--fineTessellationRate" || arg == "-ftr")
        {
            parseArgValues(1, [&]() { fineTessellationRate = (float)atof(argv[++i]); });
        }
        else if (arg == "--coarseTessellationRate" || arg == "-ctr")
        {
            parseArgValues(1, [&]() { coarseTessellationRate = (float)atof(argv[++i]); });
        }
        else if (arg == "--adaptive-tess-rate")
        {
            adaptiveTessRate = true;
        }
        else if (arg == "--no-adaptive-tess-rate")
        {
            adaptiveTessRate = false;
        }
        else if (arg == "--lodPixelError" || arg == "-lpe")
        {
            parseArgValues(1, [&]() { lodPixelError = (float)atof(argv[++i]); });
        }
        else if (arg == "--auto-exposure" || arg == "--autoexposure")
        {
            autoExposure = true;
        }
        else if (arg == "--no-auto-exposure" || arg == "--no-autoexposure")
        {
            autoExposure = false;
        }
        else if (arg == "--adaptive-error" || arg == "--adaptiveerror")
        {
            adaptiveLodError = true;
        }
        else if (arg == "--no-adaptive-error" || arg == "--no-adaptiveerror")
        {
            adaptiveLodError = false;
        }
        else if (arg == "--gridCopies" || arg == "-gc")
        {
            parseArgValues(1, [&]() { lodGridCopies = uint32_t(std::max(1, atoi(argv[++i]))); });
        }
        else if (arg == "--gridGap" || arg == "-gg")
        {
            parseArgValues(1, [&]() { lodGridGap = (float)atof(argv[++i]); });
        }
        else if (arg == "--preload")
        {
            usePreload = true;
        }
#if RTXMG_DEV_FEATURES
        else if (arg == "--linearalloc")
        {
            useLinearClasAllocator = true;
            // Optional CLAS-pool cap in MB (e.g. "--linearalloc 64"), to check
            // that compaction bounds usage to the peak resident working set.
            if (i + 1 < argc && argv[i + 1][0] != '-')
                clodBudgets.clasPoolOverrideMB = uint32_t(std::max(1, atoi(argv[++i])));
        }
        else if (arg == "--nomat" || arg == "--no-materials")
        {
            enableMaterials = false;
        }
#endif
        else if (arg == "--maxframeloadrequests" || arg == "--max-frame-load-requests")
        {
            parseArgValues(1, [&]() { clodBudgets.maxFrameLoadRequests = uint32_t(std::max(1, atoi(argv[++i]))); });
        }
        else if (arg == "--claspositionbits" || arg == "--clas-position-bits")
        {
            // 22 of 23 mantissa bits is the builder's practical ceiling.
            parseArgValues(1, [&]() { clasPositionTruncateBits = uint32_t(std::clamp(atoi(argv[++i]), 0, 22)); });
        }
        else if (arg == "--maxresidentgroups" || arg == "--max-resident-groups")
        {
            parseArgValues(1, [&]() { clodBudgets.maxResidentGroups = uint32_t(std::max(1, atoi(argv[++i]))); });
        }
        else if (arg == "--geometry-pool-mb" || arg == "--geometrypoolmb")
        {
            parseArgValues(1, [&]() { clodBudgets.maxGeometryMB = uint32_t(std::max(1, atoi(argv[++i]))); });
        }
        else if (arg == "--clas-pool-mb" || arg == "--claspoolmb")
        {
            parseArgValues(1, [&]() { clodBudgets.maxClasMB = uint32_t(std::max(1, atoi(argv[++i]))); });
        }
        else if (arg == "--render-cluster-bits" || arg == "--renderclusterbits")
        {
            // Per-frame render-cluster budget = 1<<bits.  Clamp [16,25] (matches
            // the UI slider + ClusterLodPass kMin/kMaxRenderClusterBits).
            parseArgValues(1, [&]() { clodBudgets.renderClusterBits = uint32_t(std::clamp(atoi(argv[++i]), 16, 25)); });
        }
        else if (arg == "--blassharing" || arg == "--blas-sharing" || arg == "-bs")
        {
            useBlasSharing = true;
            // Optional sharing-aggressiveness arg (coarse tail levels eligible
            // for sharing), e.g. "--blassharing 14".  Defaults to 8.
            if (i + 1 < argc && argv[i + 1][0] != '-')
                blasSharingEnabledLevels = uint32_t(std::max(0, atoi(argv[++i])));
        }
        else if (arg == "--blascaching" || arg == "--blas-caching" || arg == "-bc")
        {
            // Caching ⊆ sharing — turn sharing on implicitly so the user only
            // needs one flag.
            useBlasCaching = true;
            useBlasSharing = true;
            // Optional caching-aggressiveness arg (coarse tail levels eligible
            // for caching), e.g. "--blascaching 14".  Defaults to 8.
            if (i + 1 < argc && argv[i + 1][0] != '-')
                blasCachingEnabledLevels = uint32_t(std::max(0, atoi(argv[++i])));
        }
        else if (arg == "--blasmerging" || arg == "--blas-merging" || arg == "-bm")
        {
            // Merging requires sharing (and streaming) — turn sharing on
            // implicitly so the user only needs one flag.
            useBlasMerging = true;
            useBlasSharing = true;
        }
        else if (arg == "--no-blassharing" || arg == "--no-blas-sharing")
        {
            // Sharing is on by default; this opts out.  Caching + merging are
            // subsets of sharing, so disable them too (mirrors the UI toggle).
            useBlasSharing = false;
            useBlasCaching = false;
            useBlasMerging = false;
        }
        else if (arg == "--no-blascaching" || arg == "--no-blas-caching")
        {
            useBlasCaching = false;
        }
        else if (arg == "--no-blasmerging" || arg == "--no-blas-merging")
        {
            useBlasMerging = false;
        }
        else if (arg == "--culling" || arg == "--cull")
        {
            // The mode picks a point on the off→soft→hard→invisible escalation;
            // each is a superset of the previous.  See Args::useCulling.
            enum class CullMode { Off, Soft, Hard, Invisible };
            static const std::map<std::string, CullMode> cullModes{
                { "off",       CullMode::Off },        { "none", CullMode::Off },
                { "soft",      CullMode::Soft },       { "on",   CullMode::Soft },
                { "hard",      CullMode::Hard },
                { "invisible", CullMode::Invisible },
            };
            parseArgValues(1, [&]() {
                CullMode mode = parseEnum(argv[++i], cullModes);
                useCulling              = (mode != CullMode::Off);
                useHardCull             = (mode == CullMode::Hard || mode == CullMode::Invisible);
                hardCullForcesInvisible = (mode == CullMode::Invisible);
            });
        }
        else if (arg == "--hiz-occlusion" || arg == "--hizocclusion")
        {
            // HiZ occlusion test inside the cull: --hiz-occlusion <on|off>.
            // On by default; the frustum test keeps running when off.
            static const std::map<std::string, bool> onOff{
                { "on", true }, { "off", false },
            };
            parseArgValues(1, [&]() { useHizOcclusion = parseEnum(argv[++i], onOff); });
        }
        else if (arg == "--no-hiz-occlusion" || arg == "--no-hizocclusion")
        {
            useHizOcclusion = false;
        }
        else if (arg == "--show-occlusion-depth" || arg == "--show-hiz")
        {
            showOcclusionDepth = true;
        }
        else if (arg == "--debug-clusterlod" || arg == "--debug-allocator")
        {
            debugClusterLod = true;
        }
        else if (arg == "--compress" || arg == "--compressed")
        {
            compressClusterData = true;
        }
        else if (arg == "-v" || arg == "--verbose")
        {
            verboseLogging = true;
            rtxmg::g_verboseLogging = true;
        }
#if RTXMG_DEV_FEATURES
        else if (arg == "--clustersize")
        {
            // Bake cluster granularity (tris verts), e.g. "--clustersize 128 128".
            parseArgValues(2, [&]() {
                clusterTriangles = uint32_t(std::stoi(argv[++i]));
                clusterVertices  = uint32_t(std::stoi(argv[++i]));
            });
        }
        else if (arg == "--nostrip")
        {
            // Upload resident group blobs verbatim (no position/normal
            // stripping) — isolates the strip rewrite from bake/shader bugs.
            stripResidentData = false;
        }
        else if (arg == "--nostrippos")
        {
            // Keep positions resident but let normals still strip — exercises
            // the channels' independence (and mirrors a device with no
            // position-fetch intrinsic).
            stripResidentPositions = false;
        }
        else if (arg == "--no-uvquant")
        {
            // Bake raw float2 texcoords (disable the po2-grid UV quantizer) —
            // A/B diagnostic.
            quantizeTexCoords = false;
        }
#endif
        else if (arg == "--simplifyweights")
        {
            // Bake simplifier attribute weights (normal texcoord); BakerConfig
            // defaults to 0.5 0.5.
            parseArgValues(2, [&]() {
                simplifyNormalWeight   = float(std::stod(argv[++i]));
                simplifyTexCoordWeight = float(std::stod(argv[++i]));
            });
        }
        else if (arg == "--loderrormerge")
        {
            // LOD error propagation (previous additive); BakerConfig defaults to
            // 1.0 0.0, vk_lod_clusters ships 1.5 0.0.  Changes BakerConfig =>
            // full rebake, so pair with --cache-dir.
            parseArgValues(2, [&]() {
                lodErrorMergePrevious = float(std::stod(argv[++i]));
                lodErrorMergeAdditive = float(std::stod(argv[++i]));
            });
        }
        else if (arg == "--cache-dir" || arg == "--nvsngeocache")
        {
            parseArgValues(1, [&]() { clusterCacheDir = argv[++i]; });
        }
        else if (arg == "--bake-workers")
        {
            parseArgValues(1, [&]() { bakeWorkers = uint32_t(std::max(0, atoi(argv[++i]))); });
        }
#if RTXMG_DEV_FEATURES
        else if (arg == "--lpe-sweep")
        {
            lpeSweep = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                lpeSweepFrameInterval = uint32_t(std::max(1, atoi(argv[++i])));
        }
        else if (arg == "--dolly")
        {
            dolly = true;
        }
        else if (arg == "--dolly-frames")
        {
            dolly = true;
            parseArgValues(1, [&]() { dollyFrames = uint32_t(std::max(1, atoi(argv[++i]))); });
        }
        else if (arg == "--anim-step")
        {
            parseArgValues(1, [&]() { animStep = std::max(0.f, float(atof(argv[++i]))); });
        }
        else if (arg == "--anim-dump")
        {
            parseArgValues(1, [&]() { animDumpDir = argv[++i]; });
        }
        else if (arg == "--blas-toggle-sweep" || arg == "--blastogglesweep")
        {
            blasToggleSweep = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                blasToggleSweepFrameInterval = uint32_t(std::max(1, atoi(argv[++i])));
        }
        else if (arg == "--blas-toggle-settle")
        {
            parseArgValues(1, [&]() { blasToggleSweepSettleFrames = uint32_t(std::max(0, atoi(argv[++i]))); });
        }
        else if (arg == "--blas-toggle-mode")
        {
            static const std::map<std::string, BlasToggleMode> toggleModes{
                { "both",    BlasToggleMode::Both },
                { "sharing", BlasToggleMode::Sharing },
                { "caching", BlasToggleMode::Caching },
            };
            parseArgValues(1, [&]() { blasToggleMode = parseEnum(argv[++i], toggleModes); });
        }
        else if (arg == "--tess-budget-sweep")
        {
            tessBudgetSweep = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                tessBudgetSweepFrameInterval = uint32_t(std::max(1, atoi(argv[++i])));
        }
        else if (arg == "--tess-budget-settle")
        {
            parseArgValues(1, [&]() { tessBudgetSweepSettleFrames = uint32_t(std::max(0, atoi(argv[++i]))); });
        }
        else if (arg == "--vram-mb")
        {
            parseArgValues(1, [&]() { vramOverrideMB = std::max(0, atoi(argv[++i])); });
        }
        else if (arg == "--normalmap-sweep")
        {
            normalMapSweep = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                normalMapSweepSettleFrames = uint32_t(std::max(1, atoi(argv[++i])));
        }
#endif
        else if (arg == "--test-rebake")
        {
            testRebake = true;
        }
        else if (arg == "--camera-speed")
        {
            parseArgValues(1, [&]() { cameraSpeed = std::max(0.f, float(atof(argv[++i]))); });
        }
        else if (arg == "--enableDenoiser" || arg == "-ed")
        {
            enableDenoiser = true;
        }
        else if (arg == "--windowMaximized" || arg == "-wm")
        {
            startMaximized = true;
        }
        else if (arg == "--startFullscreen" || arg == "-fs")
        {
            startFullscreen = true;
        }
        else if (arg == "--vertexNormals" || arg == "-vn")
        {
            parseArgValues(1, [&]() { enableVertexNormals = parseBooleanArg(arg.c_str(), argv[++i], enableVertexNormals); });
        }
        else if (arg == "--clusterLodVertexNormals" || arg == "-cvn")
        {
            parseArgValues(1, [&]() { enableClusterLodVertexNormals = parseBooleanArg(arg.c_str(), argv[++i], enableClusterLodVertexNormals); });
        }
        else if (arg == "--normalmaps")
        {
            enableNormalMaps = true;
        }
        else if (arg == "--no-normalmaps")
        {
            enableNormalMaps = false;
        }
        else if (arg == "--normalmapshading")
        {
            parseArgValues(1, [&]() { normalMapShading = parseBooleanArg(arg.c_str(), argv[++i], normalMapShading); });
        }
        else if (arg == "--vsync")
        {
            parseArgValues(1, [&]() { vsync = parseBooleanArg(arg.c_str(), argv[++i], vsync); });
        }
        else
            printUsageAndExit(argv[0], std::string("Unknown option: ") + argv[i]);
    }
}


auto parseJsonEnum = []<typename T, size_t N>(const Json::Value & node,
    const std::array<const char*, N> &enums, T & result) constexpr
{
    uint8_t index = 0;
    for (const char* e : enums)
    {
        if (std::strncmp(node.asString().c_str(), e, std::strlen(e)) == 0)
        {
            result = T(index);
            break;
        }
        ++index;
    }
};

Args& operator << (Args& args, const Json::Value& node)
{
    if (const auto& value = node["envmap rotation"]; value.isDouble())
    {
        value >> args.envmapAzimuth;
        args.envmapAzimuth = (args.envmapAzimuth / 180.f) * float(std::numbers::pi);
    }

    if (const auto& value = node["envmap elevation"]; value.isDouble())
    {
        value >> args.envmapElevation;
        args.envmapElevation = (args.envmapElevation / 180.f) * float(std::numbers::pi);
    }
    if (const auto& value = node["envmap intensity"]; value.isDouble())
    {
        value >> args.envmapIntensity;
    }
    if (const auto& value = node["shading mode"]; value.isString())
        parseJsonEnum(value, kShadingModeNames, args.shadingMode);

    if (const auto& value = node["color mode"]; value.isString())
    {
        parseJsonEnum(value, kColorModeNames, args.colorMode);
    }

    if (const auto& value = node["spp"]; value.isIntegral())
        value >> args.spp;
    if (const auto& value = node["max bounces"]; value.isIntegral())
        value >> args.ptMaxBounces;

    if (const auto& value = node["firefly max intensity"]; value.isDouble())
        value >> args.firefliesClamp;

    if (const auto& value = node["exposure"]; value.isDouble())
        value >> args.exposure;

    if (const auto& value = node["auto exposure"]; value.isBool())
        value >> args.autoExposure;

    if (const auto& value = node["tonemap operator"]; value.isString())
        parseJsonEnum(value, kToneMapOperatorNames, args.tonemapOperator);

    // displacement
    if (const auto& value = node["displacementScale"]; value.isDouble())
        value >> args.dispScale;

    if (const auto& value = node["displacementBias"]; value.isDouble())
    {
        value >> args.dispBias;
        donut::log::warning("Scene arg displacementBias is not supported, ignoring");
    }

    if (const auto& value = node["wireframe"]; value.isBool())
        value >> args.enableWireframe;

    if (const auto& value = node["wireframe thickness"]; value.isDouble())
        value >> args.wireframeThickness;

    if (const auto& value = node["enableDenoiser"]; value.isBool())
        value >> args.enableDenoiser;

    return args;
}

void RestoreSceneOverridableFromCli(Args& args, const Args& cli)
{
    // Must list exactly the fields operator<< above can write; a scene setting
    // with no entry here would outrank the command line.
    args.envmapAzimuth      = cli.envmapAzimuth;
    args.envmapElevation    = cli.envmapElevation;
    args.envmapIntensity    = cli.envmapIntensity;
    args.shadingMode        = cli.shadingMode;
    args.colorMode          = cli.colorMode;
    args.spp                = cli.spp;
    args.ptMaxBounces       = cli.ptMaxBounces;
    args.firefliesClamp     = cli.firefliesClamp;
    args.exposure           = cli.exposure;
    args.autoExposure       = cli.autoExposure;
    args.tonemapOperator    = cli.tonemapOperator;
    args.dispScale          = cli.dispScale;
    args.dispBias           = cli.dispBias;
    args.enableWireframe    = cli.enableWireframe;
    args.wireframeThickness = cli.wireframeThickness;
    args.enableDenoiser     = cli.enableDenoiser;
}
