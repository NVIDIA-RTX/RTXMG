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

#include "rtxmg/profiler/stats_dump.h"

#include "rtxmg/cluster_lod/pass.h"  // kMaxTraversalInfos (the traversal-queue cap)
#include "rtxmg/profiler/profiler.h"
#include "rtxmg/profiler/statistics.h"

#include <donut/core/log.h>

#include <json/json.h>

#include <fstream>

// clang-format on

namespace fs = std::filesystem;

namespace stats
{

FrameLog                frameLog;
std::vector<ShotRecord> shotRecords;

namespace
{

// A sampler with no samples reports count 0 and nothing else: its min/max are
// still the sentinels they were initialized to.
template <typename SamplerType>
Json::Value SamplerToJson(const SamplerType& s)
{
    Json::Value v(Json::objectValue);
    v["count"] = Json::UInt64(s.samples_count);
    if (s.samples_count == 0)
        return v;

    v["mean"]   = double(s.Average());
    v["min"]    = double(s.min);
    v["max"]    = double(s.max);
    v["median"] = double(s.Median());
    v["p95"]    = double(s.Percentile(0.95));
    return v;
}

Json::Value StreamingStatsToJson(const rtxmg::StreamingStats& s)
{
    Json::Value v(Json::objectValue);

    v["residentGroups"]    = s.residentGroups;
    v["residentClusters"]  = s.residentClusters;
    v["residentTriangles"] = s.residentTriangles;
    v["maxGroups"]         = s.maxGroups;
    v["maxClusters"]       = s.maxClusters;
    v["residentNormals"]   = s.residentNormals;

    v["persistentGroups"]    = s.persistentGroups;
    v["persistentClusters"]  = s.persistentClusters;
    v["persistentTriangles"] = s.persistentTriangles;
    v["persistentDataBytes"] = Json::UInt64(s.persistentDataBytes);
    v["persistentClasBytes"] = Json::UInt64(s.persistentClasBytes);

    v["maxDataBytes"]       = Json::UInt64(s.maxDataBytes);
    v["reservedDataBytes"]  = Json::UInt64(s.reservedDataBytes);
    v["usedDataBytes"]      = Json::UInt64(s.usedDataBytes);
    v["allocatedDataBytes"] = Json::UInt64(s.allocatedDataBytes);

    v["reservedClasBytes"] = Json::UInt64(s.reservedClasBytes);
    v["usedClasBytes"]     = Json::UInt64(s.usedClasBytes);
    v["wastedClasBytes"]   = Json::UInt64(s.wastedClasBytes);
    v["maxSizedLeft"]      = s.maxSizedLeft;
    v["maxSizedReserved"]  = s.maxSizedReserved;

    v["totalTransferBytes"] = Json::UInt64(s.totalTransferBytes);
    v["totalLoads"]         = Json::UInt64(s.totalLoads);
    v["totalUnloads"]       = Json::UInt64(s.totalUnloads);

    // Saturation counters: a non-zero here means the run hit a budget wall, which
    // usually invalidates a perf comparison rather than merely degrading it.
    v["couldNotAllocateGroup"] = s.couldNotAllocateGroup;
    v["couldNotAllocateClas"]  = s.couldNotAllocateClas;
    v["couldNotTransfer"]      = s.couldNotTransfer;
    v["couldNotStore"]         = s.couldNotStore;
    v["uncompletedLoadCount"]  = s.uncompletedLoadCount;

    v["cachedBlasCount"] = s.cachedBlasCount;
    v["cachedBlasBytes"] = Json::UInt64(s.cachedBlasBytes);

    Json::Value scene(Json::objectValue);
    scene["geometryCount"]        = s.geometryCount;
    scene["instanceCount"]        = s.instanceCount;
    scene["modelTriangles"]       = Json::UInt64(s.modelTriangles);
    scene["modelClusters"]        = Json::UInt64(s.modelClusters);
    scene["modelClustersAllLods"] = Json::UInt64(s.modelClustersAllLods);
    scene["modelGroups"]          = Json::UInt64(s.modelGroups);
    scene["sceneTriangles"]       = Json::UInt64(s.sceneTriangles);
    scene["sceneClusters"]        = Json::UInt64(s.sceneClusters);
    scene["sceneClustersAllLods"] = Json::UInt64(s.sceneClustersAllLods);
    scene["sceneGroups"]          = Json::UInt64(s.sceneGroups);
    v["scene"] = scene;

    return v;
}

Json::Value CountersToJson(const shaderio::SceneBuildingCounters& c)
{
    Json::Value v(Json::objectValue);
    v["numRenderedClusters"]        = c.numRenderedClusters;
    v["desiredRenderClusters"]      = c.desiredRenderClusters;
    v["effectiveMaxRenderClusters"] = c.effectiveMaxRenderClusters;
    // Pre-clamp traversal-queue demand vs the (fixed) queue capacity: the only
    // way a run can assert that nothing was dropped by the queue cap.
    v["desiredTraversalNodes"]      = c.desiredTraversalNodes;
    v["desiredTraversalGroups"]     = c.desiredTraversalGroups;
    v["maxTraversalInfos"]          = ClusterLodPass::kMaxTraversalInfos;
    v["uniqueClusters"]             = c.uniqueClusters;
    v["totalClusters"]              = c.totalClusters;
    v["uniqueTriangles"]            = Json::UInt64(c.uniqueTriangles);
    v["totalTriangles"]             = Json::UInt64(c.totalTriangles);
    v["blasBuildCounter"]           = c.blasBuildCounter;
    v["numSharingProviders"]        = c.numSharingProviders;
    v["numSharingConsumers"]        = c.numSharingConsumers;
    v["cachedBlasCopyCounter"]      = c.cachedBlasCopyCounter;
    v["cachedClusters"]             = c.cachedClusters;
    v["cachedUniqueClusters"]       = c.cachedUniqueClusters;
    v["cachedTriangles"]            = Json::UInt64(c.cachedTriangles);
    v["cachedUniqueTriangles"]      = Json::UInt64(c.cachedUniqueTriangles);
    v["numMergedBlas"]              = c.numMergedBlas;
    return v;
}

Json::Value BakeStatsToJson(const BakeStats& b)
{
    Json::Value v(Json::objectValue);
    v["compressed"]  = b.compressed;
    v["quantizedUv"] = b.quantizedUv;
    v["bakedBytes"]  = Json::UInt64(b.bakedBytes);
    v["deviceBytes"] = Json::UInt64(b.deviceBytes);
    v["posBytes"]    = Json::UInt64(b.posBytes);
    v["nrmBytes"]    = Json::UInt64(b.nrmBytes);
    v["uvBytes"]     = Json::UInt64(b.uvBytes);
    v["triangles"]   = Json::UInt64(b.triangles);
    v["groups"]      = b.groups;
    v["clusters"]    = b.clusters;
    v["geometries"]  = b.geometries;
    return v;
}

Json::Value TextureMemToJson(const TextureMemStats& t)
{
    Json::Value v(Json::objectValue);
    v["textureCount"]         = t.textureCount;
    v["budgetableCount"]      = t.budgetableCount;
    v["droppedCount"]         = t.droppedCount;
    v["loadedCount"]          = t.loadedCount;
    v["diskBytes"]            = Json::UInt64(t.diskBytes);
    v["budgetableFullBytes"]  = Json::UInt64(t.budgetableFullBytes);
    v["keptBytes"]            = Json::UInt64(t.keptBytes);
    v["budgetBytes"]          = Json::UInt64(t.budgetBytes);
    v["loadedBytes"]          = Json::UInt64(t.loadedBytes);
    v["loadedOtherBytes"]     = Json::UInt64(t.loadedOtherBytes);
    v["vramBytes"]            = Json::UInt64(t.vramBytes);
    v["fullBytes"]            = Json::UInt64(t.FullBytes());
    return v;
}

// Parallel arrays rather than an array of objects: a 5000-frame run is the
// normal case and the key names would otherwise dominate the file.
Json::Value FrameLogToJson(const std::vector<FrameRecord>& records)
{
    Json::Value v(Json::objectValue);
    if (records.empty())
        return v;

    auto column = [&records](uint32_t FrameRecord::*field) {
        Json::Value col(Json::arrayValue);
        for (const FrameRecord& r : records)
            col.append(r.*field);
        return col;
    };

    v["frame"]            = column(&FrameRecord::frame);
    v["renderedClusters"] = column(&FrameRecord::renderedClusters);
    v["desiredClusters"]  = column(&FrameRecord::desiredClusters);
    v["uniqueClusters"]   = column(&FrameRecord::uniqueClusters);
    v["totalClusters"]    = column(&FrameRecord::totalClusters);
    v["residentGroups"]   = column(&FrameRecord::residentGroups);
    v["residentClusters"] = column(&FrameRecord::residentClusters);
    v["blasBuilds"]       = column(&FrameRecord::blasBuilds);
    return v;
}

}  // anonymous namespace

void ResetTimingSamplers()
{
    const Profiler& profiler = Profiler::Get();
    for (const auto& t : profiler.GetCPUTimers())
        t->Reset();
    for (const auto& t : profiler.GetGPUTimers())
        t->Reset();
    frameSamplers.cpuFrameTime.Reset();
    frameSamplers.uiBuildTime.Reset();
    frameSamplers.accelBuildTime.Reset();
}

bool DumpToJson(const fs::path& path, const RunIdentity& run)
{
    Json::Value root(Json::objectValue);

    // Bump when a key changes meaning, so the harness rejects a stale baseline
    // instead of silently comparing incompatible fields.
    root["schemaVersion"] = 1;

    Json::Value identity(Json::objectValue);
    identity["commandLine"]    = run.commandLine;
    identity["buildConfig"]    = run.buildConfig;
    identity["graphicsApi"]    = run.graphicsApi;
    identity["gpuName"]        = run.gpuName;
    identity["sceneFile"]      = run.sceneFile;
    identity["renderedFrames"] = run.renderedFrames;
    identity["renderWidth"]    = run.renderWidth;
    identity["renderHeight"]   = run.renderHeight;
    root["run"] = identity;

    Json::Value timers(Json::objectValue);
    const Profiler& profiler = Profiler::Get();
    for (const auto& t : profiler.GetCPUTimers())
        timers[t->name] = SamplerToJson(*t);
    for (const auto& t : profiler.GetGPUTimers())
        timers[t->name] = SamplerToJson(*t);
    root["timers"] = timers;

    Json::Value frame(Json::objectValue);
    frame[frameSamplers.cpuFrameTime.name]   = SamplerToJson(frameSamplers.cpuFrameTime);
    frame[frameSamplers.uiBuildTime.name]    = SamplerToJson(frameSamplers.uiBuildTime);
    frame[frameSamplers.accelBuildTime.name] = SamplerToJson(frameSamplers.accelBuildTime);
    root["frame"] = frame;

    const ClusterAccelSamplers& accel = clusterAccelSamplers;
    Json::Value phases(Json::objectValue);
    phases[accel.clusterLodTraversal.name]  = SamplerToJson(accel.clusterLodTraversal);
    phases[accel.clusterLodAllocation.name] = SamplerToJson(accel.clusterLodAllocation);
    phases[accel.clusterLodClasBuild.name]  = SamplerToJson(accel.clusterLodClasBuild);
    phases[accel.clusterLodBlasBuild.name]  = SamplerToJson(accel.clusterLodBlasBuild);
    phases[accel.clusterLodUpload.name]     = SamplerToJson(accel.clusterLodUpload);
    root["clusterLodPhases"] = phases;

    Json::Value geometry(Json::objectValue);
    geometry["hasClusterTess"] = accel.hasClusterTess;
    geometry["hasClusterLod"]  = accel.hasClusterLod;
    geometry[accel.numClusters.name]         = SamplerToJson(accel.numClusters);
    geometry[accel.numTriangles.name]        = SamplerToJson(accel.numTriangles);
    geometry[accel.clusterLodUniqueTriangles.name] = SamplerToJson(accel.clusterLodUniqueTriangles);
    geometry[accel.clusterLodTotalTriangles.name]  = SamplerToJson(accel.clusterLodTotalTriangles);
    geometry[accel.clusterLodUniqueClusters.name]  = SamplerToJson(accel.clusterLodUniqueClusters);
    geometry[accel.clusterLodTotalClusters.name]   = SamplerToJson(accel.clusterLodTotalClusters);
    root["geometry"] = geometry;

    root["bake"] = BakeStatsToJson(bakeStats);

    const MemUsageSamplers& mem = memUsageSamplers;
    Json::Value memory(Json::objectValue);
    memory[mem.blasSize.name]                = SamplerToJson(mem.blasSize);
    memory[mem.blasScratchSize.name]         = SamplerToJson(mem.blasScratchSize);
    memory[mem.clasSize.name]                = SamplerToJson(mem.clasSize);
    memory[mem.vertexBufferSize.name]        = SamplerToJson(mem.vertexBufferSize);
    memory[mem.vertexNormalsBufferSize.name] = SamplerToJson(mem.vertexNormalsBufferSize);
    memory[mem.clusterShadingDataSize.name]  = SamplerToJson(mem.clusterShadingDataSize);
    memory["textures"]                       = TextureMemToJson(mem.textures);

    // The VRAM Budget window's stack, so a perf baseline records device memory as
    // a whole rather than one pool at a time.
    const VramBreakdown& vram = vramBreakdown;
    Json::Value vramJson(Json::objectValue);
    vramJson["textures"]        = Json::UInt64(vram.textures);
    vramJson["envmap"]          = Json::UInt64(vram.envmap);
    vramJson["clodGeometry"]    = Json::UInt64(vram.clodGeometry);
    vramJson["clodClas"]        = Json::UInt64(vram.clodClas);
    vramJson["clodCachedBlas"]  = Json::UInt64(vram.clodCachedBlas);
    vramJson["clodMetadata"]    = Json::UInt64(vram.clodMetadata);
    vramJson["tessVertices"]    = Json::UInt64(vram.tessVertices);
    vramJson["tessClas"]        = Json::UInt64(vram.tessClas);
    vramJson["tessClusterData"] = Json::UInt64(vram.tessClusterData);
    vramJson["tessSurfaces"]    = Json::UInt64(vram.tessSurfaces);
    vramJson["blas"]            = Json::UInt64(vram.blas);
    vramJson["blasUsed"]        = Json::UInt64(vram.blasUsed);
    vramJson["asScratch"]       = Json::UInt64(vram.asScratch);
    vramJson["dlss"]            = Json::UInt64(vram.dlss);
    vramJson["renderTargets"]   = Json::UInt64(vram.renderTargets);
    vramJson["accounted"]       = Json::UInt64(vram.Accounted());
    vramJson["driverValid"]     = vram.driverValid;
    vramJson["driverUsage"]     = Json::UInt64(vram.driverUsage);
    vramJson["driverBudget"]    = Json::UInt64(vram.driverBudget);
    vramJson["unaccounted"]     = Json::UInt64(vram.Unaccounted());
    memory["vram"] = vramJson;

    root["memory"] = memory;

    const StreamingSamplers& str = streamingSamplers;
    Json::Value streaming(Json::objectValue);
    streaming["latest"]   = StreamingStatsToJson(str.latest);
    streaming["counters"] = CountersToJson(str.latestCounters);
    streaming[str.residentGroups.name]   = SamplerToJson(str.residentGroups);
    streaming[str.residentClusters.name] = SamplerToJson(str.residentClusters);
    streaming[str.geometryMB.name]       = SamplerToJson(str.geometryMB);
    streaming[str.clasMB.name]           = SamplerToJson(str.clasMB);
    streaming[str.blasBuilds.name]       = SamplerToJson(str.blasBuilds);
    streaming["blasActualBytes"]         = Json::UInt64(str.latestBlasActualBytes);
    streaming["maxStreamClasBuildMs"]    = str.maxStreamClasBuildMs;
    streaming["maxStreamTransferBytes"]  = Json::UInt64(str.maxStreamTransferBytes);
    streaming["streamFrameCount"]        = Json::UInt64(str.streamFrameCount);
    root["streaming"] = streaming;

    root["frameLog"] = FrameLogToJson(frameLog.records);

    Json::Value shots(Json::arrayValue);
    for (const ShotRecord& r : shotRecords)
    {
        Json::Value s(Json::objectValue);
        s["label"]            = r.label;
        s["mode"]             = r.mode;
        s["file"]             = r.file;
        s["settleFrames"]     = r.settleFrames;
        s["accumFrames"]      = r.accumFrames;
        s["subframeIndex"]    = r.subframeIndex;
        s["exposure"]         = r.exposure;
        s["residentGroups"]   = r.residentGroups;
        s["residentClusters"] = r.residentClusters;
        s["uniqueClusters"]   = r.uniqueClusters;
        s["totalTriangles"]   = Json::UInt64(r.totalTriangles);

        s["dlssMode"]              = r.dlssMode;
        s["renderWidth"]           = r.renderWidth;
        s["renderHeight"]          = r.renderHeight;
        s["outputWidth"]           = r.outputWidth;
        s["outputHeight"]          = r.outputHeight;
        s["lodPixelError"]         = r.lodPixelError;
        s["adaptiveLodPixelError"] = r.adaptiveLodPixelError;
        s["normalMapShading"]      = r.normalMapShading;

        s["renderedClusters"]    = r.renderedClusters;
        s["geometryBytes"]       = Json::UInt64(r.geometryBytes);
        s["geometryBudgetBytes"] = Json::UInt64(r.geometryBudgetBytes);
        s["geometryPercent"]     = r.geometryPercent;
        s["clasBytes"]           = Json::UInt64(r.clasBytes);
        s["clasBudgetBytes"]     = Json::UInt64(r.clasBudgetBytes);
        s["clasPercent"]         = r.clasPercent;
        s["processVramBytes"]    = Json::UInt64(r.processVramBytes);
        s["vramBudgetBytes"]     = Json::UInt64(r.vramBudgetBytes);
        shots.append(s);
    }
    root["shots"] = shots;

    std::ofstream ofs(path);
    if (!ofs)
    {
        donut::log::warning("--dump-stats: cannot open '%s' for writing",
                            path.generic_string().c_str());
        return false;
    }

    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    ofs << Json::writeString(builder, root);
    if (!ofs)
    {
        donut::log::warning("--dump-stats: failed writing '%s'",
                            path.generic_string().c_str());
        return false;
    }

    donut::log::info("Stats written: %s (%zu frame records)",
                     path.generic_string().c_str(), frameLog.records.size());
    return true;
}

}  // end namespace stats
