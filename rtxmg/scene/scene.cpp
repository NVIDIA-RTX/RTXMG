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

#include <donut/core/log.h>
#include <donut/engine/TextureCache.h>
#include <json/json.h>
#include <nvrhi/common/misc.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <numbers>
#include <thread>

#include "rtxmg/cluster_lod/cluster_lod_gltf_importer.h"
#include "rtxmg/cluster_lod/baking/bake_progress.h"
#include "rtxmg/scene/box_extent.h"
#include "rtxmg/scene/instance_data.h"
#include "rtxmg/scene/instance_grid.h"
#include "rtxmg/scene/material.h"
#include "rtxmg/scene/scene.h"
#include "rtxmg/scene/json.h"
#include "rtxmg/subdivision/shape.h"
#include "rtxmg/subdivision/subdivision_surface.h"
#include "rtxmg/subdivision/topology_cache.h"
#include "rtxmg/utils/parallel_for.h"
#include "rtxmg/utils/vram_bytes.h"

using namespace donut;

namespace fs = std::filesystem;

RTXMGScene::RTXMGScene(const RTXMGSceneParams& p)
    : m_device(p.device),
    m_fs(p.fs),
    m_textureCache(p.textureCache),
    m_descriptorTable(p.descriptorTable),
    m_textureLoader(p.textureCache, *p.mediaPath),
    m_materialLibrary(p.device),
    m_mediaPath(*p.mediaPath),
    m_isoLevelSharp(p.isoLevelSharp),
    m_isoLevelSmooth(p.isoLevelSmooth),
    m_commonPasses(p.commonPasses),
    m_logClusterLod(p.logClusterLod),
    m_clusterBakerConfig(p.clusterBakerConfig),
    m_clusterCacheDir(p.clusterCacheDir),
    m_clusterBakeWorkers(p.clusterBakeWorkers),
    m_cancelLoad(p.cancelLoad)
{
    m_attributes.frameRange = p.initialFrameRange;
}

RTXMGScene::~RTXMGScene() = default;

void RTXMGScene::InsertModel(Model&& model)
{
    if (model.subd)
    {
        m_attributes.frameRange.x =
            std::min(m_attributes.frameRange.x, model.frameRange.x);
        m_attributes.frameRange.y =
            std::max(m_attributes.frameRange.y, model.frameRange.y);

        for (auto& instance : model.instances)
            instance.meshID = uint32_t(m_subdMeshes.size());

        m_subdMeshes.emplace_back(std::move(model.subd));

        std::ranges::move(model.instances, std::back_inserter(m_subdInstances));
    }
}

void RTXMGScene::InsertClusterLodModel(ClusterLodModel&& model)
{
    // Assign geometry IDs relative to current count before appending.
    const uint32_t baseGeoID = (uint32_t)m_clusterLodGeometries.size();
    for (auto& inst : model.instances)
        inst.geometryID += baseGeoID;

    // Material IDs in instances and in GeometryStorage::localMaterialIDs[] are
    // GLTF-file-local; offset them so they index the combined
    // m_clusterLodMaterialDescs.
    const uint32_t baseMtlID = (uint32_t)m_clusterLodMaterialDescs.size();
    for (auto& inst : model.instances)
        inst.materialID += baseMtlID;
    for (auto& storage : model.storages)
        for (auto& localID : storage.localMaterialIDs)
            if (localID != ~0u)
                localID += baseMtlID;

    std::ranges::move(model.geometries, std::back_inserter(m_clusterLodGeometries));
    std::ranges::move(model.storages,   std::back_inserter(m_clusterLodStorages));
    std::ranges::move(model.instances,  std::back_inserter(m_clusterLodInstances));
    std::ranges::move(model.materials,  std::back_inserter(m_clusterLodMaterialDescs));

    // First mmap wins; a later model keeping its own view is harmless because its
    // spans already point into the correct storage vectors.
    if (!m_cacheFileView.IsValid() && model.cacheView.IsValid())
        m_cacheFileView = std::move(model.cacheView);

    // The shard mappings must outlive the scene: m_clusterLodGeometries holds
    // zero-copy GeometryViews pointing into them.
    if (!model.shardCache.IsEmpty())
        m_shardCaches.push_back(std::move(model.shardCache));

    m_hasClusterLod = true;
}

uint32_t RTXMGScene::TotalSubdPatchCount() const
{
    const auto& instances = GetSubdMeshInstances();
    const auto& subds = GetSubdMeshes();

    uint32_t sum{ 0 };
    for (auto i = instances.begin(); i != instances.end(); ++i)
        sum += subds[i->meshID]->SurfaceCount();
    return sum;
}

uint64_t RTXMGScene::GetSubdVramBytes() const
{
    uint64_t bytes = 0;
    for (const auto& mesh : m_subdMeshes)
        bytes += mesh->GetVramBytes();
    for (const SubdivisionSurface::Pose& pose : m_subdPoses)
        bytes += rtxmg::BufferVramBytes(pose.patchPoints);
    for (const auto& map : m_topologyMaps)
        bytes += map->GetVramBytes();
    return bytes;
}

void RTXMGScene::Animate(float animTime, float animRate)
{
    for (auto& pose : m_subdPoses)
        pose.Animate(*m_subdMeshes[pose.meshID], animTime, animRate);
}

void RTXMGScene::BuildSubdPoses(nvrhi::ICommandList* commandList)
{
    m_subdPoses.clear();
    for (uint32_t meshID = 0; meshID < m_subdMeshes.size(); ++meshID)
    {
        SubdivisionSurface& mesh = *m_subdMeshes[meshID];
        if (!mesh.HasAnimation())
            continue;

        std::vector<float> offsets;
        for (const Instance& instance : m_subdInstances)
            if (instance.meshID == meshID && std::ranges::find(offsets, instance.frameOffset) == offsets.end())
                offsets.push_back(instance.frameOffset);
        if (offsets.empty())
            continue;

        const uint32_t firstPose = uint32_t(m_subdPoses.size());
        std::ranges::move(mesh.CreatePoses(meshID, offsets, *m_descriptorTable, commandList),
                          std::back_inserter(m_subdPoses));
        for (Instance& instance : m_subdInstances)
            if (instance.meshID == meshID)
                instance.poseID = firstPose + uint32_t(std::ranges::find(offsets, instance.frameOffset) - offsets.begin());
    }
    if (!m_subdPoses.empty())
        log::info("RTXMGScene: %zu animated subd poses for %zu instances", m_subdPoses.size(),
                  m_subdInstances.size());
}

const SubdivisionSurface::Pose* RTXMGScene::GetSubdPose(const Instance& instance) const
{
    return instance.poseID < m_subdPoses.size() ? &m_subdPoses[instance.poseID] : nullptr;
}

nvrhi::BindingSetItem RTXMGScene::GetSubdPositions(uint32_t slot, const Instance& instance) const
{
    const SubdivisionSurface& mesh = *m_subdMeshes[instance.meshID];
    if (const SubdivisionSurface::Pose* pose = GetSubdPose(instance))
        return nvrhi::BindingSetItem::StructuredBuffer_SRV(slot, mesh.m_posePositions, nvrhi::Format::UNKNOWN,
                                                          pose->range);
    return nvrhi::BindingSetItem::StructuredBuffer_SRV(slot, mesh.m_positionsBuffer);
}

nvrhi::IBuffer* RTXMGScene::GetSubdPatchPoints(const Instance& instance) const
{
    const SubdivisionSurface::Pose* pose = GetSubdPose(instance);
    return pose ? pose->patchPoints.Get() : m_subdMeshes[instance.meshID]->m_vertexDeviceData.patchPoints.Get();
}

box3 RTXMGScene::GetSubdAabb(const Instance& instance) const
{
    const SubdivisionSurface::Pose* pose = GetSubdPose(instance);
    return pose ? pose->aabb : m_subdMeshes[instance.meshID]->m_aabb;
}

static Instance& operator << (Instance& instance, const Json::Value& node)
{
    if (const auto& value = node["translation"]; !value.isNull())
        value >> instance.translation;

    if (const auto& value = node["rotation"]; !value.isNull())
    {
        if (node.isArray() && node.size() == 4)
            throw std::runtime_error("expecting 4-component quaternion for node's 'rotation' (use 'euler' otherwise)");
        value >> instance.rotation;
    }
    else if (const auto& value = node["euler"]; !value.isNull())
    {
        float3 euler = { 0.0, 0.0, 0.0 };
        value >> euler;
        euler *= float(std::numbers::pi) / 180.f;
        instance.rotation = donut::math::rotationQuat<float>(euler);
    }

    if (const auto& value = node["scaling"]; !value.isNull())
        value >> instance.scaling;

    if (const auto& value = node["frame offset"]; value.isNumeric())
        instance.frameOffset = value.asFloat();

    instance.UpdateLocalTransform();

    return instance;
}


//
// View
//

static View& operator << (View& view, const Json::Value& node)
{
    if (const auto& value = node["position"]; !value.isNull())
        value >> view.position;
    if (const auto& value = node["lookat"]; !value.isNull())
        value >> view.lookat;
    if (const auto& value = node["up"]; !value.isNull())
        value >> view.up;
    if (const auto& value = node["fov"]; !value.isNull())
        value >> view.fov;

    return view;
}


static RTXMGScene::Attributes& operator << (RTXMGScene::Attributes& attrs, const Json::Value& node)
{
    if (const auto& value = node["audio"]; value.isString())
        value >> attrs.audio;

    if (const auto& value = node["audio start time"]; value.isDouble())
        value >> attrs.audioStartTime;

    if (const auto& value = node["frame range"]; value.isArray())
        value >> attrs.frameRange;

    if (const auto& value = node["frame rate"]; value.isDouble())
        value >> attrs.frameRate;

    return attrs;
}

void RTXMGScene::InsertClusterLodModelWithMetadata(ClusterLodModel&& model, nvrhi::ICommandList* commandList)
{
    const size_t numGeom = model.geometries.size();
    std::vector<ClusterLodPrebuiltGeometryMetadata> metadata(numGeom);
    if (numGeom > 0)
    {
        std::vector<const GeometryView*> geometries(numGeom);
        for (size_t i = 0; i < numGeom; ++i)
            geometries[i] = &model.geometries[i];
        const ClusterLodMetadataArena arena = ClusterLodMetadataArena::Create(geometries, m_device);

        // Leave two cores for the main and present threads.
        const uint32_t hw = std::max(1u, std::thread::hardware_concurrency());
        const uint32_t workerCount = m_loadThreadPool ? (hw > 4 ? hw - 2 : hw) : 1u;

        rtxmg::GetBakeProgress().Begin(uint32_t(numGeom), 0, workerCount, "Uploading cluster LOD geometry");
        struct EndGuard { ~EndGuard() { rtxmg::GetBakeProgress().End(); } } endGuard;
        const auto t0 = std::chrono::steady_clock::now();

        // One deferred list per worker: nvrhi allows a single open immediate list.
        std::vector<nvrhi::CommandListHandle> workerCLs(workerCount);
        auto buildOne = [&](size_t i, uint32_t w)
        {
            if (IsLoadCancelled())
                return;
            nvrhi::ICommandList* cl = commandList;
            if (m_loadThreadPool)
            {
                if (!workerCLs[w])
                {
                    workerCLs[w] = m_device->createCommandList(
                        nvrhi::CommandListParameters().setEnableImmediateExecution(false));
                    workerCLs[w]->open();
                }
                cl = workerCLs[w];
            }
            metadata[i] = ClusterLodPrebuiltGeometryMetadata::Build(
                model.geometries[i], arena, i, m_descriptorTable.get(), cl);
            rtxmg::GetBakeProgress().CompleteOne(w);
        };
        if (m_loadThreadPool)
            rtxmg::ParallelFor(*m_loadThreadPool, workerCount, numGeom, buildOne);
        else
            for (size_t i = 0; i < numGeom; ++i)
                buildOne(i, 0);

        for (auto& cl : workerCLs)
        {
            if (!cl)
                continue;
            cl->close();
            m_metadataCommandLists.push_back(std::move(cl));
        }

        if (m_logClusterLod)
            log::info("RTXMGScene: built cluster-LOD geometry metadata for %zu geometries "
                      "in %.0f ms (%u workers)", numGeom,
                      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                      workerCount);
    }

    InsertClusterLodModel(std::move(model));
    std::ranges::move(metadata, std::back_inserter(m_clusterLodPrebuiltMetadata));
    assert(m_clusterLodPrebuiltMetadata.size() == m_clusterLodGeometries.size());
}

bool RTXMGScene::InsertClusterLodGltf(const std::filesystem::path& gltfPath, const affine3& nodeTransform,
                                      nvrhi::ICommandList* commandList)
{
    const fs::path key = gltfPath.lexically_normal();
    auto inserted = std::ranges::find(m_insertedClusterModels, key, &InsertedClusterModel::path);
    const size_t firstInstance = m_clusterLodInstances.size();
    if (inserted != m_insertedClusterModels.end())
    {
        m_clusterLodInstances.insert(m_clusterLodInstances.end(),
                                     inserted->instances.begin(), inserted->instances.end());
    }
    else
    {
        ClusterLodGltfImporter gltfImporter(m_clusterBakerConfig, m_logClusterLod, m_clusterCacheDir,
                                            m_clusterBakeWorkers);
        auto clusterModel = gltfImporter.Load(gltfPath);
        if (!clusterModel.has_value())
            return false;
        InsertClusterLodModelWithMetadata(std::move(*clusterModel), commandList);
        m_insertedClusterModels.push_back(
            { key, { m_clusterLodInstances.begin() + firstInstance, m_clusterLodInstances.end() } });
    }

    if (nodeTransform != affine3::identity())
    {
        for (size_t i = firstInstance; i < m_clusterLodInstances.size(); ++i)
        {
            ClusterLodInstance& inst = m_clusterLodInstances[i];
            inst.transform = affineToHomogeneous(homogeneousToAffine(inst.transform) * nodeTransform);
        }
    }
    return true;
}

void RTXMGScene::InsertSubdObj(const std::filesystem::path& objPath, const std::string& modelName,
                               const Instance& instance, ObjImporter& objImporter,
                               nvrhi::ICommandList* commandList)
{
    const fs::path key = objPath.lexically_normal();
    auto inserted = std::ranges::find(m_insertedSubdModels, key, &InsertedSubdModel::path);
    if (inserted != m_insertedSubdModels.end())
    {
        Instance copy = instance;
        copy.meshID = inserted->meshID;
        copy.aabb   = m_subdMeshes[copy.meshID]->m_aabb * copy.localToWorld;
        m_subdInstances.push_back(std::move(copy));
        return;
    }

    auto model = objImporter.Load(modelName, *m_textureCache, { 0, 0 }, instance, commandList);
    if (!model.has_value() || !model->subd)
        return;
    m_insertedSubdModels.push_back({ key, uint32_t(m_subdMeshes.size()) });
    InsertModel(std::move(*model));
}

void RTXMGScene::LoadSceneFile(const std::filesystem::path& m_filepath, std::unique_ptr<ObjImporter>& objImporter, nvrhi::ICommandList* commandList)
{
    fs::path fp = m_filepath;

    Json::Value jsonRoot;

    try
    {
        jsonRoot = readFile(fp);
    }
    catch (const std::exception& e)
    {
        log::fatal("failed to Parse JSON file '%s': %s", fp.generic_string().c_str(), e.what());
    }
    if (jsonRoot.isObject())
    {
        objImporter->SetModelPath(fp.parent_path());

        const Json::Value& models = jsonRoot["models"];
        const Json::Value& graph = jsonRoot["graph"];

        if (!models.isArray() || !graph.isArray())
            throw std::runtime_error("need valid 'models' and 'graph' arrays in '" + fp.generic_string() + "'");

        uint32_t nmodels = models.size();

        for (uint32_t i = 0; i < graph.size(); ++i)
        {
            const Json::Value& node = graph[i];

            Instance instance;

            instance << node;

            std::string nodeName;
            if (const auto& name = node["name"]; name.isString())
                nodeName = name.asString();

            if (const auto& modelNode = node["model"]; !modelNode.isNull())
            {
                if (!modelNode.isIntegral())
                    throw std::runtime_error("'model' value for graph node '" + nodeName + "' must be an index");

                int modelIndex = modelNode.asInt();
                if (modelIndex < 0 || modelIndex >= (int)nmodels)
                    throw std::runtime_error("out of bounds 'model' index for graph node '" + nodeName + "'");

                const Json::Value& modelName = models[modelIndex];

                if (!modelName.isString())
                    throw std::runtime_error("invalid model path in 'models' section");

                const fs::path modelPath = modelName.asString();
                const auto modelExt = modelPath.extension();

                if (modelExt == ".gltf" || modelExt == ".glb")
                {
                    const fs::path resolvedPath =
                        rtxmg::ResolveMediapath(fp.parent_path() / modelPath, m_mediaPath);
                    const fs::path loadPath =
                        resolvedPath.empty() ? fp.parent_path() / modelPath : resolvedPath;

                    if (!InsertClusterLodGltf(loadPath, instance.localToWorld, commandList))
                        log::warning("RTXMGScene::LoadSceneFile: Failed to load GLTF '%s'",
                                     modelPath.string().c_str());
                }
                else
                {
                    InsertSubdObj(fp.parent_path() / modelPath, modelName.asString(), instance,
                                  *objImporter, commandList);
                }
            }

            if (const auto& type = node["type"]; type.isString())
                throw std::runtime_error("'type' token for graph node '" + nodeName + "' not supported");

            if (const auto& parent = node["parent"]; !parent.isNull())
                throw std::runtime_error("'parent' token for graph node '" + nodeName + "' not supported");

            if (const auto& children = node["children"]; !children.isNull())
                throw std::runtime_error("'children' token for graph node '" + nodeName + "' not supported");
        }

        if (Json::Value& view = jsonRoot["view"]; view.isObject())
        {
            m_view = std::make_unique<View>();
            *m_view << view;
        }

        if (Json::Value& settings = jsonRoot["settings"]; settings.isObject())
        {
            m_sceneSettings = settings; // so the app can use these settings to override some of its own behavior
            m_attributes << settings;
        }
    }
    else
    {
        log::fatal("failed to Parse JSON file '%s'", fp.generic_string().c_str());
    }
}

bool RTXMGScene::LoadWithThreadPool(const std::filesystem::path& filename,
    ThreadPool* threadPool, std::vector<nvrhi::CommandListHandle>* outRecordedCommandLists)
{
    log::info("RTXMGScene::LoadWithThreadPool: %s", filename.string().c_str());

    TopologyCache topologyCache(TopologyCache::Options{
        .isoLevelSharp = (uint8_t)m_isoLevelSharp,
        .isoLevelSmooth = (uint8_t)m_isoLevelSmooth,
        });

    fs::path sanitizedFilePath = filename;
    std::string sceneName = sanitizedFilePath.empty() ? "default_scene" : sanitizedFilePath.filename().generic_string();

    // Deferred (non-immediate): this runs on the load thread, where an immediate
    // list would collide with the main thread's — only one may be open at a time.
    auto commandList = m_device->createCommandList(
        nvrhi::CommandListParameters().setEnableImmediateExecution(false));
    commandList->open();
    {
        // Reset the per-geometry GPU-metadata table before models insert into it.
        m_clusterLodPrebuiltMetadata.clear();
        m_insertedClusterModels.clear();
        m_insertedSubdModels.clear();
        m_metadataCommandLists.clear();
        m_loadThreadPool = threadPool;

        std::unique_ptr<ObjImporter> objImporter =
            std::make_unique<ObjImporter>(m_fs, m_mediaPath, m_descriptorTable, topologyCache);

        const bool imported = ImportModels(sanitizedFilePath, objImporter, commandList);
        m_loadThreadPool = nullptr;
        if (!imported || IsLoadCancelled())
        {
            commandList->close();
            m_metadataCommandLists.clear();
            return false;
        }

        ReduceSceneStatistics();

        BuildSubdPoses(commandList);
        Animate(0.f, m_attributes.frameRate);

        m_topologyMaps = topologyCache.InitDeviceData(m_descriptorTable, commandList);

        m_inputPath = sanitizedFilePath.lexically_normal().generic_string();
    }
    commandList->close();
    // Queue submission is the caller's, on the main thread: it races the
    // Streamline-wrapped present.
    auto submit = [&](const nvrhi::CommandListHandle& cl)
    {
        if (outRecordedCommandLists)
            outRecordedCommandLists->push_back(cl);
        else
            m_device->executeCommandList(cl);
    };
    for (const auto& cl : m_metadataCommandLists)
        submit(cl);
    m_metadataCommandLists.clear();
    submit(commandList);

    // Build RTXMGSceneGraph
    auto root = std::make_shared<RTXMGSceneNode>();
    root->name = sceneName;
    m_sceneGraph.SetRootNode(root);

    m_materialLibrary.BuildSubdMaterials(m_subdMeshes, m_textureLoader, threadPool);
    BuildSubdSceneNodes(root);

    // Cluster-LoD materials append after the subd ones, so this must stay second.
    m_materialLibrary.BuildClusterLodMaterials(m_clusterLodMaterialDescs, m_textureLoader, threadPool);
    BuildClusterLodSceneNodes(root);

    return true;
}

// File-format dispatch: .obj (or no filename) goes through ObjImporter, .gltf/.glb
// through the cluster-LoD importer, and .json through LoadSceneFile, which picks
// either per graph node.
bool RTXMGScene::ImportModels(const std::filesystem::path& filepath,
                              std::unique_ptr<ObjImporter>& objImporter,
                              nvrhi::ICommandList* commandList)
{
    const auto ext = filepath.extension();

    if (filepath.empty() || ext == ".obj")
    {
        // obj importer will default to a cube without a filename
        log::info("RTXMGScene::LoadWithThreadPool: Loading an OBJ file");

        auto model =
            objImporter->Load(filepath, *m_textureCache, m_attributes.frameRange,
                Instance{}, commandList);

        if (model.has_value())
        {
            InsertModel(std::move(*model));
        }
        else
        {
            log::fatal("RTXMGScene::LoadWithThreadPool: Failed to load the OBJ file");
        }
    }
    else if (ext == ".gltf" || ext == ".glb")
    {
        log::info("RTXMGScene::LoadWithThreadPool: Loading a GLTF file");

        if (!InsertClusterLodGltf(filepath, affine3::identity(), commandList))
            log::fatal("RTXMGScene::LoadWithThreadPool: Failed to load the GLTF file");
    }
    else if (ext == ".json")
    {
        log::info("RTXMGScene::LoadWithThreadPool: Loading a JSON file");
        LoadSceneFile(filepath, objImporter, commandList);
    }
    else
    {
        log::fatal("RTXMGScene::LoadWithThreadPool: Unsupported file format");
        return false;
    }
    return true;
}

void RTXMGScene::ReduceSceneStatistics()
{
    m_attributes.averageInstanceScale = 0.0f;
    // Per-instance world sizes feed the median, which drives the default camera
    // move speed — huge vista instances would skew the average.
    std::vector<float> instanceSizes;
    instanceSizes.reserve(m_subdInstances.size() + m_clusterLodInstances.size());
    for (const auto& instance : m_subdInstances)
    {
        const float size = maxComponent(instance.aabb.m_maxs - instance.aabb.m_mins);
        m_attributes.averageInstanceScale += size;
        instanceSizes.push_back(size);
    }
    // Cluster-LoD instances live in their own list, so fold them in separately —
    // otherwise a cluster-LoD scene gets scale 0 and dead WASD.
    for (const auto& inst : m_clusterLodInstances)
    {
        const GeometryView& geo = m_clusterLodGeometries[inst.geometryID];
        const box3 obj(float3(geo.bbox.lo.x, geo.bbox.lo.y, geo.bbox.lo.z),
                       float3(geo.bbox.hi.x, geo.bbox.hi.y, geo.bbox.hi.z));
        const affine3 instXform = homogeneousToAffine(inst.transform);
        // World space, to match the subd branch above; the object-space box
        // would report a size that ignores instance placement.
        const float size = maxComponent((obj * instXform).diagonal());
        m_attributes.averageInstanceScale += size;
        instanceSizes.push_back(size);
    }
    const size_t instanceScaleCount = instanceSizes.size();
    m_attributes.medianInstanceScale = 0.f;
    if (!instanceSizes.empty())
    {
        const size_t mid = instanceSizes.size() / 2;
        std::nth_element(instanceSizes.begin(), instanceSizes.begin() + mid,
                         instanceSizes.end());
        m_attributes.medianInstanceScale = instanceSizes[mid];
    }

    // Reduce the scene-wide max/total stats; initClas sizes its CLAS scratch
    // from these, needing the observed max after baking.
    for (const auto& geo : m_clusterLodGeometries)
    {
        m_maxClusterVertices  = std::max(m_maxClusterVertices,  geo.clusterMaxVerticesCount);
        m_maxClusterTriangles = std::max(m_maxClusterTriangles, geo.clusterMaxTrianglesCount);
        // Every geometry validated against m_clusterBakerConfig at load, so
        // requested == baked; check it rather than trusting the policy.
        assert(geo.lodStats.empty()
               || bool(geo.lodStats[0].compressed) == m_clusterBakerConfig.useCompressedData);
    }

    // cgltf_alpha_mode: 0=opaque, 1=mask, 2=blend.  BLEND counts as alpha-masked
    // here because it needs the same per-triangle CLAS geometry indices.
    m_hasAlphaMask = false;
    for (const auto& desc : m_clusterLodMaterialDescs)
    {
        if (desc.alphaModeGltf != 0)
            m_hasAlphaMask = true;
    }

    if (instanceScaleCount > 0)
        m_attributes.averageInstanceScale /= float(instanceScaleCount);
    if ((m_attributes.frameRange.y - m_attributes.frameRange.x) > 1 &&
        m_attributes.frameRate == 0.f)
    {
        m_attributes.frameRate = 24.0f;
    }
}

void RTXMGScene::BuildSubdSceneNodes(const std::shared_ptr<RTXMGSceneNode>& root)
{
    uint32_t instanceIndex = 0;
    for (auto& instance : m_subdInstances)
    {
        auto node = std::make_shared<RTXMGSceneNode>();
        node->name = m_subdMeshes[instance.meshID]->GetShape()->filepath.generic_string()
                     + "_" + std::to_string(instanceIndex);
        node->translation       = double3(instance.translation);
        node->rotation          = dquat(instance.rotation);
        node->scaling           = double3(instance.scaling);
        node->objectSpaceBounds = m_subdMeshes[instance.meshID]->GetShape()->aabb;
        node->subdMeshInstance  = std::make_shared<SubdivisionMeshInstance>(SubdivisionMeshInstance{ instance.meshID });

        instance.node = node;
        m_sceneGraph.Attach(root, node);
        ++instanceIndex;
    }
}

void RTXMGScene::BuildClusterLodSceneNodes(const std::shared_ptr<RTXMGSceneNode>& root)
{
    for (auto& inst : m_clusterLodInstances)
    {
        const GeometryView& geo = m_clusterLodGeometries[inst.geometryID];

        const affine3 aff = homogeneousToAffine(inst.transform);
        double3 translation, scaling;
        dquat   rotation;
        decomposeAffine(daffine3(aff), &translation, &rotation, &scaling);

        auto node = std::make_shared<RTXMGSceneNode>();
        node->name              = inst.name.empty() ? "cluster_lod_geo_" + std::to_string(inst.geometryID) : inst.name;
        node->translation       = translation;
        node->rotation          = rotation;
        node->scaling           = scaling;
        node->objectSpaceBounds = box3(
            float3(geo.bbox.lo.x, geo.bbox.lo.y, geo.bbox.lo.z),
            float3(geo.bbox.hi.x, geo.bbox.hi.y, geo.bbox.hi.z));
        node->clusterLodMeshInstance  = std::make_shared<ClusterLodMeshInstance>(ClusterLodMeshInstance{ inst.geometryID });
        m_sceneGraph.Attach(root, node);
    }
}

// ---------------------------------------------------------------------------
// Buffer building
// ---------------------------------------------------------------------------

// Subd instances only: the cluster-LoD hit shaders take their transforms from
// t_ClusterLodInstances, so t_InstanceData is indexed by subd instance.
void RTXMGScene::BuildInstanceBuffer(nvrhi::ICommandList* commandList)
{
    const auto& subdInstances = GetSubdMeshInstances();
    if (subdInstances.empty())
        return;

    std::vector<RTXMGInstanceData> cpuInstances;
    cpuInstances.reserve(subdInstances.size());

    for (const auto& inst : subdInstances)
    {
        RTXMGInstanceData id{};
        if (inst.node)
        {
            affineToColumnMajor(inst.node->localToWorld,     id.transform.m_data);
            affineToColumnMajor(inst.node->prevLocalToWorld, id.prevTransform.m_data);
        }
        cpuInstances.push_back(id);
    }

    const size_t byteSize = cpuInstances.size() * sizeof(RTXMGInstanceData);
    if (!m_instanceBuffer || m_instanceBuffer->getDesc().byteSize != byteSize)
    {
        nvrhi::BufferDesc desc;
        desc.byteSize     = byteSize;
        desc.structStride = sizeof(RTXMGInstanceData);
        desc.debugName    = "RTXMGInstanceBuffer";
        desc.initialState = nvrhi::ResourceStates::ShaderResource;
        desc.keepInitialState = true;
        m_instanceBuffer = m_device->createBuffer(desc);
    }
    commandList->writeBuffer(m_instanceBuffer, cpuInstances.data(), byteSize);
}

void RTXMGScene::ApplyClusterLodGrid(uint32_t numCopies, float gap, uint32_t gridBits)
{
    // A double-apply would expand the already-expanded list to N² entries.
    if (m_clusterLodGridApplied)
        return;

    const rtxmg::InstanceGridResult result = rtxmg::ReplicateInstancesOnGrid(
        rtxmg::InstanceGridScene{
            .clusterLodGeometries = m_clusterLodGeometries,
            .subdMeshes           = m_subdMeshes,
            .clusterLodInstances  = m_clusterLodInstances,
            .subdInstances        = m_subdInstances,
            .sceneGraph           = m_sceneGraph },
        numCopies, gap, gridBits);

    m_originalClusterLodBbox = result.originalBbox;
    m_clusterLodGridApplied  = result.expanded;
}

void RTXMGScene::FinishedLoading(nvrhi::ICommandList* commandList)
{
    // Refresh SceneGraph transforms (frame 0).
    m_sceneGraph.Refresh();

    // Sync Instance::localToWorld from SceneNode (used by cluster tessellator).
    for (auto& inst : m_subdInstances)
    {
        if (inst.node)
            inst.localToWorld = inst.node->localToWorld;
    }

    // Build material buffer first (needed before surfaceToMaterialIndex).
    m_materialLibrary.BuildMaterialBuffer(commandList);

    // Build surfaceToMaterialIndex for every subd mesh.
    for (size_t meshIdx = 0; meshIdx < m_subdMeshes.size(); ++meshIdx)
    {
        m_subdMeshes[meshIdx]->BuildSurfaceToMaterialIndex(
            m_materialLibrary.GetSubdSubshapeMaterialIndices(meshIdx), commandList, m_descriptorTable);
    }

    m_materialLibrary.BuildClusterLodLocalMaterialIDs(m_clusterLodStorages, commandList);

    // Build instance buffer.
    BuildInstanceBuffer(commandList);
}

void RTXMGScene::Refresh(nvrhi::ICommandList* commandList, uint32_t frameIndex)
{
    (void)frameIndex;

    // Update SceneGraph (propagates TRS → localToWorld for all nodes).
    m_sceneGraph.Refresh();

    // Sync Instance::localToWorld from SceneNode (used by cluster tessellator).
    for (auto& inst : m_subdInstances)
    {
        if (inst.node)
            inst.localToWorld = inst.node->localToWorld;
    }

    // Re-upload instance buffer with fresh transforms.
    BuildInstanceBuffer(commandList);
}
