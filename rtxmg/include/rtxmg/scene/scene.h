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

#include <json/json.h>

#include "rtxmg/scene/obj_importer.h"

#include "rtxmg/cluster_lod/baked_geometry.h"
#include "rtxmg/cluster_lod/cluster_lod_gltf_importer.h"
#include "rtxmg/cluster_lod/gltf_model.h"
#include "rtxmg/cluster_lod/resources_base.h"  // ClusterLodPrebuiltGeometryMetadata

#include "rtxmg/subdivision/subdivision_surface.h"
#include "rtxmg/subdivision/topology_map.h"
#include "rtxmg/cluster_tess/cluster_tessellator.h"

#include "rtxmg/scene/material.h"
#include "rtxmg/scene/material_library.h"
#include "rtxmg/scene/scene_graph.h"
#include "rtxmg/scene/texture_loader.h"

#include <donut/core/math/math.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/engine/ShaderFactory.h>

#include <nvrhi/nvrhi.h>

#include <atomic>
#include <string>
#include <unordered_map>


using namespace donut::engine;
using namespace donut::math;

struct View
{
    float3 position = { 0.f, 0.f, -1.f };
    float3 lookat = { 0.f, 0.f, -1.f };
    float3 up = { 1.f, 1.f, 1.f };
    float fov = 35.f;
};

enum TextureType
{
    ALBEDO = 0,
    ROUGHNESS,
    SPECULAR,
    DISPLACEMENT,
    ENVMAP,
    TEXTURE_TYPE_COUNT
};

// Parameters passed to the RTXMGScene constructor.
struct RTXMGSceneParams
{
    nvrhi::IDevice*                          device          = nullptr;
    const fs::path*                          mediaPath       = nullptr;
    std::shared_ptr<CommonRenderPasses>      commonPasses;
    std::shared_ptr<donut::vfs::IFileSystem> fs;
    std::shared_ptr<TextureCache>            textureCache;
    std::shared_ptr<DescriptorTableManager>  descriptorTable;
    int2  initialFrameRange = { std::numeric_limits<int>::max(),
                                std::numeric_limits<int>::min() };
    int   isoLevelSharp     = 6;
    int   isoLevelSmooth    = 3;
    bool  logClusterLod = true;
    // Baker config for cluster-LOD gltf imports.  Part of the .nvsngeo cache key,
    // so any change to it forces a rebake.
    BakerConfig clusterBakerConfig;
    // Optional override for the cluster-LoD shard cache directory.  Empty derives
    // a shared "_nvsngeocache" folder next to the gltf instead.
    fs::path clusterCacheDir;
    // Parallel bake worker count; 0 derives it from the core count and RAM.
    uint32_t clusterBakeWorkers = 0;
    // Polled during the load; once set, LoadWithThreadPool bails and returns false.
    const std::atomic<bool>* cancelLoad = nullptr;
};

class RTXMGScene
{
public:
    struct Attributes
    {
        std::string audio;
        float audioStartTime = 0.f;

        int2 frameRange = { std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::min() };
        float frameRate = 0.f;

        float averageInstanceScale = 0.f;
        // Median world-space instance size — a robust default for camera move
        // speed, where huge vista objects would skew the average.
        float medianInstanceScale = 0.f;
    };

    explicit RTXMGScene(const RTXMGSceneParams& params);
    ~RTXMGScene();

    // outRecordedCommandLists, when given, receives the closed upload lists instead
    // of them being submitted here — so this can run off the main thread.
    bool LoadWithThreadPool(const std::filesystem::path& filename,
        ThreadPool* threadPool,
        std::vector<nvrhi::CommandListHandle>* outRecordedCommandLists = nullptr);

    // Per-texture base mips from the budget pre-pass (see TextureLoader).  Must be
    // set BEFORE LoadWithThreadPool, which is when the textures load.
    void SetTextureBaseMips(std::unordered_map<std::string, uint32_t> baseMips)
    {
        m_textureLoader.SetBaseMips(std::move(baseMips));
    }

    void FinishedLoading(nvrhi::ICommandList* commandList);

    void Refresh(nvrhi::ICommandList* commandList, uint32_t frameIndex);

    const RTXMGSceneGraph& GetSceneGraph() const { return m_sceneGraph; }

    nvrhi::IBuffer* GetInstanceBuffer()  const { return m_instanceBuffer; }
    nvrhi::IBuffer* GetMaterialBuffer()  const { return m_materialLibrary.GetMaterialBuffer(); }

    const Attributes& GetAttributes() const { return m_attributes; }
    void InsertModel(Model&& model);

    const std::vector<std::unique_ptr<SubdivisionSurface>>&
        GetSubdMeshes() const
    {
        return m_subdMeshes;
    }

    void InsertClusterLodModel(ClusterLodModel&& model);

    const std::vector<GeometryView>& GetClusterLodGeometries() const
    {
        return m_clusterLodGeometries;
    }

    const std::vector<ClusterLodInstance>& GetClusterLodInstances() const
    {
        return m_clusterLodInstances;
    }

    // One entry per cluster-LOD geometry, handed to streaming/preload init via
    // SetPrebuiltGeometryMetadata so the render thread doesn't create thousands
    // of LoD-tree buffers at scene finish.
    const std::vector<ClusterLodPrebuiltGeometryMetadata>& GetClusterLodPrebuiltGeometryMetadata() const
    {
        return m_clusterLodPrebuiltMetadata;
    }

    bool HasClusterLod() const { return m_hasClusterLod; }
    // The config every loaded geometry validated against, so it is also the one
    // they were baked with: a mismatch re-bakes or hard-fails at load.
    const BakerConfig& GetClusterBakerConfig() const { return m_clusterBakerConfig; }

    // Cluster-LoD aggregation stats, reduced across m_clusterLodGeometries at load.
    // initClas sizes scratch from these because it needs the observed max after
    // baking, not the bake-time targets.
    uint32_t GetMaxClusterTriangles()      const { return m_maxClusterTriangles;      }
    uint32_t GetMaxClusterVertices()       const { return m_maxClusterVertices;       }

    // True if any cluster-LoD material is non-OPAQUE (MASK or BLEND) / double
    // sided.  Drives the alpha-mask geometry-index dispatch and the CLAS-build
    // geometry count, so BLEND counts here even though it shades differently.
    bool HasClusterLodAlphaMask() const { return m_hasAlphaMask; }

    // cluster-LoD material indirection, all owned by the material library: base =
    // the material-buffer slot where the cluster-LoD materials start (they follow
    // the subd ones); the IDs buffer is every geometry's localMaterialIDs[]
    // concatenated, sliced by shaderio::Geometry::localMaterialsOffset/Count and
    // biased by the base in the shader, not here.
    uint32_t GetClusterLodMaterialBaseID() const { return m_materialLibrary.GetClusterLodMaterialBaseID(); }
    nvrhi::IBuffer* GetClusterLodLocalMaterialIDsBuffer() const { return m_materialLibrary.GetClusterLodLocalMaterialIDsBuffer(); }
    const std::vector<uint32_t>& GetClusterLodLocalMaterialsOffsets() const
    {
        return m_materialLibrary.GetClusterLodLocalMaterialsOffsets();
    }

    // Diagnostic: --nomat.  Must be set BEFORE LoadWithThreadPool, which is when
    // the materials are built.
    void SetEnableMaterials(bool b) { m_materialLibrary.SetEnableMaterials(b); }
    bool GetEnableMaterials() const { return m_materialLibrary.GetEnableMaterials(); }

    // --normalmaps.  Same timing constraint as SetEnableMaterials, and it must
    // match what ApplyKtxTextureBudget was told.
    void SetEnableNormalMaps(bool b) { m_materialLibrary.SetEnableNormalMaps(b); }

    // Replicate cluster-LoD instances on a grid.  Must run after model load and
    // before FinishedLoading, so the GPU buffers see the expanded instance count.
    //   gap is grid spacing as a multiple of the model AABB extent.
    //   gridBits: 0..2 = placement axes XYZ, 3..5 = random rotation axes XYZ.
    void ApplyClusterLodGrid(uint32_t numCopies, float gap, uint32_t gridBits);

    // World-space bbox of the cluster-LoD instances before any grid expansion, so
    // camera framing can look down the grid from the original instance.
    const box3& GetOriginalClusterLodBbox() const { return m_originalClusterLodBbox; }
    bool        IsClusterLodGridApplied() const { return m_clusterLodGridApplied; }

    donut::engine::DescriptorTableManager* GetDescriptorTable() const
    {
        return m_descriptorTable.get();
    }

    const std::vector<std::unique_ptr<TopologyMap const>>&
        GetTopologyMaps() const
    {
        return m_topologyMaps;
    }
    std::span<Instance>       GetSubdMeshInstances()       { return m_subdInstances; }
    std::span<Instance const> GetSubdMeshInstances() const { return m_subdInstances; }
    uint32_t TotalSubdPatchCount() const;
    // Subd meshes, their animation poses and the topology maps.
    uint64_t GetSubdVramBytes() const;

    std::span<SubdivisionSurface::Pose const> GetSubdPoses() const { return m_subdPoses; }
    // Null for an instance of a static mesh, which reads the mesh's rest pose.
    const SubdivisionSurface::Pose* GetSubdPose(const Instance& instance) const;
    // SRV of the instance's current positions: its pose slice, or the rest pose.
    nvrhi::BindingSetItem GetSubdPositions(uint32_t slot, const Instance& instance) const;
    nvrhi::IBuffer* GetSubdPatchPoints(const Instance& instance) const;
    box3 GetSubdAabb(const Instance& instance) const;  // object space, current frame

    const View* GetView() const { return m_view.get(); }

    void Animate(float animTime, float animRate);

    nvrhi::SamplerHandle GetDisplacementSampler() const { return m_commonPasses->m_LinearWrapSampler; }

    const Json::Value& GetSceneSettings() const { return m_sceneSettings; }
    std::string& GetInputPath() { return m_inputPath; }
    const std::string& GetInputPath() const { return m_inputPath; }

    const std::vector<std::shared_ptr<RTXMGMaterial>>& GetMaterials() const
    {
        return m_materialLibrary.GetMaterials();
    }

protected:
    void LoadSceneFile(const std::filesystem::path& filename,
        std::unique_ptr<ObjImporter>& objImporter,
        nvrhi::ICommandList* commandList);

private:
    // LoadWithThreadPool's phases, in call order.
    bool ImportModels(const std::filesystem::path& filepath,
                      std::unique_ptr<ObjImporter>& objImporter,
                      nvrhi::ICommandList* commandList);
    void ReduceSceneStatistics();
    void BuildSubdSceneNodes(const std::shared_ptr<RTXMGSceneNode>& root);
    void BuildClusterLodSceneNodes(const std::shared_ptr<RTXMGSceneNode>& root);

    void BuildInstanceBuffer(nvrhi::ICommandList* commandList);

    // ---------------------------------------------------------------------------
    // Owned infrastructure
    // ---------------------------------------------------------------------------
    nvrhi::IDevice*                               m_device          = nullptr;
    std::shared_ptr<donut::vfs::IFileSystem>      m_fs;
    std::shared_ptr<TextureCache>                 m_textureCache;
    std::shared_ptr<DescriptorTableManager>       m_descriptorTable;
    rtxmg::TextureLoader                          m_textureLoader;
    rtxmg::MaterialLibrary                        m_materialLibrary;

    // GPU buffers owned by RTXMGScene
    nvrhi::BufferHandle m_instanceBuffer;  // RTXMGInstanceData[]

    // ---------------------------------------------------------------------------
    // Scene data
    // ---------------------------------------------------------------------------
    Attributes m_attributes;
    Json::Value m_sceneSettings;
    const fs::path& m_mediaPath;

    int m_isoLevelSharp;
    int m_isoLevelSmooth;

    std::unique_ptr<View> m_view;

    RTXMGSceneGraph m_sceneGraph;

    std::vector<std::unique_ptr<TopologyMap const>> m_topologyMaps;
    std::vector<std::unique_ptr<SubdivisionSurface>> m_subdMeshes;
    std::vector<Instance> m_subdInstances;
    std::vector<SubdivisionSurface::Pose> m_subdPoses;
    // One pose per (animated mesh, frame offset); sets each Instance::poseID.
    void BuildSubdPoses(nvrhi::ICommandList* commandList);
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;

    // Cluster LOD CPU data
    std::vector<GeometryView>            m_clusterLodGeometries;
    std::vector<GeometryStorage>         m_clusterLodStorages;
    CacheFileView                        m_cacheFileView;      // monolith mmap lifetime
    std::vector<ShardCache>              m_shardCaches;        // per-model shard mmap lifetimes
    std::vector<ClusterLodInstance>      m_clusterLodInstances;
    std::vector<ClusterLodMaterialDesc>  m_clusterLodMaterialDescs;
    // Pre-uploaded per-geometry GPU metadata (see GetClusterLodPrebuiltGeometryMetadata).
    std::vector<ClusterLodPrebuiltGeometryMetadata> m_clusterLodPrebuiltMetadata;
    bool                                 m_hasClusterLod = false;
    bool                                 m_logClusterLod = false;
    BakerConfig                          m_clusterBakerConfig;
    fs::path                             m_clusterCacheDir;  // shard-cache override (testing)
    uint32_t                             m_clusterBakeWorkers = 0;
    const std::atomic<bool>*             m_cancelLoad = nullptr;
    bool IsLoadCancelled() const { return m_cancelLoad && m_cancelLoad->load(std::memory_order_relaxed); }

    // Valid only inside LoadWithThreadPool: the pool the metadata build fans out
    // over, and the per-worker upload lists it records.
    ThreadPool*                           m_loadThreadPool = nullptr;
    std::vector<nvrhi::CommandListHandle> m_metadataCommandLists;

    // Each gltf of the current load, keyed by normalized path, with its instances
    // as first inserted; later references only add instances.
    struct InsertedClusterModel
    {
        std::filesystem::path           path;
        std::vector<ClusterLodInstance> instances;
    };
    std::vector<InsertedClusterModel> m_insertedClusterModels;

    // Each OBJ of the current load, keyed by resolved path, with its mesh.
    struct InsertedSubdModel
    {
        std::filesystem::path path;
        uint32_t              meshID;
    };
    std::vector<InsertedSubdModel> m_insertedSubdModels;

    // Imports the OBJ once per load and adds `instance` of it on every call.
    void InsertSubdObj(const std::filesystem::path& objPath, const std::string& modelName,
                       const Instance& instance, ObjImporter& objImporter,
                       nvrhi::ICommandList* commandList);

    // Inserts the gltf at `gltfPath` once per load and instances it under
    // `nodeTransform` on every call.
    bool InsertClusterLodGltf(const std::filesystem::path& gltfPath, const affine3& nodeTransform,
                              nvrhi::ICommandList* commandList);
    // Appends a model's geometries and builds their GPU metadata, keeping
    // m_clusterLodPrebuiltMetadata index-aligned with the geometries.
    void InsertClusterLodModelWithMetadata(ClusterLodModel&& model, nvrhi::ICommandList* commandList);

    // Scene-wide cluster-LoD aggregation stats — reduced across
    // m_clusterLodGeometries in LoadWithThreadPool alongside the AABB pass.
    uint32_t m_maxClusterTriangles       = 0;
    uint32_t m_maxClusterVertices        = 0;

    // Scene-wide summary flag derived from m_clusterLodMaterialDescs.  Deliberately
    // conservative: one material in one geometry is enough to flip it.
    bool     m_hasAlphaMask              = false;

    // Captured at the top of ApplyClusterLodGrid (see GetOriginalClusterLodBbox).
    box3 m_originalClusterLodBbox       = box3::empty();
    bool m_clusterLodGridApplied        = false;

    std::string m_inputPath;
};
