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

// ClusterLodResourcesBase — shared base for ClusterLodPreloaded and
// ClusterLodStreaming.  Holds the per-scene resources both modes need:
//   - Per-geometry shaderio table + its GPU buffer
//   - Per-instance render-instance table + its GPU buffer
//   - The low-detail-BLAS buffer (one sub-BLAS per geometry, used as the
//     fallback BLAS for the TLAS when streaming hasn't filled in a more
//     detailed BLAS for an instance)
//   - Maximum LoD-node tree depth (for traversal pass sizing)

#pragma once

#include <algorithm>
#include <vector>

#include <donut/engine/DescriptorTableManager.h>
#include <nvrhi/nvrhi.h>

#include "rtxmg/cluster_lod/baked_geometry.h"  // GeometryView (param to UploadGeometryMetadata)
#include "rtxmg/cluster_lod/baker.h"           // BakerConfig (ResolveClasPositionTruncateBits)
#include "rtxmg/cluster_lod/resources.h"       // ClusterLodResources interface
#include "rtxmg/cluster_lod/shaderio.h"
#include "rtxmg/utils/buffer.h"

// Bits the CLAS builder drops from each vertex position: the explicit knob,
// raised to the bake's mantissa drop when the geometry was baked compressed
// (those bits are already zero, so it costs no extra precision).  Every build
// path resolves it here, and each path's sizing site must declare the same
// value its per-cluster args use — minPositionTruncateBitCount is a Min field,
// so over-declaring at sizing time undersizes scratch.
inline uint32_t ResolveClasPositionTruncateBits(uint32_t knobBits, const BakerConfig& bakerConfig)
{
    return std::max(knobBits,
                    bakerConfig.useCompressedData ? bakerConfig.compressionPosDropBits : 0u);
}

// A geometry's element range inside a buffer shared by many geometries.
template <typename T>
struct GeometryMetadataSlice
{
    using ElementType = T;

    nvrhi::BufferHandle buffer;
    uint32_t            firstElement = 0;
    uint32_t            numElements  = 0;

    uint64_t           ByteOffset() const { return uint64_t(firstElement) * sizeof(T); }
    uint64_t           ByteSize() const { return uint64_t(numElements) * sizeof(T); }
    nvrhi::BufferRange Range() const { return nvrhi::BufferRange(ByteOffset(), ByteSize()); }
};

// ---------------------------------------------------------------------------
// ClusterLodMetadataArena — one shared buffer per metadata kind for a batch of
// geometries, each geometry owning an aligned slice, so a city-scale scene needs
// a handful of buffers instead of thousands.  The slices' buffer handles keep the
// buffers alive, so the arena itself is only needed while the batch is being built.
// ---------------------------------------------------------------------------
struct ClusterLodMetadataArena
{
    RTXMGBuffer<shaderio::LodLevel>     lodLevels;
    RTXMGBuffer<shaderio::Node>         lodNodes;
    RTXMGBuffer<shaderio::BBox>         lodNodeBboxes;
    RTXMGBuffer<shaderio::GroupAddress> groupAddresses;

    struct Slices
    {
        GeometryMetadataSlice<shaderio::LodLevel>     lodLevels;
        GeometryMetadataSlice<shaderio::Node>         lodNodes;
        GeometryMetadataSlice<shaderio::BBox>         lodNodeBboxes;
        GeometryMetadataSlice<shaderio::GroupAddress> groupAddresses;
    };
    std::vector<Slices> slices;  // parallel to Create's input; empty for null entries

    // Null entries get no slice, so a batch can skip already-built geometries.
    static ClusterLodMetadataArena Create(const std::vector<const GeometryView*>& geometries,
                                          nvrhi::IDevice*                        device);
};

// ---------------------------------------------------------------------------
// BaseGeometry — per-geometry resources both preloaded and streaming need.
// The per-cluster CLAS-address / cluster-address tables are NOT here: they are
// scene-global on ClusterLodResourcesBase, indexed by `clusterResidentID`.
// ---------------------------------------------------------------------------
struct BaseGeometry
{
    // LoD hierarchy (one entry per LoD level / node).
    GeometryMetadataSlice<shaderio::LodLevel>     lodLevels;
    GeometryMetadataSlice<shaderio::Node>         lodNodes;
    GeometryMetadataSlice<shaderio::BBox>         lodNodeBboxes;

    // Per-group blob address table.
    GeometryMetadataSlice<shaderio::GroupAddress> streamingGroupAddresses;

    // Descriptor handles — keep ResourceDescriptorHeap registrations alive
    // for the lifetime of this geometry.  Heap indices stored in the
    // corresponding shaderio::Geometry fields.  When the metadata was
    // pre-uploaded at scene load (ClusterLodPrebuiltGeometryMetadata), the SCENE
    // owns the handles and these stay empty.
    donut::engine::DescriptorHandle lodLevelsSRVHandle;
    donut::engine::DescriptorHandle nodesSRVHandle;
    donut::engine::DescriptorHandle nodeBboxesSRVHandle;
    donut::engine::DescriptorHandle streamingGroupAddressesSRVHandle;
    donut::engine::DescriptorHandle streamingGroupAddressesUAVHandle;
};

// ---------------------------------------------------------------------------
// ClusterLodPrebuiltGeometryMetadata — the expensive half of UploadGeometryMetadata
// (arena upload + bindless registration + node-tree depth scan),
// pre-built per geometry on the async scene-load thread with loading-screen
// progress instead of stalling the render thread inside streaming/preload init
// (~8 s for a scene with a couple of thousand geometries).  The scene owns the
// entries and their descriptor registrations; init copies only the refcounted
// handles + heap indices, so re-inits reuse them.
// ---------------------------------------------------------------------------
struct ClusterLodPrebuiltGeometryMetadata
{
    BaseGeometry base;   // buffers + descriptor handles (owned here)

    // Heap indices of base's registrations (become shaderio::Geometry fields).
    uint32_t lodLevelsSRV      = 0;
    uint32_t nodesSRV          = 0;
    uint32_t nodeBboxesSRV     = 0;
    uint32_t groupAddressesSRV = 0;
    uint32_t groupAddressesUAV = 0;

    // LoD node tree depth (traversal pass sizing).
    uint32_t nodeTreeDepth = 1;

    // False for a default-constructed entry whose geometry hasn't been built yet.
    bool IsValid() const { return base.lodLevels.buffer != nullptr; }

    // Upload + register everything for one geometry into its slice
    // (arena.slices[arenaIndex]).  Safe to call from the scene-load thread
    // concurrently with render-thread texture finalize and with other geometries
    // of the same arena (DescriptorTableManager locks internally; per-thread
    // command-list recording is free-threaded and slices are disjoint).
    static ClusterLodPrebuiltGeometryMetadata Build(
        const GeometryView&                     geom,
        const ClusterLodMetadataArena&          arena,
        size_t                                  arenaIndex,
        donut::engine::DescriptorTableManager*  descriptorTable,
        nvrhi::ICommandList*                    commandList);
};

class ClusterLodResourcesBase : public ClusterLodResources
{
public:
    // ---- ClusterLodResources interface overrides ----------------------------

    const RTXMGBuffer<shaderio::Geometry>& GetShaderGeometriesBuffer() const override
    {
        return m_shaderGeometriesBuffer;
    }

    const RTXMGBuffer<shaderio::RenderInstance>& GetShaderRenderInstancesBuffer() const override
    {
        return m_renderInstancesBuffer;
    }

    uint32_t GetRenderInstanceCount() const override
    {
        return static_cast<uint32_t>(m_renderInstances.size());
    }

    uint32_t GetMaxNodeTreeDepth() const override { return m_maxNodeTreeDepth; }

    uint32_t GetMaxPerGeometryGroups() const override { return m_maxPerGeometryGroups; }

    uint32_t GetMaxClustersPerGroup() const override { return m_maxClustersPerGroup; }

    // ---- Common queries -----------------------------------------------------

    // Low-detail-LOD BLAS buffer (one sub-BLAS per geometry, backed by a
    // single shared allocation).  Used as the fallback BLAS for the TLAS
    // when streaming hasn't filled in a higher-detail per-instance BLAS.
    const RTXMGBuffer<uint8_t>& GetLowDetailBlasBuffer() const
    {
        return m_clasLowDetailBlasBuffer;
    }

    // Scene-global per-resident-cluster tables.  Indexed by
    // `clusterResidentID` (allocator-issued in streaming, bump-allocated at
    // Init in preload).  Bound directly into the BLAS-build + hit-shader
    // binding sets — no per-geom bindless detour.
    const RTXMGBuffer<uint64_t>& GetResidentClasAddressesBuffer() const override
    {
        return m_residentClasAddresses;
    }

    const RTXMGBuffer<shaderio::ClusterAddress>& GetResidentClustersBuffer() const override
    {
        return m_residentClusters;
    }

    // Scene-global StreamingGroup table.  Streaming sizes it to
    // maxResidentGroups; preload allocates one slot per scene-global group
    // resident ID so traversal_run_groups' age reset has a valid UAV target
    // even though preload never age-filters.
    const RTXMGBuffer<shaderio::StreamingGroup>& GetResidentGroupsBuffer() const override
    {
        return m_residentGroupsBufferBase;
    }

    void SetDebugClusterLod(bool enabled)
    {
        m_debugClusterLod = enabled;
    }

    // Debug override: when false, every cluster is treated as opaque /
    // single-sided / material 0 at runtime, bypassing the alpha-mask
    // geometryIndexAndFlagsBuffer chain on host and GPU.  Must be set BEFORE
    // Init() so the RenderInstance / CLAS-arg writes see it.
    void SetEnableMaterials(bool enabled)
    {
        m_enableMaterials = enabled;
    }
    bool GetEnableMaterials() const { return m_enableMaterials; }

    // Scene-owned pre-uploaded per-geometry metadata (one entry per geometry,
    // same order as the geometry array).  Must be set BEFORE Init and outlive
    // this object; UploadGeometryMetadata adopts its buffers and registrations.
    void SetPrebuiltGeometryMetadata(const std::vector<ClusterLodPrebuiltGeometryMetadata>* prebuilt)
    {
        m_prebuiltMetadata = prebuilt;
    }

    // Scene-owned per-geometry offsets into the flat localMaterialIDs buffer, in
    // geometry order. Same contract as above: set BEFORE Init and outlive this
    // object. Absent leaves every offset 0, which is correct only for a scene
    // whose geometries have no local materials.
    void SetClusterLodLocalMaterialsOffsets(const std::vector<uint32_t>* offsets)
    {
        m_clusterLodLocalMaterialsOffsets = offsets;
    }

protected:
    // Shared per-geometry init, called from both derived classes' Init loops:
    // adopts the pre-built LoD tree buffers and bindless SRVs, fills the base
    // fields of `outShaderGeom`, and updates m_maxNodeTreeDepth.
    //
    // Does NOT touch:
    //   - groupData / clasData (preload-only, in PreloadGeometry)
    //   - the scene-global resident CLAS / cluster tables (preload fills them
    //     via InitClas; streaming fills the coarsest LoD at Init and the rest
    //     incrementally)
    //   - streamingGroupAddresses entries beyond the invalid default (derived
    //     classes patch in preload / low-detail residency)
    void UploadGeometryMetadata(
        size_t                                  geomIndex,
        const GeometryView&                     geom,
        BaseGeometry&                           outBase,
        shaderio::Geometry&                     outShaderGeom,
        uint32_t                                instancesOffset,
        uint32_t                                instancesCount);

    // Protected so the derived classes can read and patch them in place.
    std::vector<shaderio::Geometry>          m_shaderGeometries;
    std::vector<shaderio::RenderInstance>    m_renderInstances;

    RTXMGBuffer<shaderio::Geometry>          m_shaderGeometriesBuffer;
    RTXMGBuffer<shaderio::RenderInstance>    m_renderInstancesBuffer;
    RTXMGBuffer<uint8_t>                     m_clasLowDetailBlasBuffer;

    // Scene-global per-resident-cluster tables.  Created by the
    // derived class' Init (with the right capacity for that mode):
    //   - ClusterLodPreloaded:  sceneTotalClusters (all clusters of all geoms).
    //   - ClusterLodStreaming:  maxResidentClusters (= maxResidentGroups *
    //                           maxClustersPerGroup).
    // Population:
    //   - m_residentClasAddresses written by stream_allocator_alloc_groups
    //     (streaming) / Init (preload + streaming initial residency).
    //   - m_residentClusters written CPU-side by Init + per-frame
    //     UpdateGPU (streaming) / Init (preload).
    RTXMGBuffer<uint64_t>                    m_residentClasAddresses;
    RTXMGBuffer<shaderio::ClusterAddress>    m_residentClusters;

    // ClusterLodStreaming overrides the accessor to return its
    // StreamingResident-owned table; ClusterLodPreloaded fills this scene-sized
    // placeholder, needed only because the traversal binding set is shared.
    RTXMGBuffer<shaderio::StreamingGroup>    m_residentGroupsBufferBase;

    uint32_t m_maxNodeTreeDepth = 1;

    // Largest per-geometry group count seen during UploadGeometryMetadata.
    // Bounds the per-geom groupIndex; sizes ClusterLodPass' diagnostic bitmap.
    uint32_t m_maxPerGeometryGroups = 0;

    // Streaming overwrites this with the bake's BakerConfig::clusterGroupSize;
    // the default matches BakerConfig so preload (which never runs the merge
    // kernel) still reports a sane value.
    uint32_t m_maxClustersPerGroup = 32;

    // Scene-owned pre-uploaded metadata (see SetPrebuiltGeometryMetadata),
    // required: every geometry's entry is built during scene load.
    const std::vector<ClusterLodPrebuiltGeometryMetadata>* m_prebuiltMetadata = nullptr;

    // Scene-global cluster-LoD material base, copied by Init() into every
    // shaderio::Geometry::materialBaseID.
    uint32_t m_clusterLodMaterialBaseID = 0;
    // Scene-owned offsets into RTXMGScene::m_clusterLodLocalMaterialIDsBuffer, read
    // by geometry index (see SetClusterLodLocalMaterialsOffsets).
    const std::vector<uint32_t>* m_clusterLodLocalMaterialsOffsets = nullptr;
    // Scene-wide: does any cluster-LoD material use alpha masking?  Drives the
    // CLAS-build maxUniqueGeometryCount/maxGeometryIndex (2/1 vs 1/0) and gates
    // the per-geometry mixed-cluster geometryIndexAndFlagsBuffer allocation.
    bool     m_hasAlphaMaskScene        = false;
    // Resolved once at Init so every CLAS sizing site and per-cluster arg on
    // this path declares one value (see ResolveClasPositionTruncateBits).
    uint32_t m_clasPositionTruncateBits  = 0;
    bool     m_debugClusterLod  = false;
    // See SetEnableMaterials.
    bool     m_enableMaterials  = true;
};
