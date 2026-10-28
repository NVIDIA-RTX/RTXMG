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

// ClusterLodBlasPass — builds one dynamic BLAS per cluster LOD instance per frame.
//
// Pipeline (called AFTER ClusterLodPass::Execute):
//   1. blas_reserve_clusters (direct) — allocates CLAS-address pool sub-arrays per instance
//   2. blas_insert_clusters (indirect via SceneBuildingCounters dispatch args) — writes CLAS VAs
//   3. executeMultiIndirectClusterOperation(BlasBuild) — builds one BLAS per instance
//
// After Execute(), GetBlasAddressesBuffer() holds the per-instance BLAS device addresses,
// ready to be fed into RTXMGRenderer::FillInstanceDescs.

#pragma once

#include <array>
#include <vector>

#include <nvrhi/nvrhi.h>
#include <donut/engine/ShaderFactory.h>

#include "rtxmg/cluster_lod/pass.h"
#include "rtxmg/cluster_lod/resources.h"
#include "rtxmg/utils/buffer.h"
#include "rtxmg/utils/cluster_operation.h"

struct BlasBuildParams;

struct ClusterLodBlasPassParams
{
    // Runtime-togglable (no recompile): the BLAS pass selects the matching
    // instance_assign_blas permutation and runs the cached-BLAS
    // setup/build/copy/move blocks only when useBlasCaching is set.
    bool useBlasSharing = false;
    bool useBlasCaching = false;
    // Host-side count of cached BLASes that need (re)building this frame.  Only
    // meaningful when useBlasCaching.
    uint32_t patchCachedBlasCount = 0u;

    // Streaming-owned resources the caching dispatches consume; null/0 on the
    // non-caching path.
    //  * geometryPatches — StructuredBuffer<StreamingGeometryPatch> view into
    //    the streaming patches ring; byteOffset selects this frame's task slot.
    //  * cachedBlasPoolBlocks — the cached-BLAS pool's AS-storage blocks, which
    //    the pass transitions to AccelStructBuildBlas after the MOVE.
    nvrhi::IBuffer* geometryPatchesBuffer     = nullptr;
    uint64_t        geometryPatchesByteOffset = 0u;
    const std::vector<nvrhi::IBuffer*>* cachedBlasPoolBlocks = nullptr;
};

// instance_assign_blas.hlsl is compiled as a USE_BLAS_SHARING × USE_BLAS_CACHING
// cross-product; this 2-bit permutation indexes the lazy PSO array so either
// feature is runtime-togglable without a recompile.  Caching implies sharing,
// so the {caching=1, sharing=0} slot is never requested.
class InstanceAssignBlasPermutation
{
public:
    enum BitIndices : uint32_t
    {
        Sharing = 0,
        Caching,
        Count
    };
    static constexpr size_t kCount = 1u << BitIndices::Count;  // 4

    InstanceAssignBlasPermutation(bool sharing, bool caching)
        : m_bits((sharing ? (1u << BitIndices::Sharing) : 0u)
               | (caching ? (1u << BitIndices::Caching) : 0u))
    {}

    uint32_t Index()            const { return m_bits; }

private:
    uint32_t m_bits = 0;
};

class ClusterLodBlasPass
{
public:
    // Init — allocates buffers and creates PSOs.
    // Takes references to the cluster-LOD resources and the traversal pass
    // (which exposes the instance-build-info buffer written during traversal).
    void Init(const ClusterLodResources&   resources,
              const ClusterLodPass&        traversalPass,
              nvrhi::DescriptorTableHandle  descriptorTable,
              donut::engine::ShaderFactory* shaderFactory,
              nvrhi::IDevice*              device,
              bool                         debugClusterLod = false,
              // Extra BLAS-build slots reserved for per-geometry cached BLASes;
              // the per-frame build arrays are sized to numInstances + this.
              // 0 when caching is unavailable.
              uint32_t                     maxCachedBlasBuilds = 0u,
              // Total cached-BLAS pool budget in bytes; sizes the MOVE_OBJECTS
              // scratch.
              uint64_t                     cachedBlasPoolBytes = 0u);

    // Execute — runs the Blas Build Preparation + Blas Build + Blas Copy +
    // Tlas Preparation block.  Must be called AFTER ClusterLodPass::Execute().
    void Execute(nvrhi::IDevice*       device,
                 nvrhi::ICommandList*  commandList,
                 ClusterLodPass&       traversalPass,
                 const ClusterLodBlasPassParams& params = {});

    // Per-instance BLAS device addresses indexed by BLAS build order.
    // For TLAS fill use GetInstanceBlasAddrsBuffer() (from ClusterLodPass) instead.
    nvrhi::IBuffer* GetBlasAddressesBuffer() const { return m_blasAddresses.GetBuffer(); }

    // Per-build BLAS byte sizes (entries [0..blasBuildCounter) are valid after
    // Execute).  Summed host-side for the Profiler "Streaming" tab's BLAS memory.
    ClusterOperationSizesBuffer& GetBlasSizesTyped() { return m_blasSizes; }

    // Build args and address/size arrays, but NOT m_blasBuffer — the BLAS
    // storage is its own VRAM bucket.  Feeds Cluster LOD Metadata.
    uint64_t GetMetadataBytes() const;
    // The BLAS storage at its worst-case size, of which only part is built each frame.
    uint64_t GetBlasBufferBytes() const { return m_blasBuffer.GetVramBytes(); }

private:
    void CreatePipelines(donut::engine::ShaderFactory* shaderFactory,
                         nvrhi::IDevice* device);
    // Lazily creates (on first use) and returns the instance_assign_blas PSO for
    // the requested permutation, so only dispatched permutations get compiled.
    nvrhi::ComputePipelineHandle GetInstanceAssignBlasPipeline(bool useBlasSharing, bool useBlasCaching);
    void DebugReadbackTraversalOutput(nvrhi::ICommandList* commandList,
                                      ClusterLodPass& traversalPass);

    uint32_t m_numInstances = 0;
    // numInstances + maxCachedBlasBuilds: cached BLASes append after the
    // per-instance builds within the same BLAS-build op.
    uint32_t m_maxBlasBuilds = 0;
    uint32_t m_maxCachedBlasBuilds = 0;
    bool     m_debugClusterLod = false;

    // GPU buffers
    RTXMGBuffer<nvrhi::rt::cluster::IndirectArgs> m_blasArgs;       // one per build (instances + cached)
    RTXMGBuffer<nvrhi::GpuVirtualAddress>         m_blasClasAddrs;  // shared CLAS VA pool
    // Zero the shared CLAS-VA pool on the first Execute after (re)Init: a
    // re-stream reallocates it onto recycled heap pages still holding the
    // previous streaming instance's CLAS VAs, and an under-filled gather would
    // hand one of those dangling VAs to the BLAS builder (device-removed).
    bool                                          m_blasClasAddrsNeedsClear = true;
    RTXMGBuffer<nvrhi::GpuVirtualAddress>         m_blasAddresses;  // output: per-build BLAS VA
    ClusterOperationSizesBuffer                   m_blasSizes;     // output: per-build BLAS size
    RTXMGBuffer<uint32_t>                         m_perGeomSeen;    // render-stats: per-geom low-detail/cached unique dedup
    RTXMGBuffer<uint8_t>                          m_blasBuffer;     // BLAS storage

    // Per-frame MOVE_OBJECTS src/dst address arrays staged by
    // blas_cache_stage_move; the move copies each freshly-built cached BLAS
    // into its persistent pool allocation.
    RTXMGBuffer<nvrhi::GpuVirtualAddress>         m_cachedBlasAddressesSrc;
    RTXMGBuffer<nvrhi::GpuVirtualAddress>         m_cachedBlasAddressesDst;
    nvrhi::rt::cluster::OperationParams           m_blasMoveParams{};
    nvrhi::rt::cluster::OperationSizeInfo         m_blasMoveSizeInfo{};

    // Persistent scene data
    const RTXMGBuffer<shaderio::Geometry>*       m_geometriesBuffer      = nullptr;
    const RTXMGBuffer<shaderio::RenderInstance>* m_renderInstancesBuffer = nullptr;
    // Scene-global resident-CLAS-address table (one VA per resident cluster,
    // indexed by clusterResidentID).  Cached at Init and bound as SRV(t4)
    // into blas_insert_clusters.
    const RTXMGBuffer<uint64_t>* m_residentClasAddrsBuffer = nullptr;

    // BLAS build params (from getClusterOperationSizeInfo)
    nvrhi::rt::cluster::OperationParams   m_blasParams{};
    nvrhi::rt::cluster::OperationSizeInfo m_blasSizeInfo{};

    // PSOs.
    struct Pipelines
    {
        nvrhi::ComputePipelineHandle computeBlasSetupInsertion;
        nvrhi::ComputePipelineHandle computeBlasInsertClusters;
        // instance_assign_blas.hlsl — one PSO per (sharing,caching) permutation,
        // lazily created by GetInstanceAssignBlasPipeline() and indexed by
        // InstanceAssignBlasPermutation::Index().
        std::array<nvrhi::ComputePipelineHandle, InstanceAssignBlasPermutation::kCount>
            computeInstanceAssignBlas{};

        // Compiled USE_BLAS_CACHING=1; created up front, dispatched only when
        // params.useBlasCaching.
        nvrhi::ComputePipelineHandle computeBlasCachingSetupBuild;
        nvrhi::ComputePipelineHandle computeBlasCachingSetupCopy;
    };
    Pipelines m_pipelines;

    nvrhi::BindingLayoutHandle   m_setupLayout;
    nvrhi::BindingLayoutHandle   m_insertLayout;
    nvrhi::BindingLayoutHandle   m_updateBlasLayout;
    // Binding layouts for the two cached-BLAS setup shaders.
    nvrhi::BindingLayoutHandle   m_cachingBuildLayout;
    nvrhi::BindingLayoutHandle   m_cachingCopyLayout;
    // Bindless layout for blas_cache_gather_clusters's ResourceDescriptorHeap
    // access (per-geometry lodLevels / groupAddresses SRVs + group-data blocks).
    nvrhi::BindingLayoutHandle   m_cachingBindlessLayout;

    nvrhi::DescriptorTableHandle m_descriptorTable;

    // Stashed for lazy PSO creation (GetInstanceAssignBlasPipeline).
    donut::engine::ShaderFactory* m_shaderFactory = nullptr;
    nvrhi::IDevice*               m_device        = nullptr;

    // Per-frame CB for setup_insertion / blas_insert_clusters /
    // instance_assign_blas, writeBuffer'd at the top of each Execute().
    nvrhi::BufferHandle          m_blasBuildParamsCB;

    uint64_t                     m_frameCount = 0;
};
