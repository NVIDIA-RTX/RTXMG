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

// Cluster-LoD streaming runtime.

#pragma once

#include <array>
#include <vector>

#include <donut/engine/DescriptorTableManager.h>
#include <donut/engine/ShaderFactory.h>

#include "rtxmg/cluster_lod/baked_geometry.h"     // GeometryView
#include "rtxmg/cluster_lod/gltf_model.h"         // ClusterLodInstance
#include "rtxmg/cluster_lod/resources_base.h"     // ClusterLodResourcesBase
#include "rtxmg/cluster_lod/streaming_hooks.h"    // IClusterLodStreamingHooks
#include "rtxmg/cluster_lod/streaming_utils.h"    // sub-managers
#include "rtxmg/profiler/streaming_stats.h"       // StreamingStats (GetStats)

namespace rtxmg {

//////////////////////////////////////////////////////////////////////////
//
// ClusterLodStreaming
//
// The scene is not loaded fully but streamed in: lowest-detail clusters (and
// their CLAS) are persistently resident; anything else is handled dynamically
// through a request -> upload -> update pipeline.

class ClusterLodStreaming : public ClusterLodResourcesBase, public IClusterLodStreamingHooks
{
public:
    // Out-of-line so the destructor can call deinit() — the renderer just drops
    // the unique_ptr, and m_resident's IDPool asserts on still-allocated IDs.
    ~ClusterLodStreaming() override;

    // This is the only backend with streaming hooks; preload inherits the
    // ClusterLodResources default and returns null.
    const IClusterLodStreamingHooks* GetStreamingHooks() const override { return this; }
    IClusterLodStreamingHooks*       GetStreamingHooks()       override { return this; }

    using FrameSettings = IClusterLodStreamingHooks::FrameSettings;


    // ---- init / deinit / reset / UpdateClasRequired

    // pointers must stay valid during lifetime
    //
    // maxClusterTriangles / maxClusterVertices are the scene-wide OBSERVED
    // maxima that size initClas's CLAS scratch queries — distinct from
    // bakerConfig.clusterTriangles / clusterVertices, which are bake-time
    // targets (upper bounds).
    // hasAlphaMask drives the mixed-cluster per-triangle CLAS geometry-index
    // dispatch.
    // clusterLodMaterialBaseID = slot in the scene material buffer where
    // cluster-LoD materials start; copied into every
    // shaderio::Geometry::materialBaseID so the hit shaders'
    // ResolveMaterialIDFromLocal() can turn per-cluster local material IDs into
    // t_MaterialConstants slots.
    bool Init(const std::vector<GeometryView>&        geometries,
              const std::vector<ClusterLodInstance>&  instances,
              const BakerConfig&                      bakerConfig,
              uint32_t                                maxClusterTriangles,
              uint32_t                                maxClusterVertices,
              bool                                    hasAlphaMask,
              uint32_t                                clusterLodMaterialBaseID,
              const StreamingConfig&                  config,
              donut::engine::DescriptorTableManager*  descriptorTable,
              donut::engine::ShaderFactory*           shaderFactory,
              nvrhi::IDevice*                         device,
              nvrhi::ICommandList*                    commandList);

    // Tear down — safe to call without init.
    void Deinit();

    // run prior the renderer starts referencing resources
    // if true CLAS for all clusters will be built
    bool UpdateClasRequired(bool state, nvrhi::ICommandList* commandList);

    // Frees all cached BLAS allocations and re-uploads m_shaderGeometries
    // (which carries cachedBlasLodLevel / cachedBlasAddress per geometry).
    void ResetCachedBlas(nvrhi::ICommandList* commandList) override;

    // called by render thread
    // render thread must take care of barriers prior/after these operations
    // triggers main setup, ensures data is uploaded, unloaded etc.
    //    implicitly does "handleCompletedUpdate" and "handleCompletedStorage"
    //    explicitly calls HandleCompletedRequest
    // barriers: none
    void StageResidencyUpdate(nvrhi::ICommandList* commandList,
                              const FrameSettings& settings) override;

    // Pre-traversal compute dispatch.  Runs before any traversal /
    // BLAS-build work consumes the streaming state.  Dispatches:
    //   * unload_groups (persistent CLAS allocator + unloads present)
    //   * update_scene  (residency-flip; HAS_ALPHA_TEST permutation)
    //   * StreamingResident::CommitActiveGroupsDelta  (active-list compaction copy)
    //   * mixed-cluster CLAS geometry-indices fill (alpha-mask scenes)
    //   * CLAS Implicit build
    //   * build_freegaps + setup_insertion + freegaps_insert (persistent CLAS
    //     allocator load-path), or the compaction defrag
    //
    // barriers: requires the sub-manager upload copies to have completed
    // (the renderer is expected to insert a transfer-write/compute-read
    // barrier between StageResidencyUpdate and ApplyResidencyUpdate).
    void ApplyResidencyUpdate(nvrhi::ICommandList* commandList) override;

    // Post-traversal compute dispatch.  Runs after traversal +
    // BLAS-build are done with last frame's CLAS data.  Dispatches:
    //   * age_filter        (resident-group age decay; gated on runAgeFilter)
    //   * load_groups       (persistent CLAS allocator: place new CLAS)
    //   * CLAS Move (old + new)
    //   * stream_dispatch_setup status branch  (status / max-sized counters)
    //
    // barriers: requires compute writes from ApplyResidencyUpdate + traversal to
    // be completed (the renderer inserts the barrier).
    void FinalizeResidency(nvrhi::ICommandList* commandList, bool runAgeFilter) override;

    // triggers request download
    // barriers: requires compute shader writes from FinalizeResidency to be completed
    void CaptureFrameRequests(nvrhi::ICommandList* commandList) override;

    // Fences this frame's streaming tasks against the frame's submission —
    // call after the frame's command list has been executed, never before.
    void SignalTasksSubmitted() override;

    // ---- public accessors

    void   GetStats(StreamingStats& stats) const override;

    // Fraction [0,1] of the streaming budgets in use — the max of geometry-pool
    // bytes and CLAS-pool BYTES occupancy (used + waste vs pool; deliberately
    // NOT the fragmentation-ratcheting maxSizedLeft metric — see the impl).
    float  GetLoadFactor() const override;

    // Requesting a report also arms the CLAS-sizes readback whose result the
    // NEXT report carries: it copies the whole per-group CLAS-sizes buffer and
    // sweeps the resident set on the CPU, so it only runs on frames that ask.
    bool GetResidencyReport(ResidencyReport& out) override;

    size_t GetClasSize(bool reserved) const;
    size_t GetBlasSize(bool reserved) const;
    size_t GetGeometrySize(bool reserved) const;
    size_t GetOperationsSize() const { return m_operationsSize + m_clasOperationsSize; }


    const StreamingConfig& GetStreamingConfig() const override { return m_config; }

    // ClusterLodResources getter overrides — forward base-class accessors to
    // StreamingResident, which owns the scene-global resident tables for the
    // streaming path. Preloaded still populates the base-class buffers directly.
    const RTXMGBuffer<uint64_t>& GetResidentClasAddressesBuffer() const override;
    const RTXMGBuffer<shaderio::ClusterAddress>& GetResidentClustersBuffer() const override;
    const RTXMGBuffer<shaderio::StreamingGroup>& GetResidentGroupsBuffer() const override;

    // traversal_run.hlsl load-emit bindings.
    nvrhi::IBuffer* GetStreamingShaderBuffer()         const override;
    nvrhi::IBuffer* GetStreamingLoadGroupsBuffer()     const override;

    // BLAS merging — age-filter buffers consumed by
    // traversal_blas_merging (forward to the resident / request sub-objects).
    nvrhi::IBuffer* GetActiveGroupsBuffer()            const override;
    nvrhi::IBuffer* GetGroupIDsBuffer()                const override;
    nvrhi::IBuffer* GetUnloadRequestBuffer()           const override;
    uint64_t        GetUnloadRequestRingBytes()        const override;
    uint32_t        GetActiveGroupsCount()             const override
    {
        return m_shaderData.resident.activeGroupsCount;
    }

    // BLAS caching.  GetMaxCachedBlasBuilds() sizes the BLAS pass's
    // build arrays at Init (caching builds append after the per-instance
    // builds); the override accessors feed its per-frame caching dispatches
    // (consumed only when useBlasCaching).
    uint32_t GetMaxCachedBlasBuilds() const override { return m_updates.GetMaxCachedBlasBuilds(); }

    uint32_t GetPatchCachedBlasCount() const override
    {
        return m_shaderData.update.patchCachedBlasCount;
    }
    uint32_t GetPatchCachedClustersCount() const override
    {
        return m_shaderData.update.patchCachedClustersCount;
    }
    nvrhi::IBuffer* GetGeometryPatchesBuffer() const override
    {
        return m_updates.GetGeometryPatchesBuffer();
    }
    uint64_t GetGeometryPatchesByteOffset() const override
    {
        // kInvalidTaskIndex (no update this frame) yields offset 0 — the BLAS
        // pass skips the caching dispatches when patchCachedBlasCount is 0.
        return m_updates.GetGeometryPatchesByteOffsetForTask(m_shaderData.update.taskIndex);
    }
    const std::vector<nvrhi::IBuffer*>* GetCachedBlasPoolBlocks() override;

private:
    // Streaming-side per-geometry struct.  Extends BaseGeometry so the
    // shared LoD-tree / flat-metadata buffers + bindless SRV handles come
    // from there; only the streaming-only extras live here.
    struct PersistentGeometry : BaseGeometry
    {
        uint32_t                        lodLevelsCount                                = 0;
        uint32_t                        lodLoadedGroupsCount[shaderio::kMaxLodLevels] = {};
        uint32_t                        lodGroupsCount[shaderio::kMaxLodLevels]       = {};
        // Index of the persistent low-detail group within this geometry's
        // groups (= sceneGeometry.lodLevels.back().groupOffset), cached so
        // ResetGeometryStreamingState doesn't need the geometries vector.
        uint32_t                        lastLodGroupOffset                            = 0;
        uint32_t                        cachedBlasUpdateFrame                         = 0;
        uint32_t                        cachedBlasLevel                               = shaderio::kTraversalInvalidLodLevel;
        // Host-only handle wrapping block location + size + allocator metadata,
        // so free() doesn't need a parallel size field.
        rtxmg::BufferSubAllocation      cachedBlasAllocation                          = {};
    };

    // Compute-shader handles (compiled by donut::engine::ShaderFactory).
    struct Shaders
    {
        nvrhi::ShaderHandle computeAgeFilterGroups;
        // stream_update_scene is compiled with HAS_ALPHA_TEST={0,1}; indexed by
        // m_hasAlphaMask at dispatch so alpha-mask-free scenes compile out the
        // mixed-cluster task-append (and its InterlockedAdd) entirely.
        nvrhi::ShaderHandle computeUpdateScene[2];
        nvrhi::ShaderHandle computeUpdateClasGeometryIndices;
        nvrhi::ShaderHandle computeSetup;

        // if usePersistentClasAllocator
        nvrhi::ShaderHandle computeAllocatorBuildFreeGaps;
        nvrhi::ShaderHandle computeAllocatorFreeGapsInsert;
        nvrhi::ShaderHandle computeAllocatorSetupInsertion;
        nvrhi::ShaderHandle computeAllocatorUnloadGroups;
        nvrhi::ShaderHandle computeAllocatorLoadGroups;
        // else
        nvrhi::ShaderHandle computeCompactionClasOld;
        nvrhi::ShaderHandle computeCompactionClasNew;
    };

    // ---- members ---------------------------------------------------------
    nvrhi::IDevice* m_device = nullptr;
    donut::engine::ShaderFactory* m_shaderFactory = nullptr;
    // Stashed at Init so initClas can hand the cached-BLAS BufferSubAllocator a
    // DescriptorTableManager (m_descriptorTable is only the nvrhi table handle).
    donut::engine::DescriptorTableManager* m_descriptorTableManager = nullptr;

    StreamingConfig m_config              = {};
    BakerConfig     m_bakerConfig         = {};   // cached at init; clusterGroupSize drives the CLAS budget math
    // Scene-wide observed cluster maxima; size initClas's CLAS scratch queries.
    uint32_t        m_maxClusterTriangles = 0;
    uint32_t        m_maxClusterVertices  = 0;
    bool            m_requiresClas        = false;
    bool            m_hasAlphaMask        = false;
    // LoD-metadata buffers + low-detail blobs, measured by InitGeometries.
    size_t          m_persistentGeometrySize = 0;
    size_t          GetPersistentDataBytes() const
    {
        return m_persistentGeometrySize + m_lowDetailClasGeometryIndices.GetVramBytes();
    }
    size_t          m_operationsSize      = 0;
    size_t          m_clasOperationsSize  = 0;
    size_t          m_blasSize            = 0;
    size_t          m_peakGeometrySize    = 0;
    uint32_t        m_peakFrameIndex      = ~0u;
    uint32_t        m_frameIndex          = 0;
    // Resident CLAS bytes per [geometryID][lodLevel]; see GetResidencyReport().
    std::vector<std::array<uint64_t, shaderio::kMaxLodLevels>> m_residentClasBytes;
    bool m_residentClasStatsRequested = false;
    // The CLAS-allocator status scalars (clasCompactionUsedSize /
    // clasAllocatedMaxSizedLeft) live GPU-side in
    // StreamingResident::m_residentPersistentBuffer, not here.
    StreamingStats  m_stats               = {};

    // Reused by FillStreamingGroupData to expand a compressed group before the
    // strip rewrite.  Grows to the largest group seen and is freed with the
    // object; both call sites run on the render thread.
    std::vector<uint8_t> m_decompressScratch;

    // Persistent scene data.
    std::vector<PersistentGeometry> m_persistentGeometries;

    // Every geometry's low-detail group blob, packed at 16-byte offsets behind
    // one raw SRV; each resident low-detail group addresses its blob by offset.
    RTXMGBuffer<uint8_t>            m_lowDetailGroupsData;
    donut::engine::DescriptorHandle m_lowDetailGroupsDataSRVHandle;
    // Per-triangle CLAS geometry-index/flags entries for the low-detail
    // clusters that need mixed-geometry-index encoding; empty when none do.
    RTXMGBuffer<nvrhi::rt::cluster::GeometryIndexAndFlags> m_lowDetailClasGeometryIndices;

    // Per-geometry data (groupInfos, lodLevels, groupData) cached at init.  The
    // spans inside GeometryView reference scene-owned storage; the caller
    // guarantees lifetime through to deinit.
    std::vector<GeometryView> m_geometries;

    // Aggregate per-frame scalar header.  StageResidencyUpdate mutates the host
    // snapshot then writeBuffer's it wholesale into m_shaderBuffer; sub-manager
    // readbacks source from that buffer at the matching offsetof().
    // The streaming shaders bind the aggregate at u7 and read the resident,
    // update, request and allocator sub-structures from it.
    shaderio::SceneStreaming              m_shaderData = {};
    RTXMGBuffer<shaderio::SceneStreaming> m_shaderBuffer;

    // Streaming pipeline: request -> upload -> update.
    StreamingTaskQueue m_requestsTaskQueue;
    StreamingTaskQueue m_storageTaskQueue;
    StreamingTaskQueue m_updatesTaskQueue;

    StreamingRequests  m_requests;
    StreamingResident  m_resident;
    StreamingAllocator m_clasAllocator;     // only valid if usePersistentClasAllocator
    StreamingStorage   m_storage;
    StreamingUpdates   m_updates;

    // Cached-BLAS pool (driven by useBlasCaching from FrameSettings).  Created
    // lazily on the first cached-BLAS allocation, where a failed reservation is
    // fatal — so use sites can assume it is valid once isInitialized().
    rtxmg::BufferSubAllocator m_cachedBlasAllocator;
    uint32_t                  m_cachedBlasAlignment = 4;
    // Scratch view rebuilt by GetCachedBlasPoolBlocks() each frame so the BLAS
    // pass can transition the pool's AS-storage blocks after its MOVE_OBJECTS.
    std::vector<nvrhi::IBuffer*> m_cachedBlasPoolBlocksView;

    // Persistent-residency CLAS resources.  The base class
    // (ClusterLodResourcesBase) owns m_clasLowDetailBlasBuffer — only the
    // CLAS-data buffer that backs it is owned here.
    RTXMGBuffer<uint8_t> m_clasLowDetailBuffer;
    size_t               m_clasLowDetailSize = 0;

    // CLAS scratch sizing, from getClusterOperationSizeInfo at init.
    size_t m_clasSingleMaxSize       = 0;
    size_t m_clasScratchNewClasSize  = 0;
    size_t m_clasScratchAlignment    = 0;
    nvrhi::rt::cluster::OperationClasBuildParams m_clasTriangleInput = {};

    Shaders   m_shaders;

    // ---- dispatch infrastructure --------------------------------------------
    // Unified streaming binding layout — covers allocator family,
    // stream_update_scene, and stream_age_groups, reused across every streaming
    // dispatch.  Slot map lives next to the createBindingLayout call in
    // InitShadersAndPipelines.
    nvrhi::BindingLayoutHandle  m_streamingBindingLayout;
    nvrhi::BindingSetHandle     m_streamingBindingSet;      // rebuilt per frame in UpdateBindings()

    // stream_dispatch_setup.hlsl is multi-branch (5 branches dispatched with different
    // push values); its layout adds PushConstants(b0, 4) on top of the
    // streaming layout.  The binding set adds a matching PushConstants entry.
    nvrhi::BindingLayoutHandle  m_setupBindingLayout;
    nvrhi::BindingSetHandle     m_setupBindingSet;

    // update_scene, allocator load/unload and compaction_old pair the unified
    // layout with a bindless heap layout for their ResourceDescriptorHeap[]
    // per-geometry / per-block lookups.
    nvrhi::BindingLayoutHandle  m_updateSceneBindlessLayout;
    // [HAS_ALPHA_TEST 0 or 1]. Pick at dispatch time on m_hasAlphaMask.
    nvrhi::ComputePipelineHandle m_updateScenePso[2];
    // Mixed-cluster geometry-indices fill — dispatched indirectly after
    // stream_update_scene when m_hasAlphaMask.
    nvrhi::ComputePipelineHandle m_updateClasGeometryIndicesPso;

    nvrhi::ComputePipelineHandle m_buildFreegapsPso;
    nvrhi::ComputePipelineHandle m_setupInsertionPso;
    nvrhi::ComputePipelineHandle m_freegapsInsertPso;
    nvrhi::ComputePipelineHandle m_unloadGroupsPso;
    nvrhi::ComputePipelineHandle m_loadGroupsPso;
    nvrhi::ComputePipelineHandle m_compactionOldPso;  // stream_compact_defrag_old.hlsl (compact allocator)
    nvrhi::ComputePipelineHandle m_compactionNewPso;  // stream_compact_append_new.hlsl (compact allocator)
    // Compaction: dedicated space1 raw view of the SceneStreaming aggregate
    // for the 64-bit moveClasSize InterlockedAdd64 (kept out of the unified
    // space0 layout so it never collides with the mutable bindless heap).
    nvrhi::BindingLayoutHandle   m_compactionRawLayout;
    nvrhi::BindingSetHandle      m_compactionRawSet;
    // GPU-persistent allocator scalars (StreamingResidentPersistent) in
    // their own space2 set, bound only to stream_dispatch_setup (the sole reader/writer).
    // Kept out of the host-uploaded SceneStreaming so the GPU's cursor/budget
    // survive frame-to-frame.
    nvrhi::BindingLayoutHandle   m_residentPersistentLayout;
    nvrhi::BindingSetHandle      m_residentPersistentSet;
    nvrhi::ComputePipelineHandle m_setupPso;      // stream_dispatch_setup.hlsl multi-branch (1x1x1 dispatches with different push)

    nvrhi::ComputePipelineHandle m_agefilterPso;

    // Fills unified-set slots that have no backing buffer in the current
    // configuration (e.g. u1 u_AllocatorMem / u4 u_ResidentGroupClasSizes in the
    // compaction path).  Only the PSOs that never dispatch there read them, so
    // the dummy is never actually sampled — it just keeps the set valid.
    RTXMGBuffer<uint32_t>       m_streamingDummyBuffer;

    // Per-frame CLAS buffers.  Transient build/move scratch is nvrhi-managed
    // (OperationDesc only takes scratchSizeInBytes, no IBuffer*).
    //
    //   m_clasIndirectArgsBuffer    — input args (IndirectTriangleClasArgs[])
    //   m_clasScratchBuffer         — Implicit-mode temporary CLAS data lands
    //                                 here (= outAccelerationStructuresBuffer
    //                                 in nvrhi::rt::cluster::OperationDesc;
    //                                 sized to m_clasScratchNewClasSize).
    nvrhi::BufferHandle m_clasIndirectArgsBuffer;
    nvrhi::BufferHandle m_clasScratchBuffer;

    // DispatchIndirectArguments for the alpha-mask geometry-indices pass:
    // stream_dispatch_setup's grid is copied here out of m_shaderBuffer, which cannot be
    // dispatched from directly — it is simultaneously the u7 UAV, and D3D12 has
    // no valid UnorderedAccess|IndirectArgument state.  Alpha-mask scenes only.
    nvrhi::BufferHandle m_clasGeometryIndicesDispatchBuffer;

    // Scene-wide bindless descriptor table.  Passed to init(); streaming
    // shaders' bindless lookups (geom.streamingGroupAddressesSRV/UAV +
    // patch.groupAddress.srvIndex) read through it.  Used for bindless
    // per-geometry and per-block views.
    nvrhi::DescriptorTableHandle m_descriptorTable;

    uint32_t            m_maxNewClustersPerFrame = 0;

    struct PendingSubmittedTask
    {
        enum class TaskQueue
        {
            Requests,
            Updates,
            Storage,
        };

        TaskQueue taskQueue = TaskQueue::Requests;
        uint32_t  taskIndex = kInvalidTaskIndex;
    };

    std::vector<PendingSubmittedTask> m_pendingSubmittedTasks;
    // -------------------------------------------------------------------------

    void InitGeometries(const std::vector<GeometryView>&        geometries,
                        const std::vector<ClusterLodInstance>&  instances,
                        donut::engine::DescriptorTableManager*  descriptorTable,
                        nvrhi::IDevice*                         device,
                        nvrhi::ICommandList*                    commandList);
    void ResetGeometryStreamingState(nvrhi::ICommandList* commandList);
    // Reset streaming state.  Waits the device idle, resets all task queues
    // and sub-managers, re-issues the persistent group-addresses upload, and
    // (if CLAS / BLAS caching is on) resets the cached-BLAS pool +
    // persistent CLAS allocator.
    void Reset(nvrhi::ICommandList* commandList);
    bool InitShadersAndPipelines(donut::engine::ShaderFactory* shaderFactory,
                                 nvrhi::IDevice*               device);
    void DeinitShadersAndPipelines();
    bool DebugClusterLodLoggingEnabled() const;
    void LogDebugRendererBeginFrame() const;
    void LogDebugUpdateApply(uint32_t taskIndex) const;
    void LogDebugResidentActive() const;
    void LogDebugRequestReadback(uint64_t requestFrame,
                                 uint32_t taskIndex,
                                 uint32_t loadCount,
                                 uint32_t unloadCount,
                                 const StreamingRequests::TaskInfo& request) const;
    void DebugDumpAllocatorFreelist(nvrhi::ICommandList* commandList,
                                    const shaderio::StreamingUpdate& update);
    void DebugReadbackMoveArgs(nvrhi::ICommandList* commandList,
                               RTXMGBuffer<uint64_t>& moveSrcBuffer,
                               RTXMGBuffer<uint64_t>& moveDstBuffer,
                               const shaderio::StreamingUpdate& update);

    // Rebuild m_streamingBindingSet for this frame — the patches sub-range
    // offset varies with the active update-task ring slot.  Called from
    // StageResidencyUpdate so the set is live before any dispatch.
    void UpdateBindings(nvrhi::ICommandList* commandList);

    // The four ApplyResidencyUpdate dispatch groups, in call order.  The
    // persistent/compaction allocator fork lives wholly inside the last.
    void UpdateSceneResidency(nvrhi::ICommandList*             commandList,
                              const shaderio::StreamingUpdate& update);
    void FillMixedClusterGeometryIndices(nvrhi::ICommandList*             commandList,
                                         const shaderio::StreamingUpdate& update);
    void BuildNewClas(nvrhi::ICommandList*             commandList,
                      const shaderio::StreamingUpdate& update);
    void PrepareClasAllocator(nvrhi::ICommandList*             commandList,
                              const shaderio::StreamingUpdate& update);

    // Host-side bookkeeping for the load/unload requests the GPU recorded last
    // frame.  Returns the update-task index, or kInvalidTaskIndex if there was
    // no work to do.
    uint32_t HandleCompletedRequest(nvrhi::ICommandList* commandList,
                                    const FrameSettings& settings,
                                    uint32_t             popRequestIndex);

    // What StageLoads accumulates that HandleCompletedRequest still needs after
    // the staging loop, for its debug log and its transfer accounting.
    struct LoadStageResult
    {
        uint32_t skippedResidentCount = 0;
        uint32_t clasBudgetForLoads   = 0;
        uint32_t futureGroups         = 0;
        uint64_t transferBytes        = 0;
    };

    void CheckRequestErrors(const StreamingRequests::TaskInfo& request,
                            uint64_t                           requestFrame,
                            uint32_t                           popRequestIndex) const;
    void UpdateClasBudgetStats(const StreamingRequests::TaskInfo& request,
                               uint64_t                           requestFrame);
    void StageUnloads(const StreamingRequests::TaskInfo& request,
                      StreamingUpdates::TaskInfo&        updateTask,
                      uint32_t                           unloadCount,
                      bool                               useBlasCaching);
    LoadStageResult StageLoads(const StreamingRequests::TaskInfo& request,
                               StreamingStorage::TaskInfo&        storageTask,
                               StreamingUpdates::TaskInfo&        updateTask,
                               uint32_t                           loadCount,
                               uint64_t                           requestFrame,
                               bool                               useBlasCaching);

    void HandleBlasCaching(StreamingUpdates::TaskInfo& updateTask,
                           const FrameSettings&        settings);
    bool AllocateCachedBlas(const PersistentGeometry&   geometry,
                            uint32_t                    lodClustersCount,
                            const FrameSettings&        settings,
                            rtxmg::BufferSubAllocation& subAllocation);
    bool InitClas(nvrhi::IDevice* device, nvrhi::ICommandList* commandList);

    // The persistent low-detail prefix: one BLAS arg per geometry and one CLAS
    // arg per cluster, plus the sizing the CLAS/BLAS builds need back.
    struct LowDetailClasArgs
    {
        std::vector<nvrhi::rt::cluster::nv::IndirectTriangleClasArgs> clasArgs;
        std::vector<nvrhi::rt::cluster::IndirectArgs>             blasArgs;
        uint32_t groupsCount           = 0;
        uint32_t clustersCount         = 0;
        uint32_t maxGroupClustersCount = 0;
        uint32_t maxTriangleCount      = 0;
        uint32_t maxVertexCount        = 0;
        uint32_t totalTriangleCount    = 0;
        uint32_t totalVertexCount      = 0;
    };

    void InitClasSizing(nvrhi::IDevice* device, uint32_t maxNewPerFrameClusters);
    void InitClasAllocator(nvrhi::IDevice* device, uint32_t clusterByteAlignment);
    LowDetailClasArgs BuildLowDetailClasArgs(nvrhi::IDevice*      device,
                                             nvrhi::ICommandList* commandList);
    void BuildLowDetailClas(nvrhi::IDevice*          device,
                            nvrhi::ICommandList*     commandList,
                            const LowDetailClasArgs& lowDetail);
    void BuildLowDetailBlas(nvrhi::IDevice*          device,
                            nvrhi::ICommandList*     commandList,
                            const LowDetailClasArgs& lowDetail);
    bool CreatePerFrameClasBuffers(nvrhi::IDevice* device,
                                   uint32_t        maxNewPerFrameClusters);

    void DeinitClas();
};

}  // namespace rtxmg


// Legacy global spelling used by demo code.
using ClusterLodStreaming    = rtxmg::ClusterLodStreaming;
