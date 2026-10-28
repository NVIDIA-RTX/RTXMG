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

#include "rtxmg/cluster_lod/blas_pass.h"
#include "rtxmg/cluster_lod/blas_build_params.h"
#include "rtxmg/cluster_lod/shader_registers.h"  // canonical binding slots
#include "rtxmg/utils/bindless_layout.h"

#include <donut/core/log.h>
#include <nvrhi/nvrhi.h>

#include <algorithm>
#include <vector>

using namespace donut::log;
using namespace nvrhi::rt;

namespace {
uint32_t GetThreadGroupCount(uint32_t itemCount, uint32_t itemsPerGroup)
{
    return (itemCount + itemsPerGroup - 1u) / itemsPerGroup;
}
} // anonymous namespace

// ---------------------------------------------------------------------------
// ClusterLodBlasPass::Init
// ---------------------------------------------------------------------------

void ClusterLodBlasPass::Init(
    const ClusterLodResources&   resources,
    const ClusterLodPass&        traversalPass,
    nvrhi::DescriptorTableHandle  descriptorTable,
    donut::engine::ShaderFactory* shaderFactory,
    nvrhi::IDevice*              device,
    bool                         debugClusterLod,
    uint32_t                     maxCachedBlasBuilds,
    uint64_t                     cachedBlasPoolBytes)
{
    m_numInstances             = resources.GetRenderInstanceCount();
    m_maxCachedBlasBuilds      = maxCachedBlasBuilds;
    // Cached BLASes append after the per-instance builds in the same build op, so
    // every per-build array and the result storage must reserve room for them.
    m_maxBlasBuilds            = m_numInstances + maxCachedBlasBuilds;
    m_geometriesBuffer         = &resources.GetShaderGeometriesBuffer();
    m_renderInstancesBuffer    = &resources.GetShaderRenderInstancesBuffer();
    m_residentClasAddrsBuffer  = &resources.GetResidentClasAddressesBuffer();
    m_descriptorTable          = descriptorTable;
    m_debugClusterLod          = debugClusterLod;

    if (m_numInstances == 0)
        return;

    // One CLAS-VA slot per emittable render cluster.
    const uint32_t maxBlasClasEntries = traversalPass.GetMaxRenderClusters();

    // ---- Allocate GPU buffers ------------------------------------------------

    // One IndirectArgs per build (every instance, plus the cached BLASes).  This
    // is the BLAS-build op's srcInfosArray, which the Vulkan validation layer
    // requires BUILD_INPUT_READ_ONLY creation usage on.
    {
        auto desc = GetGenericDesc(m_maxBlasBuilds,
                                   uint32_t(sizeof(cluster::IndirectArgs)),
                                   "ClusterLodBlasArgs")
                        .setIsAccelStructBuildInput(true);
        m_blasArgs.Create(desc, device);
    }

    // Shared CLAS VA pool.  Unlike m_blasArgs this needs no build-input usage:
    // the build args reach it by raw device address, not as a srcInfosArray.
    {
        auto desc = nvrhi::BufferDesc()
            .setByteSize(uint64_t(maxBlasClasEntries) * sizeof(nvrhi::GpuVirtualAddress))
            .setStructStride(sizeof(nvrhi::GpuVirtualAddress))
            .setDebugName("ClusterLodBlasClasAddrs")
            .setCanHaveUAVs(true)
            .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
            .setKeepInitialState(true);
        m_blasClasAddrs.Create(desc, device);
        m_blasClasAddrsNeedsClear = true;  // freshly (re)allocated — zero on first Execute
    }

    m_blasAddresses.Create(m_maxBlasBuilds, "ClusterLodBlasAddresses", device);
    m_blasSizes.Create(m_maxBlasBuilds,     "ClusterLodBlasSizes",     device);

    // Render-stats: per-geometry "seen this frame" flags so instance_assign_blas
    // counts each geometry's low-detail/cached BLAS triangles only once.
    m_perGeomSeen.Create(std::max(m_geometriesBuffer->GetNumElements(), 1u), "ClusterLodPerGeomSeen", device);

    // ---- Size the BLAS storage buffer ----------------------------------------
    m_blasParams = {};
    m_blasParams.maxArgCount            = m_maxBlasBuilds;
    m_blasParams.type                   = cluster::OperationType::BlasBuild;
    m_blasParams.mode                   = cluster::OperationMode::ImplicitDestinations;
    m_blasParams.flags                  = cluster::OperationFlags::None;
    m_blasParams.blas.maxClasPerBlasCount = maxBlasClasEntries;
    m_blasParams.blas.maxTotalClasCount   = maxBlasClasEntries;

    m_blasSizeInfo = device->getClusterOperationSizeInfo(m_blasParams);

    {
        nvrhi::BufferDesc blasDesc = nvrhi::BufferDesc()
            .setByteSize(m_blasSizeInfo.resultMaxSizeInBytes)
            .setDebugName("ClusterLodBlasBuffer")
            .setCanHaveUAVs(true)
            .setIsAccelStructStorage(true)
            .setInitialState(nvrhi::ResourceStates::AccelStructWrite)
            .setKeepInitialState(true);
        m_blasBuffer.Create(blasDesc, device);
    }

    // Deliberately NOT volatile: the diagnostic readbacks in Execute() close and
    // reopen the command list, which would retire a volatile CB's upload-heap
    // region.  A real GPU buffer survives that.
    {
        nvrhi::BufferDesc cbDesc;
        cbDesc.byteSize         = sizeof(BlasBuildParams);
        cbDesc.isConstantBuffer = true;
        cbDesc.initialState     = nvrhi::ResourceStates::ConstantBuffer;
        cbDesc.keepInitialState = true;
        cbDesc.debugName        = "ClusterLodBlasBuildParams";
        m_blasBuildParamsCB     = device->createBuffer(cbDesc);
    }

    // ---- BLAS caching — MOVE src/dst + scratch sizing ------------------------
    // One move per cached BLAS, copying it out of m_blasBuffer into its
    // persistent cached-pool allocation.
    if (m_maxCachedBlasBuilds > 0)
    {
        // Only SRC is the op's srcInfosArray, so only it needs build-input usage.
        {
            const uint32_t stride = uint32_t(sizeof(nvrhi::GpuVirtualAddress));

            m_cachedBlasAddressesSrc.Create(
                GetGenericDesc(m_maxCachedBlasBuilds, stride, "ClusterLodCachedBlasSrc")
                    .setIsAccelStructBuildInput(true),
                device);

            m_cachedBlasAddressesDst.Create(
                GetGenericDesc(m_maxCachedBlasBuilds, stride, "ClusterLodCachedBlasDst"),
                device);
        }

        m_blasMoveParams                = {};
        m_blasMoveParams.maxArgCount    = m_maxCachedBlasBuilds;
        m_blasMoveParams.type           = cluster::OperationType::Move;
        m_blasMoveParams.mode           = cluster::OperationMode::ExplicitDestinations;
        m_blasMoveParams.flags          = cluster::OperationFlags::NoOverlap;
        m_blasMoveParams.move.type      = cluster::OperationMoveType::BottomLevel;
        m_blasMoveParams.move.maxBytes  = uint32_t(cachedBlasPoolBytes);
        m_blasMoveSizeInfo              = device->getClusterOperationSizeInfo(m_blasMoveParams);
    }

    // ---- Create PSOs ---------------------------------------------------------
    CreatePipelines(shaderFactory, device);
}

uint64_t ClusterLodBlasPass::GetMetadataBytes() const
{
    return m_blasArgs.GetVramBytes()
         + m_blasClasAddrs.GetVramBytes()
         + m_blasAddresses.GetVramBytes()
         + m_blasSizes.GetVramBytes()
         + m_perGeomSeen.GetVramBytes()
         + m_cachedBlasAddressesSrc.GetVramBytes()
         + m_cachedBlasAddressesDst.GetVramBytes();
}

// ---------------------------------------------------------------------------
// ClusterLodBlasPass::CreatePipelines
// ---------------------------------------------------------------------------

void ClusterLodBlasPass::CreatePipelines(
    donut::engine::ShaderFactory* shaderFactory,
    nvrhi::IDevice*               device)
{
    // Stash for lazy PSO creation (instance_assign_blas permutations).
    m_shaderFactory = shaderFactory;
    m_device        = device;

    // This function recreates the binding layouts, so the lazily-built PSOs must
    // go too: each pins the previous m_updateBlasLayout and would fail nvrhi's
    // binding-set/layout match check.
    m_pipelines.computeInstanceAssignBlas = {};

    // --- blas_reserve_clusters layout ---
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.bindings = {
            nvrhi::BindingLayoutItem::ConstantBuffer(0),
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_COUNTERS),
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_INSTANCE_BUILD_INFOS),
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_BLAS_ARGS),
        };
        m_setupLayout = device->createBindingLayout(desc);
    }

    {
        auto shader = shaderFactory->CreateShader(
            "cluster_lod/blas_reserve_clusters.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);
        m_pipelines.computeBlasSetupInsertion = device->createComputePipeline(
            nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(m_setupLayout));
    }

    // --- blas_insert_clusters layout ---
    // No bindless layout: the resident-CLAS-address table is bound directly.
    // Everything this kernel only reads is an SRV, including instanceBuildInfos.
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.bindings = {
            nvrhi::BindingLayoutItem::ConstantBuffer(0),
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_BLAS_ARGS),
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_BLAS_CLAS_ADDRS),
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_GEOMETRIES),
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_RENDER_INSTANCES),
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_COUNTERS),
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_RENDER_CLUSTERS),
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_RESIDENT_CLAS_ADDRS),
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_INSTANCE_BUILD_INFOS),
        };
        m_insertLayout = device->createBindingLayout(desc);
    }

    {
        auto shader = shaderFactory->CreateShader(
            "cluster_lod/blas_insert_clusters.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);
        m_pipelines.computeBlasInsertClusters = device->createComputePipeline(
            nvrhi::ComputePipelineDesc()
                .setComputeShader(shader)
                .addBindingLayout(m_insertLayout));
    }

    // --- instance_assign_blas layout ---
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.bindings = {
            nvrhi::BindingLayoutItem::ConstantBuffer(0),
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_GEOMETRIES),  // cached-VA read + render-stats lowDetailTriangles
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_RENDER_INSTANCES),  // render-stats geometryID
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_INSTANCE_BUILD_INFOS),
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_BLAS_ADDRESSES),
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_PER_INSTANCE_TRIANGLES),  // render-stats
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_COUNTERS),  // render-stats atomics
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_INSTANCE_BLAS_ADDRS),
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_PER_GEOM_SEEN),  // render-stats unique dedup
        };
        m_updateBlasLayout = device->createBindingLayout(desc);
    }

    // instance_assign_blas PSOs are built lazily in GetInstanceAssignBlasPipeline().

    // --- BLAS caching pipelines ----------------------------------------------
    if (m_maxCachedBlasBuilds > 0)
    {
        // Vulkan needs this layout-compatible with the renderer's global bindless
        // layout, so use the shared canonical desc.
        m_cachingBindlessLayout = device->createBindlessLayout(rtxmg::MakeGlobalBindlessLayoutDesc());

        std::vector<donut::engine::ShaderMacro> cacheMacro = { { "USE_BLAS_CACHING", "1" } };

        // blas_cache_gather_clusters layout
        {
            nvrhi::BindingLayoutDesc desc;
            desc.visibility = nvrhi::ShaderType::Compute;
            desc.bindings = {
                nvrhi::BindingLayoutItem::ConstantBuffer(0),
                nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_GEOMETRIES),
                nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_RESIDENT_CLAS_ADDRS),
                nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_GEOMETRY_PATCHES),
                nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_COUNTERS),
                nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_BLAS_ARGS),
                nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_GEOMETRY_BUILD_INFOS),
                nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_BLAS_CLAS_ADDRS),
            };
            m_cachingBuildLayout = device->createBindingLayout(desc);

            auto shader = shaderFactory->CreateShader(
                "cluster_lod/blas_cache_gather_clusters.hlsl", "main", &cacheMacro, nvrhi::ShaderType::Compute);
            m_pipelines.computeBlasCachingSetupBuild = device->createComputePipeline(
                nvrhi::ComputePipelineDesc()
                    .setComputeShader(shader)
                    .addBindingLayout(m_cachingBuildLayout)
                    .addBindingLayout(m_cachingBindlessLayout));
        }

        // blas_cache_stage_move layout
        {
            nvrhi::BindingLayoutDesc desc;
            desc.visibility = nvrhi::ShaderType::Compute;
            desc.bindings = {
                nvrhi::BindingLayoutItem::ConstantBuffer(0),
                nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_BLAS_ADDRESSES),
                nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_GEOMETRY_PATCHES),
                nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_GEOMETRY_BUILD_INFOS),
                nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_COUNTERS),
                nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_CACHED_BLAS_SRC),
                nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_CACHED_BLAS_DST),
            };
            m_cachingCopyLayout = device->createBindingLayout(desc);

            auto shader = shaderFactory->CreateShader(
                "cluster_lod/blas_cache_stage_move.hlsl", "main", &cacheMacro, nvrhi::ShaderType::Compute);
            m_pipelines.computeBlasCachingSetupCopy = device->createComputePipeline(
                nvrhi::ComputePipelineDesc()
                    .setComputeShader(shader)
                    .addBindingLayout(m_cachingCopyLayout));
        }
    }
}

// ---------------------------------------------------------------------------
// ClusterLodBlasPass::GetInstanceAssignBlasPipeline
// Builds only the (sharing, caching) permutation actually dispatched this run;
// shaders.cfg must declare all of them.
// ---------------------------------------------------------------------------
nvrhi::ComputePipelineHandle ClusterLodBlasPass::GetInstanceAssignBlasPipeline(bool useBlasSharing,
                                                                               bool useBlasCaching)
{
    const InstanceAssignBlasPermutation perm(useBlasSharing, useBlasCaching);
    const uint32_t idx = perm.Index();
    if (!m_pipelines.computeInstanceAssignBlas[idx])
    {
        std::vector<donut::engine::ShaderMacro> defines = {
            { "USE_BLAS_SHARING", useBlasSharing ? "1" : "0" },
            { "USE_BLAS_CACHING", useBlasCaching ? "1" : "0" } };
        auto shader = m_shaderFactory->CreateShader(
            "cluster_lod/instance_assign_blas.hlsl", "main", &defines, nvrhi::ShaderType::Compute);
        m_pipelines.computeInstanceAssignBlas[idx] = m_device->createComputePipeline(
            nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(m_updateBlasLayout));
    }
    return m_pipelines.computeInstanceAssignBlas[idx];
}

void ClusterLodBlasPass::DebugReadbackTraversalOutput(nvrhi::ICommandList* commandList,
                                                      ClusterLodPass& traversalPass)
{
    if (!m_debugClusterLod)
        return;

    const std::vector<shaderio::SceneBuildingCounters> counters =
        traversalPass.GetCountersTyped().Download(commandList);
    if (counters.empty())
        return;

    const shaderio::SceneBuildingCounters& c = counters[0];
    donut::log::info("ClusterLod traversal frame=%llu nodes=%u groups=%u rawClusters=%u renderedClusters=%u sharingProviders=%u sharingConsumers=%u",
                     (unsigned long long)m_frameCount++,
                     c.traversalNodeWriteCounter,
                     c.traversalGroupWriteCounter,
                     c.renderClusterCounter,
                     c.numRenderedClusters,
                     c.numSharingProviders,
                     c.numSharingConsumers);
}

// ---------------------------------------------------------------------------
// ClusterLodBlasPass::Execute
// ---------------------------------------------------------------------------

void ClusterLodBlasPass::Execute(
    nvrhi::IDevice*       device,
    nvrhi::ICommandList*  commandList,
    ClusterLodPass& traversalPass,
    const ClusterLodBlasPassParams& params)
{
    if (m_numInstances == 0)
        return;

    nvrhi::IBuffer* countersBuffer         = traversalPass.GetCountersBuffer();
    nvrhi::IBuffer* instanceBuildInfos     = traversalPass.GetInstanceBuildInfosBuffer();
    nvrhi::IBuffer* renderClusters         = traversalPass.GetRenderClusterInfosBuffer();

    // Cleared unconditionally so the render-stats dedup flags stay deterministic
    // regardless of the shader-side gate.
    commandList->clearBufferUInt(m_perGeomSeen.GetBuffer(), 0u);

    // Zero the CLAS-VA pool once after (re)Init, before any gather writes: a
    // cached-BLAS build must not reference a stale VA left in recycled heap memory.
    if (m_blasClasAddrsNeedsClear)
    {
        commandList->clearBufferUInt(m_blasClasAddrs.GetBuffer(), 0u);
        m_blasClasAddrsNeedsClear = false;
    }

    DebugReadbackTraversalOutput(commandList, traversalPass);

    const bool useCaching = params.useBlasCaching && m_maxCachedBlasBuilds > 0
                          && params.patchCachedBlasCount > 0u;

    BlasBuildParams cbParams{};
    cbParams.blasClasAddressesBaseVA = m_blasClasAddrs.GetGpuVirtualAddress();
    cbParams.numRenderInstances      = m_numInstances;
    cbParams.patchCachedBlasCount    = useCaching ? params.patchCachedBlasCount : 0u;
    nvrhi::IBuffer* cbuf = m_blasBuildParamsCB;
    commandList->writeBuffer(cbuf, &cbParams, sizeof(cbParams));

    auto uavBarrier = [&](nvrhi::IBuffer* buf) {
        commandList->setBufferState(buf, nvrhi::ResourceStates::UnorderedAccess);
        commandList->commitBarriers();
    };

    // =========================================================================
    // Blas Build Preparation
    // =========================================================================

    // (1) blas_reserve_clusters — preps per-blas CLAS reference list starting
    //     positions and resets per-blas CLAS counters.
    {
        nvrhi::BindingSetDesc bsDesc;
        bsDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, cbuf),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_COUNTERS, countersBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_INSTANCE_BUILD_INFOS, instanceBuildInfos),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_BLAS_ARGS, m_blasArgs.GetBuffer()),
        };
        nvrhi::BindingSetHandle bs = device->createBindingSet(bsDesc, m_setupLayout);

        nvrhi::ComputeState state;
        state.pipeline = m_pipelines.computeBlasSetupInsertion;
        state.bindings = { bs };
        commandList->setComputeState(state);
        commandList->dispatch(GetThreadGroupCount(m_numInstances, shaderio::kBlasSetupInsertionThreads), 1, 1);
    }

    uavBarrier(m_blasArgs.GetBuffer());
    uavBarrier(countersBuffer);

    // (2) blas_insert_clusters — fills clusters from the unsorted render list into
    //     per-blas CLAS reference lists.  Dispatch args come from
    //     traversal_setup[BlasInsertion].
    {
        nvrhi::BindingSetDesc bsDesc;
        bsDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, cbuf),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_BLAS_ARGS, m_blasArgs.GetBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_BLAS_CLAS_ADDRS, m_blasClasAddrs.GetBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_GEOMETRIES, m_geometriesBuffer->GetBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_RENDER_INSTANCES, m_renderInstancesBuffer->GetBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_COUNTERS, countersBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_RENDER_CLUSTERS, renderClusters),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_RESIDENT_CLAS_ADDRS, m_residentClasAddrsBuffer->GetBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_INSTANCE_BUILD_INFOS, instanceBuildInfos),
        };
        nvrhi::BindingSetHandle bs     = device->createBindingSet(bsDesc, m_insertLayout);

        uint32_t offset = static_cast<uint32_t>(
            offsetof(shaderio::SceneBuildingCounters, indirectDispatchBlasInsertionX));

        nvrhi::ComputeState state;
        state.pipeline = m_pipelines.computeBlasInsertClusters;
        state.bindings = { bs };
        state.indirectParams = countersBuffer;
        commandList->setComputeState(state);
        commandList->dispatchIndirect(offset);
    }

    // (3) Caching seed: gather the cached LoD level's CLAS refs into the shared
    //     pool and append IndirectArgs slots, so the build below covers them.
    if (useCaching)
    {
        nvrhi::BindingSetDesc bsDesc;
        bsDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, cbuf),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_GEOMETRIES, m_geometriesBuffer->GetBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_RESIDENT_CLAS_ADDRS, m_residentClasAddrsBuffer->GetBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_GEOMETRY_PATCHES, params.geometryPatchesBuffer,
                                                        nvrhi::Format::UNKNOWN,
                                                        nvrhi::BufferRange(params.geometryPatchesByteOffset, ~0ull)),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_COUNTERS, countersBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_BLAS_ARGS, m_blasArgs.GetBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_GEOMETRY_BUILD_INFOS, traversalPass.GetGeometryBuildInfosBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_BLAS_CLAS_ADDRS, m_blasClasAddrs.GetBuffer()),
        };
        nvrhi::BindingSetHandle bs = device->createBindingSet(bsDesc, m_cachingBuildLayout);

        nvrhi::ComputeState state;
        state.pipeline = m_pipelines.computeBlasCachingSetupBuild;
        state.bindings = { bs, m_descriptorTable };
        commandList->setComputeState(state);
        commandList->dispatch(params.patchCachedBlasCount, 1, 1);  // one thread group per cached BLAS
    }

    // (4) A UAV barrier is enough here: executeMultiIndirectClusterOperation
    //     transitions the args / CLAS-ref arrays into ShaderResource itself.
    uavBarrier(m_blasArgs.GetBuffer());
    uavBarrier(m_blasClasAddrs.GetBuffer());

    // =========================================================================
    // Blas Build
    // =========================================================================
    {
        uint32_t countOffset = static_cast<uint32_t>(
            offsetof(shaderio::SceneBuildingCounters, blasBuildCounter));

        cluster::OperationDesc buildDesc = {};
        buildDesc.params                         = m_blasParams;
        buildDesc.scratchSizeInBytes             = m_blasSizeInfo.scratchSizeInBytes;
        buildDesc.inIndirectArgCountBuffer       = countersBuffer;
        buildDesc.inIndirectArgCountOffsetInBytes = countOffset;
        buildDesc.inIndirectArgsBuffer           = m_blasArgs.GetBuffer();
        buildDesc.inOutAddressesBuffer           = m_blasAddresses.GetBuffer();
        buildDesc.outSizesBuffer                 = m_blasSizes.GetBuffer();
        buildDesc.outAccelerationStructuresBuffer = m_blasBuffer.GetBuffer();

        commandList->executeMultiIndirectClusterOperation(buildDesc);
    }

    // =========================================================================
    // Blas Copy — MOVE each freshly-built cached BLAS out of m_blasBuffer into
    // its persistent cached-pool allocation.
    // =========================================================================
    if (useCaching)
    {
        // (a) setup_copy — stage the MOVE src/dst address arrays.
        {
            nvrhi::BindingSetDesc bsDesc;
            bsDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, cbuf),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_BLAS_ADDRESSES, m_blasAddresses.GetBuffer()),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_GEOMETRY_PATCHES, params.geometryPatchesBuffer,
                                                            nvrhi::Format::UNKNOWN,
                                                            nvrhi::BufferRange(params.geometryPatchesByteOffset, ~0ull)),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_GEOMETRY_BUILD_INFOS, traversalPass.GetGeometryBuildInfosBuffer()),
                nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_COUNTERS, countersBuffer),
                nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_CACHED_BLAS_SRC, m_cachedBlasAddressesSrc.GetBuffer()),
                nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_CACHED_BLAS_DST, m_cachedBlasAddressesDst.GetBuffer()),
            };
            nvrhi::BindingSetHandle bs = device->createBindingSet(bsDesc, m_cachingCopyLayout);

            nvrhi::ComputeState state;
            state.pipeline = m_pipelines.computeBlasCachingSetupCopy;
            state.bindings = { bs };
            commandList->setComputeState(state);
            commandList->dispatch(GetThreadGroupCount(params.patchCachedBlasCount,
                                                      shaderio::kBlasCachingSetupCopyThreads), 1, 1);
        }

        uavBarrier(m_cachedBlasAddressesSrc.GetBuffer());
        uavBarrier(m_cachedBlasAddressesDst.GetBuffer());
        uavBarrier(countersBuffer);

        // (b) MOVE_OBJECTS / ExplicitDestinations — count = cachedBlasCopyCounter.
        {
            cluster::OperationDesc moveDesc = {};
            moveDesc.params                          = m_blasMoveParams;
            moveDesc.scratchSizeInBytes              = m_blasMoveSizeInfo.scratchSizeInBytes;
            moveDesc.inIndirectArgCountBuffer        = countersBuffer;
            moveDesc.inIndirectArgCountOffsetInBytes =
                static_cast<uint32_t>(offsetof(shaderio::SceneBuildingCounters, cachedBlasCopyCounter));
            moveDesc.inIndirectArgsBuffer            = m_cachedBlasAddressesSrc.GetBuffer();  // src VAs
            moveDesc.inOutAddressesBuffer            = m_cachedBlasAddressesDst.GetBuffer();  // dst VAs
            commandList->executeMultiIndirectClusterOperation(moveDesc);
        }

        // (c) nvrhi can't track the move's raw-VA destination, so transition the
        //     pool blocks explicitly for the TLAS build that reads them.
        if (params.cachedBlasPoolBlocks)
        {
            for (nvrhi::IBuffer* block : *params.cachedBlasPoolBlocks)
            {
                if (block)
                    commandList->setBufferState(block, nvrhi::ResourceStates::AccelStructBuildBlas);
            }
            commandList->commitBarriers();
        }
    }

    // =========================================================================
    // Tlas Preparation — instance_assign_blas replaces traversal_init's
    // lowDetailBlasAddress seed with the freshly-built BLAS VA, for instances
    // that actually got a build (clusterReferencesCount > 0).
    // =========================================================================
    {
        nvrhi::IBuffer* instanceBlasAddrs  = traversalPass.GetInstanceBlasAddrsBuffer();

        nvrhi::BindingSetDesc bsDesc;
        bsDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, cbuf),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_GEOMETRIES, m_geometriesBuffer->GetBuffer()),  // cached-VA read + render-stats lowDetailTriangles
            nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_RENDER_INSTANCES, m_renderInstancesBuffer->GetBuffer()),  // render-stats geometryID
            nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_INSTANCE_BUILD_INFOS, instanceBuildInfos),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_BLAS_ADDRESSES, m_blasAddresses.GetBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_PER_INSTANCE_TRIANGLES, traversalPass.GetPerInstanceTrianglesBuffer()), // render-stats
            nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_COUNTERS, traversalPass.GetCountersBuffer()),  // render-stats atomics
            nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_INSTANCE_BLAS_ADDRS, instanceBlasAddrs),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_PER_GEOM_SEEN, m_perGeomSeen.GetBuffer()),  // render-stats unique dedup
        };
        nvrhi::BindingSetHandle bs = device->createBindingSet(bsDesc, m_updateBlasLayout);

        nvrhi::ComputeState state;
        state.pipeline = GetInstanceAssignBlasPipeline(params.useBlasSharing, params.useBlasCaching);
        state.bindings = { bs };
        commandList->setComputeState(state);
        commandList->dispatch(GetThreadGroupCount(m_numInstances, shaderio::kInstancesAssignBlasThreads), 1, 1);

        uavBarrier(instanceBlasAddrs);
    }

}
