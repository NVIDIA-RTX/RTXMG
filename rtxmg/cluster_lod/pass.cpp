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

#include "rtxmg/cluster_lod/pass.h"

#include <algorithm>
#include <cassert>
#include <donut/core/log.h>
#include <nvrhi/nvrhi.h>
#include <nvrhi/utils.h>
#include <string>
#include <vector>

#include "rtxmg/cluster_lod/shader_registers.h"  // canonical binding slots
#include "rtxmg/hiz/hiz_buffer_constants.h"  // HIZ_MAX_LODS
#include "rtxmg/hiz/zbuffer.h"               // ZBuffer (inline HiZ accessors)
#include "rtxmg/utils/bindless_layout.h"

using namespace donut::log;

// ---------------------------------------------------------------------------
// ClusterLodPass::Init
// ---------------------------------------------------------------------------

void ClusterLodPass::Init(
    const ClusterLodResources&    resources,
    nvrhi::DescriptorTableHandle  descriptorTable,
    donut::engine::ShaderFactory* shaderFactory,
    nvrhi::IDevice*               device,
    bool                          debugClusterLod,
    uint32_t                      renderClusterBits)
{
    renderClusterBits  = std::clamp(renderClusterBits, kMinRenderClusterBits, kMaxRenderClusterBits);
    m_maxRenderClusters = 1u << renderClusterBits;
    m_geometriesBuffer      = &resources.GetShaderGeometriesBuffer();
    m_renderInstancesBuffer = &resources.GetShaderRenderInstancesBuffer();
    m_residentGroupsBuffer  = &resources.GetResidentGroupsBuffer();
    m_numInstances          = resources.GetRenderInstanceCount();
    m_numGeometries         = m_geometriesBuffer->GetNumElements();  // BLAS sharing
    m_maxNodeTreeDepth      = resources.GetMaxNodeTreeDepth();
    m_descriptorTable       = descriptorTable;
    m_debugClusterLod       = debugClusterLod;

    // traversal_run.hlsl load-emit bindings.  Preload mode has no streaming
    // buffers, so bind dummies of the right stride (D3D12 validates a
    // StructuredBuffer view's stride against the shader's element type); the
    // shader's load-emit branch never fires there.
    const IClusterLodStreamingHooks* streaming = resources.GetStreamingHooks();
    m_streamingShaderBuffer     = streaming ? streaming->GetStreamingShaderBuffer()     : nullptr;
    m_streamingLoadGroupsBuffer = streaming ? streaming->GetStreamingLoadGroupsBuffer() : nullptr;
    if (!m_streamingShaderBuffer)
    {
        nvrhi::BufferDesc d = nvrhi::BufferDesc()
            .setByteSize(sizeof(shaderio::SceneStreaming))
            .setStructStride(sizeof(shaderio::SceneStreaming))
            .setCanHaveUAVs(true)
            .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
            .setKeepInitialState(true)
            .setDebugName("ClusterLodPassStreamingDummyAggregate");
        m_streamingDummyAggregate = device->createBuffer(d);
        m_streamingShaderBuffer   = m_streamingDummyAggregate;
    }
    if (!m_streamingLoadGroupsBuffer)
    {
        constexpr uint64_t kDummyLoadGroupsBytes = 64 * 1024;
        nvrhi::BufferDesc d = nvrhi::BufferDesc()
            .setByteSize(kDummyLoadGroupsBytes)
            .setStructStride(8)  // uint2
            .setCanHaveUAVs(true)
            .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
            .setKeepInitialState(true)
            .setDebugName("ClusterLodPassStreamingDummyLoadGroups");
        m_streamingDummyLoadGroups = device->createBuffer(d);
        m_streamingLoadGroupsBuffer     = m_streamingDummyLoadGroups;
    }

    // traversal_blas_merging's age-filter bindings, on the same shared layout, so
    // preload needs the same right-stride dummies (merging is never dispatched there).
    m_activeGroupsBuffer  = streaming ? streaming->GetActiveGroupsBuffer()     : nullptr;
    m_groupIDsBuffer      = streaming ? streaming->GetGroupIDsBuffer()         : nullptr;
    m_unloadRequestBuffer = streaming ? streaming->GetUnloadRequestBuffer()    : nullptr;
    m_unloadRingBytes     = streaming ? streaming->GetUnloadRequestRingBytes() : 0u;
    {
        auto makeDummy = [&](uint32_t stride, const char* name) {
            return device->createBuffer(nvrhi::BufferDesc()
                .setByteSize(stride)
                .setStructStride(stride)
                .setCanHaveUAVs(true)
                .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
                .setKeepInitialState(true)
                .setDebugName(name));
        };
        if (!m_activeGroupsBuffer)
        {
            m_dummyActiveGroups  = makeDummy(4, "ClusterLodPassDummyActiveGroups");
            m_activeGroupsBuffer = m_dummyActiveGroups;
        }
        if (!m_groupIDsBuffer)
        {
            m_dummyGroupIDs  = makeDummy(4, "ClusterLodPassDummyGroupIDs");
            m_groupIDsBuffer = m_dummyGroupIDs;
        }
        if (!m_unloadRequestBuffer)
        {
            m_dummyUnloadRequest  = makeDummy(8, "ClusterLodPassDummyUnloadRequest");  // uint2
            m_unloadRequestBuffer = m_dummyUnloadRequest;
            m_unloadRingBytes     = 0u;
        }
    }

    // ---- Allocate GPU buffers ------------------------------------------------

    // Counters double as the dispatchIndirect argument source.
    {
        nvrhi::BufferDesc desc = nvrhi::BufferDesc()
            .setByteSize(sizeof(shaderio::SceneBuildingCounters))
            .setStructStride(sizeof(shaderio::SceneBuildingCounters))
            .setDebugName("ClusterLodBuildCounters")
            .setCanHaveUAVs(true)
            .setIsDrawIndirectArgs(true)
            // srcInfosCount source for the cluster BLAS-build/move ops; the Vulkan
            // validation layer requires BUILD_INPUT_READ_ONLY creation usage on it.
            .setIsAccelStructBuildInput(true)
            .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
            .setKeepInitialState(true);
        m_counters.Create(desc, device);
    }

    m_instanceBuildInfos.Create(std::max(m_numInstances, 1u), "ClusterLodInstanceBuildInfos", device);

    // Per-instance BLAS addresses — pre-filled by traversal_init, patched by instance_assign_blas.
    m_instanceBlasAddrs.Create(std::max(m_numInstances, 1u), "ClusterLodInstanceBlasAddrs", device);

    // Allocated even when merging is off, so the shared BindingSet stays valid.
    m_instanceVisibility.Create(std::max(m_numInstances, 1u), "ClusterLodInstanceVisibility", device);

    // Node queue holds interior-node traversal tasks; group queue holds leaf
    // groups whose clusters are resolved by traversal_run_groups.
    m_traversalNodeQ.Create(kMaxTraversalInfos, "ClusterLodTraversalNodeQ", device);
    m_traversalGroupQ.Create(kMaxTraversalInfos, "ClusterLodTraversalGroupQ", device);

    m_renderClusterInfos.Create(m_maxRenderClusters, "ClusterLodRenderClusterInfos", device);

    // Render-stats: the unique-cluster dedup flags are indexed by clusterID, so
    // size them to the resident-cluster table rather than the render list.
    m_perInstanceTriangles.Create(std::max(m_numInstances, 1u), "ClusterLodPerInstanceTriangles", device);
    {
        const uint32_t residentClusterCount =
            std::max(resources.GetResidentClustersBuffer().GetNumElements(), 1u);
        m_uniqueSeenClusters.Create(residentClusterCount, "ClusterLodUniqueSeenClusters", device);
    }

    // Allocated even when sharing is off, so the shared BindingSet stays valid.
    m_geometryBuildInfos.Create(std::max(m_numGeometries, 1u), "ClusterLodGeometryBuildInfos", device);
    m_geometryHistograms.Create(std::max(m_numGeometries, 1u), "ClusterLodGeometryHistograms", device);

    // Volatile: writeBuffer() spills into the upload heap; three versions cover
    // the frames in flight without aliasing.
    m_constantsCB = device->createBuffer(
        nvrhi::utils::CreateVolatileConstantBufferDesc(
            sizeof(shaderio::SceneBuildingConstants),
            "ClusterLodBuildConstants",
            /*maxVersions=*/3));

    m_maxClustersPerGroup = resources.GetMaxClustersPerGroup();

    // ---- Create shader pipelines ---------------------------------------------
    CreateBindingLayout(device);
    CreatePipelines(shaderFactory, device);
}

uint64_t ClusterLodPass::GetMetadataBytes() const
{
    return m_counters.GetVramBytes()
         + m_instanceBuildInfos.GetVramBytes()
         + m_instanceBlasAddrs.GetVramBytes()
         + m_traversalNodeQ.GetVramBytes()
         + m_traversalGroupQ.GetVramBytes()
         + m_renderClusterInfos.GetVramBytes()
         + m_geometryBuildInfos.GetVramBytes()
         + m_geometryHistograms.GetVramBytes()
         + m_instanceVisibility.GetVramBytes()
         + m_perInstanceTriangles.GetVramBytes()
         + m_uniqueSeenClusters.GetVramBytes();
}

// ---------------------------------------------------------------------------
// ClusterLodPass::CreateBindingLayout
// ---------------------------------------------------------------------------

void ClusterLodPass::CreateBindingLayout(nvrhi::IDevice* device)
{
    // Slots come from shader_registers.h, shared with traversal_common.hlsli.
    nvrhi::BindingLayoutDesc desc;
    desc.visibility = nvrhi::ShaderType::Compute;
    // Vulkan: registerSpace == descriptor set.  nvrhi requires the flag to match
    // across every layout of a pipeline, and the culling PSOs add space-1 HiZ.
    desc.registerSpaceIsDescriptorSet = true;
    desc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),      // b0
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_COUNTERS),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_TRAVERSAL_NODE_Q),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_TRAVERSAL_GROUP_Q),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_RENDER_CLUSTERS),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_INSTANCE_BUILD_INFOS),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_INSTANCE_BLAS_ADDRS),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_RESIDENT_GROUPS),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_STREAMING),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_LOAD_GEOMETRY_GROUPS),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_GEOMETRY_BUILD_INFOS),   // BLAS sharing
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_GEOMETRY_HISTOGRAMS),    // BLAS sharing
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_INSTANCE_VISIBILITY),    // BLAS merging
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_PER_INSTANCE_TRIANGLES), // render-stats
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_UNIQUE_SEEN_CLUSTERS),   // render-stats
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_ACTIVE_GROUPS),          // whole; shader adds persistentGroupsCount
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_GROUP_IDS),              // BLAS merging age filter
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(CLOD_U_UNLOAD_GEOMETRY_GROUPS), // BLAS merging age filter
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_GEOMETRIES),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(CLOD_T_RENDER_INSTANCES),
        nvrhi::BindingLayoutItem::PushConstants(1, sizeof(uint32_t)), // b1 traversal_setup mode
    };
    m_bindingLayout = device->createBindingLayout(desc);

    // Bindless layout for ResourceDescriptorHeap (SM 6.6 bindless SRV/UAV/CBV).
    // Must be layout-compatible with the renderer's global bindless layout on
    // Vulkan (same capacity + visibility) — use the shared canonical desc.
    m_bindlessLayout = device->createBindlessLayout(rtxmg::MakeGlobalBindlessLayoutDesc());

    // HiZ occlusion layout (space 1) — matches t_ClusterLodHiZ[HIZ_MAX_LODS] +
    // s_ClusterLodHiZSampler in the CLUSTER_LOD_HIZ_OCCLUSION shader variants.
    {
        nvrhi::BindingLayoutDesc hizDesc;
        hizDesc.visibility = nvrhi::ShaderType::Compute;
        hizDesc.registerSpace = 1;
        hizDesc.registerSpaceIsDescriptorSet = true; // Vulkan: space1 -> set 1 (ignored on D3D12)
        hizDesc.bindings = {
            nvrhi::BindingLayoutItem::Texture_SRV(0).setSize(HIZ_MAX_LODS), // t0[9] space1
            nvrhi::BindingLayoutItem::Sampler(0),                          // s0     space1
        };
        m_hizBindingLayout = device->createBindingLayout(hizDesc);

        m_hizSampler = device->createSampler(nvrhi::SamplerDesc()
            .setAllFilters(false)                              // point — HiZ is gathered, not filtered
            .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp));
    }
}


// ---------------------------------------------------------------------------
// ClusterLodPass::CreatePipelines
// ---------------------------------------------------------------------------

void ClusterLodPass::CreatePipelines(
    donut::engine::ShaderFactory* shaderFactory,
    nvrhi::IDevice*               device)
{
    auto loadCS = [&](const char* name) -> nvrhi::ShaderHandle {
        return shaderFactory->CreateShader(
            name, "main", nullptr, nvrhi::ShaderType::Compute);
    };
    // BLAS-sharing shaders ship as a USE_BLAS_SHARING=1 × USE_BLAS_CACHING={0,1}
    // permutation; the matching entry must be declared in shaders.cfg.
    auto loadCSSharing = [&](const char* name, bool caching) -> nvrhi::ShaderHandle {
        std::vector<donut::engine::ShaderMacro> defines = {
            { "USE_BLAS_SHARING", "1" },
            { "USE_BLAS_CACHING", caching ? "1" : "0" } };
        return shaderFactory->CreateShader(
            name, "main", &defines, nvrhi::ShaderType::Compute);
    };

    nvrhi::ComputePipelineDesc psoDesc;
    psoDesc.bindingLayouts = { m_bindingLayout, m_bindlessLayout };

    // HiZ-occlusion variant: adds the space-1 HiZ layout.
    nvrhi::ComputePipelineDesc psoDescHiz = psoDesc;
    psoDescHiz.bindingLayouts = { m_bindingLayout, m_bindlessLayout, m_hizBindingLayout };
    auto loadCSHiz = [&](const char* name, std::vector<donut::engine::ShaderMacro> defines)
        -> nvrhi::ShaderHandle {
        defines.push_back({ "CLUSTER_LOD_HIZ_OCCLUSION", "1" });
        return shaderFactory->CreateShader(name, "main", &defines, nvrhi::ShaderType::Compute);
    };

    psoDescHiz.CS = loadCSHiz("cluster_lod/traversal_init.hlsl", {});
    m_pipelines.computeTraversalInit = device->createComputePipeline(psoDescHiz);

    psoDescHiz.CS = loadCSHiz("cluster_lod/traversal_run.hlsl", {});
    m_pipelines.computeTraversalRun = device->createComputePipeline(psoDescHiz);

    psoDescHiz.CS = loadCSHiz("cluster_lod/traversal_run_groups.hlsl", {});
    m_pipelines.computeTraversalGroups = device->createComputePipeline(psoDescHiz);

    // traversal_setup ignores HiZ but still takes the 3-layout root signature so the
    // whole traversal loop keeps ONE stable signature.  Toggling it mid-loop makes
    // nvrhi re-evaluate bindings and demand m_counters as UnorderedAccess and as
    // IndirectArgument in one barrier batch — an invalid combination that removes
    // the device.
    psoDescHiz.CS = loadCS("cluster_lod/traversal_setup.hlsl");
    m_pipelines.computeBuildSetup = device->createComputePipeline(psoDescHiz);

    // Both caching variants are built so the renderer can toggle caching per frame.
    for (uint32_t caching = 0; caching < 2; ++caching)
    {
        const bool c = (caching != 0);
        psoDescHiz.CS = loadCSHiz("cluster_lod/instance_classify_lod.hlsl",
            { { "USE_BLAS_SHARING", "1" }, { "USE_BLAS_CACHING", c ? "1" : "0" } });
        m_pipelines.computeInstanceClassifyLod[caching] = device->createComputePipeline(psoDescHiz);

        psoDesc.CS = loadCSSharing("cluster_lod/blas_elect_sharing_provider.hlsl", c);
        m_pipelines.computeGeometryBlasSharing[caching] = device->createComputePipeline(psoDesc);

        psoDesc.CS = loadCSSharing("cluster_lod/traversal_init_blas_sharing.hlsl", c);
        m_pipelines.computeTraversalInitBlasSharing[caching] = device->createComputePipeline(psoDesc);
    }

    // No caching permutation here: the merge kernel's caching branch is
    // runtime-gated on streamingRW.useBlasCaching.
    {
        // The kernel's per-group cluster bitmask is a uint4, so a bake with more
        // than 128 clusters per group would silently drop the high clusters.
        uint32_t groupClusterCount = m_maxClustersPerGroup;
        if (groupClusterCount > kMaxGroupClusterCount)
        {
            donut::log::error("ClusterLodPass: clusterGroupSize %u exceeds the %u the "
                              "BLAS-merging bitmask can address; clamping.",
                              groupClusterCount, kMaxGroupClusterCount);
            groupClusterCount = kMaxGroupClusterCount;
        }
        // Round up to the next precompiled shaders.cfg permutation.
        groupClusterCount = std::max(32u, (groupClusterCount + 31u) & ~31u);

        // USE_BLAS_SHARING gives the shared header's u10 geometryBuildInfos; it is
        // part of the permutation key, so shaders.cfg declares it too.
        std::vector<donut::engine::ShaderMacro> defines = {
            { "USE_BLAS_SHARING", "1" },
            { "GROUP_CLUSTER_COUNT", std::to_string(groupClusterCount) } };

        psoDesc.CS = shaderFactory->CreateShader(
            "cluster_lod/traversal_blas_merging.hlsl", "main", &defines,
            nvrhi::ShaderType::Compute);
        m_pipelines.computeTraversalMerge = device->createComputePipeline(psoDesc);
    }
}

// ---------------------------------------------------------------------------
// ClusterLodPass::Execute
// ---------------------------------------------------------------------------

namespace {
// The shaderio::BuildSetup mode IDs pushed below live in shaderio.h.

uint32_t GetThreadGroupCount(uint32_t itemCount, uint32_t itemsPerGroup)
{
    return (itemCount + itemsPerGroup - 1u) / itemsPerGroup;
}
} // anonymous namespace

void ClusterLodPass::Execute(
    nvrhi::IDevice*             device,
    nvrhi::ICommandList*        commandList,
    const ClusterLodPassParams& params)
{
    if (m_numInstances == 0)
        return;

    shaderio::SceneBuildingConstants constants = params.constants;
    constants.numRenderInstances          = m_numInstances;
    constants.numGeometries               = m_numGeometries;
    constants.maxTraversalInfos           = kMaxTraversalInfos;
    constants.maxRenderClusters           = m_maxRenderClusters;
    // Cached-BLAS builds append their CLAS refs to the same pool as the
    // per-instance builds, so reserve room by shrinking the traversal budget.
    if (params.useBlasCaching && params.patchCachedClustersCount < m_maxRenderClusters)
        constants.maxRenderClusters -= params.patchCachedClustersCount;
    // Merging is a runtime gate inside the sharing shaders, so it requires sharing.
    constants.useBlasMerging = (params.useBlasSharing && params.useBlasMerging) ? 1u : 0u;
    // hizNumLODs==0 makes the shader-side IntersectHiz a no-op ("always
    // visible"), so the HiZ set stays bound harmlessly when occlusion is off.
    if (constants.useCulling && params.useHizOcclusion && params.zbuffer)
    {
        constants.hizNumLODs = uint32_t(params.zbuffer->GetNumHiZLODs());
        constants.hizInvSize = params.zbuffer->GetInvHiZSize();
    }
    else
    {
        constants.hizNumLODs = 0u;
        constants.hizInvSize = float2(0.f, 0.f);
    }
    commandList->writeBuffer(m_constantsCB, &constants, sizeof(constants));

    // One BindingSet shared by every traversal dispatch below.
    nvrhi::BindingSetDesc bsDesc;
    bsDesc.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_constantsCB),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_COUNTERS, m_counters.GetBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_TRAVERSAL_NODE_Q, m_traversalNodeQ.GetBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_TRAVERSAL_GROUP_Q, m_traversalGroupQ.GetBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_RENDER_CLUSTERS, m_renderClusterInfos.GetBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_INSTANCE_BUILD_INFOS, m_instanceBuildInfos.GetBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_INSTANCE_BLAS_ADDRS, m_instanceBlasAddrs.GetBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_RESIDENT_GROUPS, m_residentGroupsBuffer->GetBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_STREAMING, m_streamingShaderBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_LOAD_GEOMETRY_GROUPS, m_streamingLoadGroupsBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_GEOMETRY_BUILD_INFOS, m_geometryBuildInfos.GetBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_GEOMETRY_HISTOGRAMS, m_geometryHistograms.GetBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_INSTANCE_VISIBILITY, m_instanceVisibility.GetBuffer()),   // BLAS merging
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_PER_INSTANCE_TRIANGLES, m_perInstanceTriangles.GetBuffer()), // render-stats
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_UNIQUE_SEEN_CLUSTERS, m_uniqueSeenClusters.GetBuffer()),  // render-stats
        // activeGroups is bound whole (Vulkan needs 16B-aligned descriptor
        // offsets); the merge kernel skips the low-detail prefix itself via
        // resident.persistentGroupsCount.
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_ACTIVE_GROUPS, m_activeGroupsBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_GROUP_IDS, m_groupIDsBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(CLOD_U_UNLOAD_GEOMETRY_GROUPS, m_unloadRequestBuffer,
                                                   nvrhi::Format::UNKNOWN,
                                                   m_unloadRingBytes
                                                       ? nvrhi::BufferRange{ 0, m_unloadRingBytes }
                                                       : nvrhi::EntireBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_GEOMETRIES, m_geometriesBuffer->GetBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(CLOD_T_RENDER_INSTANCES, m_renderInstancesBuffer->GetBuffer()),
        nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),
    };
    nvrhi::BindingSetHandle bs = device->createBindingSet(bsDesc, m_bindingLayout);

    auto uavBarrier = [&](nvrhi::IBuffer* buf) {
        commandList->setBufferState(buf, nvrhi::ResourceStates::UnorderedAccess);
        commandList->commitBarriers();
    };
    auto makeState = [&](nvrhi::ComputePipelineHandle pso) {
        nvrhi::ComputeState state;
        state.pipeline = pso;
        state.bindings = { bs, m_descriptorTable };
        return state;
    };

    // Space-1 set for the HiZ PSOs; rebuilt per frame because the ZBuffer's
    // pyramid textures change.
    nvrhi::BindingSetHandle hizSet;
    if (params.zbuffer)
    {
        nvrhi::BindingSetDesc hizBs;
        for (uint32_t i = 0; i < uint32_t(HIZ_MAX_LODS); ++i)
            hizBs.addItem(nvrhi::BindingSetItem::Texture_SRV(0, params.zbuffer->GetHierarchyTexture(i))
                              .setArrayElement(i));
        hizBs.addItem(nvrhi::BindingSetItem::Sampler(0, m_hizSampler));
        hizSet = device->createBindingSet(hizBs, m_hizBindingLayout);
    }
    auto makeStateHiz = [&](nvrhi::ComputePipelineHandle pso) {
        nvrhi::ComputeState state;
        state.pipeline = pso;
        state.bindings = { bs, m_descriptorTable, hizSet };
        return state;
    };
    auto makeIndirectStateHiz = [&](nvrhi::ComputePipelineHandle pso) {
        nvrhi::ComputeState state = makeStateHiz(pso);
        state.indirectParams = m_counters.GetBuffer();
        return state;
    };

    // ---- Clear counters, instance build infos, and traversal queues ---------
    {
        shaderio::SceneBuildingCounters zero{};
        commandList->writeBuffer(m_counters.GetBuffer(), &zero, sizeof(zero));
        commandList->clearBufferUInt(m_instanceBuildInfos.GetBuffer(), 0u);
        // Visibility bits are re-derived from scratch by the sharing shaders.
        commandList->clearBufferUInt(m_instanceVisibility.GetBuffer(), 0u);
        commandList->clearBufferUInt(m_traversalNodeQ.GetBuffer(),  0xFFFFFFFFu);
        commandList->clearBufferUInt(m_traversalGroupQ.GetBuffer(), 0xFFFFFFFFu);
        // Cleared unconditionally so the render-stats buffers stay deterministic
        // regardless of the shader-side TRACK_RENDER_STATS gate.
        commandList->clearBufferUInt(m_perInstanceTriangles.GetBuffer(), 0u);
        commandList->clearBufferUInt(m_uniqueSeenClusters.GetBuffer(),   0u);
    }

    // nvrhi validation requires setPushConstants() on every dispatch using a
    // layout that declares b1, even the shaders that never read it.
    const uint32_t kUnusedPushConstant = 0u;

    if (params.useBlasSharing)
    {
        // instance_classify_lod fills the per-geometry LoD histograms;
        // blas_elect_sharing_provider then elects each geometry's sharing provider.
        const uint32_t cachingPerm = params.useBlasCaching ? 1u : 0u;

        commandList->clearBufferUInt(m_geometryHistograms.GetBuffer(), 0u);
        uavBarrier(m_geometryHistograms.GetBuffer());

        commandList->setComputeState(makeStateHiz(m_pipelines.computeInstanceClassifyLod[cachingPerm]));
        commandList->setPushConstants(&kUnusedPushConstant, sizeof(kUnusedPushConstant));
        commandList->dispatch(GetThreadGroupCount(m_numInstances, shaderio::kInstanceClassifyLodThreads), 1, 1);
        uavBarrier(m_geometryHistograms.GetBuffer());
        uavBarrier(m_instanceBuildInfos.GetBuffer());
        uavBarrier(m_instanceBlasAddrs.GetBuffer());

        commandList->setComputeState(makeState(m_pipelines.computeGeometryBlasSharing[cachingPerm]));
        commandList->setPushConstants(&kUnusedPushConstant, sizeof(kUnusedPushConstant));
        commandList->dispatch(GetThreadGroupCount(m_numGeometries, shaderio::kGeometryBlasSharingThreads), 1, 1);
        uavBarrier(m_geometryBuildInfos.GetBuffer());
    }

    // Seed the queue with instance root nodes.  The sharing variant enqueues only
    // providers and view-dependent instances, and needs no HiZ set because
    // instance_classify_lod already culled.
    if (params.useBlasSharing)
        commandList->setComputeState(makeState(
            m_pipelines.computeTraversalInitBlasSharing[params.useBlasCaching ? 1u : 0u]));
    else
        commandList->setComputeState(makeStateHiz(m_pipelines.computeTraversalInit));
    commandList->setPushConstants(&kUnusedPushConstant, sizeof(kUnusedPushConstant));
    commandList->dispatch(GetThreadGroupCount(m_numInstances, shaderio::kTraversalInitThreads), 1, 1);
    uavBarrier(m_counters.GetBuffer());
    uavBarrier(m_traversalNodeQ.GetBuffer());

    // Do not add a synchronous Download() before the end of the traversal loop:
    // it reopens the command list, dropping the volatile m_constantsCB version,
    // and every later dispatch then reads a dead CB.  Re-write the CB if you must.

    // Fixup kernel for counters in case we tried to add more than available
    // space in the traversal queue.
    {
        commandList->setComputeState(makeStateHiz(m_pipelines.computeBuildSetup));
        const uint32_t setupID = uint32_t(shaderio::BuildSetup::TraversalRun);
        commandList->setPushConstants(&setupID, sizeof(setupID));
        commandList->dispatch(1, 1, 1);
    }
    uavBarrier(m_counters.GetBuffer());  // feeds the next dispatchIndirect

    // =========================================================================
    // Traversal Run
    // =========================================================================
    {
        // this is typically faster
        constexpr bool batchGroupsAtEnd = true;

        const uint32_t kNodesOffset = static_cast<uint32_t>(
            offsetof(shaderio::SceneBuildingCounters, indirectDispatchNodesX));
        const uint32_t kGroupsOffset = static_cast<uint32_t>(
            offsetof(shaderio::SceneBuildingCounters, indirectDispatchGroupsX));

        const uint32_t numPasses = std::max(1u, m_maxNodeTreeDepth);
        for (uint32_t p = 0; p < numPasses; p++)
        {
            commandList->setComputeState(makeIndirectStateHiz(m_pipelines.computeTraversalRun));
            commandList->setPushConstants(&kUnusedPushConstant, sizeof(kUnusedPushConstant));
            commandList->dispatchIndirect(kNodesOffset);
            uavBarrier(m_counters.GetBuffer());
            uavBarrier(m_traversalNodeQ.GetBuffer());
            uavBarrier(m_traversalGroupQ.GetBuffer());

            const bool isLast = (numPasses - 1u == p);
            const uint32_t setupID = uint32_t(!isLast && batchGroupsAtEnd
                ? shaderio::BuildSetup::TraversalRunPassNodesOnly
                : shaderio::BuildSetup::TraversalRunPassCombined);

            commandList->setComputeState(makeStateHiz(m_pipelines.computeBuildSetup));
            commandList->setPushConstants(&setupID, sizeof(setupID));
            commandList->dispatch(1, 1, 1);
            uavBarrier(m_counters.GetBuffer());  // feeds next iteration's dispatchIndirect

            if (setupID == uint32_t(shaderio::BuildSetup::TraversalRunPassCombined))
            {
                commandList->setComputeState(makeIndirectStateHiz(m_pipelines.computeTraversalGroups));
                commandList->setPushConstants(&kUnusedPushConstant, sizeof(kUnusedPushConstant));
                commandList->dispatchIndirect(kGroupsOffset);
                uavBarrier(m_counters.GetBuffer());
                uavBarrier(m_renderClusterInfos.GetBuffer());
                uavBarrier(m_instanceBuildInfos.GetBuffer());
            }
        }
    }

    const bool useBlasMerging = params.useBlasSharing && params.useBlasMerging;
    if (useBlasMerging)
    {
        // This kernel also does the streaming age update for every resident
        // group, which is why the standalone stream_age_groups is skipped.
        assert(params.useStreaming && "useBlasMerging requires streaming");
        if (params.activeGroupsCount)
        {
            commandList->setComputeState(makeState(m_pipelines.computeTraversalMerge));
            commandList->setPushConstants(&kUnusedPushConstant, sizeof(kUnusedPushConstant));
            commandList->dispatch(GetThreadGroupCount(params.activeGroupsCount,
                                                      shaderio::kTraversalBlasMergingThreads), 1, 1);
            uavBarrier(m_counters.GetBuffer());
            uavBarrier(m_renderClusterInfos.GetBuffer());
            uavBarrier(m_instanceBuildInfos.GetBuffer());
        }
    }

    // Fixup kernel for counters in case we tried to add more than available
    // space in the render list.
    {
        commandList->setComputeState(makeStateHiz(m_pipelines.computeBuildSetup));
        const uint32_t setupID = uint32_t(shaderio::BuildSetup::BlasInsertion);
        commandList->setPushConstants(&setupID, sizeof(setupID));
        commandList->dispatch(1, 1, 1);
    }
    uavBarrier(m_counters.GetBuffer());

}
