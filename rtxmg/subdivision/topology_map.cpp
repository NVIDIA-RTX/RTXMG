/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "rtxmg/subdivision/topology_map.h"

#include "rtxmg/profiler/statistics.h"
#include "rtxmg/subdivision/segmented_vector.h"
#include "rtxmg/subdivision/subdivision_plan_hlsl.h"
#include "rtxmg/utils/buffer.h"

#include <opensubdiv/tmr/neighborhood.h>
#include <opensubdiv/tmr/topologyMap.h>

#include <fstream>


using namespace OpenSubdiv;

// clang-format on

TopologyMap::TopologyMap(
    std::unique_ptr<OpenSubdiv::OPENSUBDIV_VERSION::Tmr::TopologyMap>
    atopologyMap)
    : aTopologyMap(std::move(atopologyMap))
{
}

uint64_t TopologyMap::GetVramBytes() const
{
    return rtxmg::BufferVramBytes(subpatchTreesArraysBuffer) + rtxmg::BufferVramBytes(patchPointIndicesArraysBuffer)
         + rtxmg::BufferVramBytes(stencilMatrixArraysBuffer) + rtxmg::BufferVramBytes(plansBuffer);
}

void TopologyMap::InitDeviceData(std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTable,
    nvrhi::ICommandList* commandList, bool keepHostData)
{
    assert(aTopologyMap);

    Sdc::SchemeType schemeType = aTopologyMap->GetTraits().getSchemeType();
    Tmr::EndCapType endcapType = aTopologyMap->GetTraits().getEndCapType();

    const int numPlans = aTopologyMap->GetNumSubdivisionPlans();

    std::vector<SubdivisionPlanHLSL> gpuPlans(numPlans);

    segmented_vector<uint32_t> gpuTrees;
    gpuTrees.Reserve(numPlans);

    segmented_vector<int> gpuPatchPointIndexArrays;
    gpuPatchPointIndexArrays.Reserve(numPlans);

    segmented_vector<float> gpuStencilMatrixArrays;
    gpuStencilMatrixArrays.Reserve(numPlans);

    for (auto planIndex = 0; planIndex < numPlans; ++planIndex)
    {
        SubdivisionPlanHLSL& gpuPlan = gpuPlans[planIndex];
        // Data is set in the plan, but shader code references hardcodes these
        gpuPlan.scheme = (SchemeType)schemeType;
        gpuPlan.endCap = (EndCapType)endcapType;

        if (const auto* plan = aTopologyMap->GetSubdivisionPlan(planIndex))
        {
            const auto& tree_desc = plan->GetTreeDescriptor();
            gpuPlan.numControlPoints = tree_desc.GetNumControlPoints();
            gpuPlan.coarseFaceQuadrant = tree_desc.GetSubfaceIndex();
            gpuPlan.coarseFaceSize = tree_desc.GetFaceSize();

            gpuPatchPointIndexArrays.Append(plan->GetPatchPoints());
            gpuTrees.Append(plan->GetPatchTreeData());
            gpuStencilMatrixArrays.Append(plan->GetStencilMatrix());
        }
        else
        {
            gpuPlan.numControlPoints = 0;
            gpuPlan.coarseFaceSize = 4;
            gpuPlan.coarseFaceQuadrant = -1;
            static std::vector<int> empty{};
            gpuPatchPointIndexArrays.Append(empty);
            gpuTrees.Append(empty);
            gpuStencilMatrixArrays.Append(
                reinterpret_cast<std::vector<float> &>(empty));
        }
    }

    subpatchTreesArraysBuffer = CreateAndUploadBuffer<uint32_t>(gpuTrees.elements, "subpatch trees arrays", commandList);
    patchPointIndicesArraysBuffer = CreateAndUploadBuffer<int>(gpuPatchPointIndexArrays.elements, "patch point indices arrays", commandList);
    stencilMatrixArraysBuffer = CreateAndUploadBuffer<float>(gpuStencilMatrixArrays.elements, "stencil matrix arrays", commandList);

    for (auto i_plan = 0; i_plan < numPlans; ++i_plan)
    {
        gpuPlans[i_plan].treeOffset = gpuTrees.offsets[i_plan];
        gpuPlans[i_plan].treeSize = gpuTrees.sizes[i_plan];

        gpuPlans[i_plan].patchPointsOffset = gpuPatchPointIndexArrays.offsets[i_plan];
        gpuPlans[i_plan].patchPointsSize = gpuPatchPointIndexArrays.sizes[i_plan];

        gpuPlans[i_plan].stencilMatrixOffset = gpuStencilMatrixArrays.offsets[i_plan];
        gpuPlans[i_plan].stencilMatrixSize = gpuStencilMatrixArrays.sizes[i_plan];
    }

    plansBuffer = CreateAndUploadBuffer<SubdivisionPlanHLSL>(gpuPlans, "plans", commandList);

    subpatchTreesDescriptor = descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, subpatchTreesArraysBuffer));
    patchPointIndicesDescriptor = descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, patchPointIndicesArraysBuffer));
    stencilMatrixDescriptor = descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, stencilMatrixArraysBuffer));
    plansDescriptor = descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, plansBuffer));

    if (!keepHostData)
        aTopologyMap.reset();
}

stats::TopologyMapStats
TopologyMap::ComputeStatistics(const Tmr::TopologyMap& topologyMap,
    int histogramSize)
{
    using namespace OpenSubdiv;

    stats::TopologyMapStats stats;

    {
        auto hashStats = topologyMap.ComputeHashTableStatistics();

        stats.pslMean = hashStats.pslMean;
        stats.hashCount = hashStats.hashCount;
        stats.addressCount = hashStats.addressCount;
        stats.loadFactor = hashStats.loadFactor;
    }

    uint32_t nplans = static_cast<uint32_t>(topologyMap.GetNumSubdivisionPlans());

    if (nplans == 0)
        return stats;

    size_t stencilSum = 0;

    for (uint32_t planIndex = 0; planIndex < nplans; ++planIndex)
    {
        Tmr::SubdivisionPlan const* plan =
            topologyMap.GetSubdivisionPlan(planIndex);

        if (planIndex == 0 && !plan)
        {
            assert(Tmr::TopologyMap::kRegularPlanAtIndexZero);
            continue;
        }

        if (plan->IsRegularFace())
            ++stats.regularFacePlansCount;

        stats.maxFaceSize =
            std::max((uint32_t)plan->GetFaceSize(), stats.maxFaceSize);

        if (plan->GetNumNeighborhoods())
        {

            Tmr::Neighborhood const& n = plan->GetNeighborhood(0);

            Tmr::ConstFloatArray corners = n.GetCornerSharpness();
            Tmr::ConstFloatArray creases = n.GetCreaseSharpness();

            if (bool hasSharpness = !(corners.empty() && creases.empty()))
            {
                ++stats.sharpnessCount;
                for (int i = 0; i < corners.size(); ++i)
                    stats.sharpnessMax = std::max(stats.sharpnessMax, corners[i]);
                for (int i = 0; i < creases.size(); ++i)
                    stats.sharpnessMax = std::max(stats.sharpnessMax, creases[i]);
            }
        }

        size_t nstencils = plan->GetNumStencils();

        stencilSum += nstencils;

        stats.stencilCountMin =
            std::min(stats.stencilCountMin, (uint32_t)nstencils);
        stats.stencilCountMax =
            std::max(stats.stencilCountMax, (uint32_t)nstencils);

        stats.plansByteSize += plan->GetByteSize(true);
    }

    stats.plansCount = nplans - (topologyMap.GetSubdivisionPlan(0) == nullptr);

    stats.stencilCountAvg = float(stencilSum) / float(stats.plansCount);

    // fill stencil counts histogram
    if (stats.stencilCountMin == stats.stencilCountMax)
    {
        // all the plans have the same number of stencils (ex. single cube)
        stats.stencilCountHistogram.push_back(stats.plansCount);
    }
    else
    {
        stats.stencilCountHistogram.resize(histogramSize);

        float delta =
            float(stats.stencilCountMax - stats.stencilCountMin) / histogramSize;

        for (uint32_t planIndex = 0; planIndex < nplans; ++planIndex)
        {
            Tmr::SubdivisionPlan const* plan =
                topologyMap.GetSubdivisionPlan(planIndex);

            if (planIndex == 0 && !plan)
                continue;

            size_t nstencils = plan->GetNumStencils();

            uint32_t i = (uint32_t)std::floor(
                float(nstencils - stats.stencilCountMin) / delta);

            ++stats.stencilCountHistogram[std::min(uint32_t(histogramSize - 1), i)];
        }
    }
    return stats;
}
