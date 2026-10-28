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
#pragma once

#include <opensubdiv/version.h>

#include <cstdint>
#include <span>

#include <nvrhi/utils.h>
#include <donut/engine/DescriptorTableManager.h>
// clang-format on

namespace OpenSubdiv::OPENSUBDIV_VERSION::Tmr
{
    class TopologyMap;
}
namespace stats
{
    struct TopologyMapStats;
}

struct TopologyMap
{
    nvrhi::BufferHandle subpatchTreesArraysBuffer;
    nvrhi::BufferHandle patchPointIndicesArraysBuffer;
    nvrhi::BufferHandle stencilMatrixArraysBuffer;
    nvrhi::BufferHandle plansBuffer;

    donut::engine::DescriptorHandle subpatchTreesDescriptor;
    donut::engine::DescriptorHandle patchPointIndicesDescriptor;
    donut::engine::DescriptorHandle stencilMatrixDescriptor;
    donut::engine::DescriptorHandle plansDescriptor;

    std::unique_ptr<OpenSubdiv::OPENSUBDIV_VERSION::Tmr::TopologyMap>
        aTopologyMap;

    TopologyMap(const TopologyMap& other) = delete;
    TopologyMap(TopologyMap&& other) = delete;

    TopologyMap(std::unique_ptr<OpenSubdiv::OPENSUBDIV_VERSION::Tmr::TopologyMap>
        atopologyMap);

    void InitDeviceData(std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTable, nvrhi::ICommandList* commandList, bool keepHostData = true);

    uint64_t GetVramBytes() const;

    // statistics

    struct SubdivisionPlanStats
    {
        uint32_t plansCount = 0;
        size_t plansByteSize = 0;

        uint32_t regularFacePlansCount = 0; // plans created without quadrangulation

        uint32_t maxFaceSize = 0;
        uint32_t sharpnessCount = 0;
        float sharpnessMax = 0.f;

        uint32_t stencilCountMin = ~uint32_t(0);
        uint32_t stencilCountMax = 0;
        float stencilCountAvg = 0.f;
        std::vector<uint32_t> stencilCountHistogram;
    };

    static stats::TopologyMapStats ComputeStatistics(
        const OpenSubdiv::OPENSUBDIV_VERSION::Tmr::TopologyMap& topologyMap,
        int histogramSize = 50);
};
