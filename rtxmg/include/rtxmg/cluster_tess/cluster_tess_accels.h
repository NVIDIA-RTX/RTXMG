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

#pragma once

// clang-format off
#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>

using namespace donut::math;

#include "rtxmg/utils/buffer.h"
#include "rtxmg/utils/cluster_operation.h"
#include "rtxmg/cluster_tess/tessellator_config.h"
// clang-format on

#include <memory>
#include <span>

struct ClusterTessAccels
{
    RTXMGBuffer<uint8_t> blasBuffer;
    RTXMGBuffer<uint8_t> clasBuffer;

    RTXMGBuffer<nvrhi::GpuVirtualAddress> clasPtrsBuffer;  // address of each CLAS header in clasBuffer
    RTXMGBuffer<nvrhi::GpuVirtualAddress> blasPtrsBuffer;  // handles in device memory
    ClusterOperationSizesBuffer blasSizesBuffer;

    // -------------------------------------------------------------------------
    // Cluster data buffer for shading information
    //
    RTXMGBuffer<ClusterTessShadingData> clusterShadingDataBuffer;

    // -------------------------------------------------------------------------
    // Vertex Position buffer that we stage into before creating CLASes
    //
    RTXMGBuffer<float3> clusterVertexPositionsBuffer;

    // -------------------------------------------------------------------------
    // Vertex Normal buffer (optional - only allocated when vertex normals are enabled)
    //
    RTXMGBuffer<float3> clusterVertexNormalsBuffer;
};

struct ClusterTessStatistics
{    
    struct BufferStatistics
    {
        uint32_t m_numClusters = 0;
        uint32_t m_numTriangles = 0;
        size_t m_blasScratchSize = 0;
        size_t m_blasSize = 0;
        size_t m_vertexBufferSize = 0;
        size_t m_vertexNormalsBufferSize = 0;
        size_t m_clasSize = 0;
        size_t m_clusterDataSize = 0;
    };

    BufferStatistics desired;
    BufferStatistics allocated;

    // Adaptive scale on the tessellation rates of the build `desired` was measured
    // on, and whether that build changed it.
    float tessRateScale = 1.f;
    bool  tessRateAdjusted = false;
    // The scale each budget asks for on its own; tessRateScale is the smallest.
    float clusterRateScale = 1.f;
    float clasRateScale = 1.f;
    float vertexRateScale = 1.f;
};
