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

#ifdef __cplusplus
#include <cstdint>
#endif

#include <nvrhi/nvrhiHLSL.h>

// Scratch device memory needed while filling clusters
struct TessellationCounters
{
    uint32_t clusters;
    uint32_t desiredClusters;
    uint32_t desiredVertices;
    uint32_t desiredTriangles;
    uint32_t desiredClasBlocks;

    // Scale on the tessellation rates for this build, from update_tess_rate.hlsl,
    // and whether it differs from the previous build's.
    float rateScale;
    uint32_t rateAdjusted;

    // Scale each budget asks for on its own; rateScale is the smallest.
    float clusterRateScale;
    float clasRateScale;
    float vertexRateScale;

    // Pad for vulkan minStorageBufferOffsetAlignment = 16
    uint32_t pad[2];

#ifdef __cplusplus
    size_t DesiredClasBytes() const { return size_t(desiredClasBlocks) * nvrhi::rt::cluster::kClasByteAlignment; }
#endif
};

#ifdef __cplusplus
constexpr uint32_t kClusterCountByteOffset = offsetof(TessellationCounters, clusters);
#endif

