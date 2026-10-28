/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>

// Cluster-LoD pool budgets; 0 = StreamingConfig's default.  They size GPU pools,
// so they take effect when RTXMGRenderer creates the accel structs.
struct ClusterLodBudgets
{
    // Max simultaneously-resident groups (residency slot-count cap).  Raise for
    // scenes whose full-detail working set exceeds the default.
    uint32_t maxResidentGroups = 0;
    uint32_t maxGeometryMB     = 0;
    uint32_t geometryBlockMB   = 0;
    uint32_t maxClasMB         = 0;
    uint32_t maxBlasCachingMB  = 0;
    // Streaming request throttle.
    uint32_t maxFrameLoadRequests = 0;
    // CLAS-pool cap for the compaction allocator ("--linearalloc <MB>").
    uint32_t clasPoolOverrideMB = 0;
    // Per-frame render-cluster budget exponent: the traversal can emit up to
    // 1 << renderClusterBits clusters; past it clusters drop.  Clamped to [16,25].
    uint32_t renderClusterBits = 20;
};
