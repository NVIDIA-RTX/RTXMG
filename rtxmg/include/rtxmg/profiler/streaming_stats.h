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

#pragma once

#include <cstdint>

// The cluster-LoD streaming snapshot the profiler displays.  It lives here, not
// in cluster_lod/, so the profiler's public header needs nothing from the
// streaming engine: this is a self-contained POD.

namespace rtxmg {

// Composed in one place, ClusterLodStreaming::GetStats: the sub-managers hand
// back their own slices (StreamingResident::ResidencyStats,
// StreamingStorage::PoolStats) and everything else comes off m_stats.
struct StreamingStats
{
    uint32_t residentGroups    = 0;
    uint32_t residentClusters  = 0;
    uint32_t residentTriangles = 0;
    uint32_t maxGroups         = 0;
    uint32_t maxClusters       = 0;

    // Whether per-vertex normals are kept in the resident geometry pool; drives
    // the Memory-tab "Geometry (…)" channel label.
    bool     residentNormals   = false;

    // Every table the streaming engine allocates that is neither the geometry
    // pool nor the CLAS pool: resident slot tables (sized by maxGroups), the
    // request/patch rings and per-frame CLAS-build arrays (maxPerFrameLoad-
    // Requests), and the persistent allocator's management buffers.  VRAM
    // occupied (rtxmg::BufferVramBytes), so the CPU-mapped rings are excluded.
    uint64_t operationsBytes = 0;

    uint32_t persistentGroups    = 0;
    uint32_t persistentClusters  = 0;
    uint32_t persistentTriangles = 0;
    // VRAM occupied by each geometry's LOD hierarchy + pinned low-detail data,
    // and by the pinned low-detail CLAS.
    uint64_t persistentDataBytes = 0;
    uint64_t persistentClasBytes = 0;

    uint64_t maxDataBytes      = 0;
    uint64_t reservedDataBytes = 0;
    uint64_t usedDataBytes     = 0;
    // VRAM the pool's block buffers occupy — a multiple of the block size, and
    // the only one of the three that is what the card actually gave up.
    uint64_t allocatedDataBytes = 0;

    uint64_t reservedClasBytes = 0;
    uint64_t usedClasBytes     = 0;
    uint64_t wastedClasBytes   = 0;
    uint32_t maxSizedLeft      = 0;
    uint32_t maxSizedReserved  = 0;

    uint64_t maxTransferBytes     = 0;
    uint64_t transferBytes        = 0;
    uint32_t transferCount        = 0;
    uint32_t loadCount            = 0;
    uint32_t unloadCount          = 0;
    uint32_t uncompletedLoadCount = 0;
    uint32_t maxLoadCount         = 0;
    uint32_t maxUnloadCount       = 0;

    // Monotonic since init.  The counters above latch the last completed batch,
    // so only the per-frame delta of these reads 0 while streaming is idle.
    uint64_t totalTransferBytes = 0;
    uint64_t totalLoads         = 0;
    uint64_t totalUnloads       = 0;

    uint32_t couldNotAllocateGroup = 0;
    uint32_t couldNotAllocateClas  = 0;
    uint32_t couldNotTransfer      = 0;
    uint32_t couldNotStore         = 0;

    // Standing population of the resident BLAS cache, not the copies issued this
    // frame (that's SceneBuildingCounters::cachedBlasCopyCounter).
    uint32_t cachedBlasCount = 0;
    uint64_t cachedBlasBytes = 0;
    // Same distinction as allocatedDataBytes: the pool grows in 16 MiB blocks.
    uint64_t allocatedCachedBlasBytes = 0;
    uint64_t maxCachedBlasBytes       = 0;

    // Static scene totals, filled once by InitGeometries: "model" sums each
    // unique geometry once, "scene" weights by instance references.
    uint32_t geometryCount        = 0;
    uint32_t instanceCount        = 0;
    uint64_t modelTriangles       = 0;
    uint64_t modelClusters        = 0;  // LOD0 only
    uint64_t modelClustersAllLods = 0;
    uint64_t modelGroups          = 0;  // all LODs (streaming granularity)
    uint64_t sceneTriangles       = 0;
    uint64_t sceneClusters        = 0;  // LOD0 only
    uint64_t sceneClustersAllLods = 0;  // all LODs, instance-weighted
    uint64_t sceneGroups          = 0;  // all LODs, instance-weighted
};

}  // namespace rtxmg
