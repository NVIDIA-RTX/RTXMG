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

// shader_dispatch.h — the cluster-LoD dispatch protocol: the thread-group
// sizes HLSL declares in `numthreads` and C++ divides its dispatch grids by,
// and the mode/branch IDs the multi-branch setup kernels switch on.  Nothing
// here is ever stored in a GPU buffer; that is shaderio.h, which includes this
// header so a consumer of either gets both.

#pragma once

#ifdef __cplusplus
#include <cstdint>
#endif

namespace shaderio {

// Lanes per wave.  Not overridable: the ballot code packs lane bits into a
// single uint32, so a wider wave needs source changes, not a new value.
// Must match the [WaveSize] in rtxmg/utils/wave_size.hlsli.
static const uint32_t kWaveSize = 32;

// traversal_setup.hlsl mode IDs.  Shared between the C++ caller (which writes
// the value as a b1 push constant) and the shader (which switches on it).
// 4 is unused.
enum class BuildSetup
{
    TraversalRun              = 1,
    TraversalRunPassCombined  = 2,
    TraversalRunPassNodesOnly = 3,
    BlasInsertion             = 5,
};

// Cluster-LOD BLAS-pass thread-group sizes.  Shared between C++ (dispatch
// sizing) and HLSL (numthreads).
static const uint32_t kBlasSetupInsertionThreads    = 128;
static const uint32_t kInstancesAssignBlasThreads   = 128;
static const uint32_t kBlasInsertThreads            =  64;

static const uint32_t kBlasCachingSetupBuildThreads =  64;
static const uint32_t kBlasCachingSetupCopyThreads  =  64;

// Traversal thread-group sizes.  Here rather than in traversal_common.hlsli
// because pass.cpp sizes its dispatches from them: that header is HLSL-only,
// so the C++ side had to hand-copy the literals.
static const uint32_t kTraversalInitThreads       = 128;
static const uint32_t kTraversalRunThreads        = 128;
static const uint32_t kTraversalGroupsThreads     =  64;
static const uint32_t kInstanceClassifyLodThreads = 128;
static const uint32_t kGeometryBlasSharingThreads =  64;

// Active streaming compute thread-group sizes.
static const uint32_t kStreamUpdateSceneThreads           = 64;
static const uint32_t kStreamAgeFilterGroupsThreads       = 64;
static const uint32_t kStreamAllocatorLoadGroupsThreads   = 64;
static const uint32_t kStreamAllocatorUnloadGroupsThreads = 64;
// BLAS merging: one thread per resident active group.
static const uint32_t kTraversalBlasMergingThreads        = 64;
// traversal_blas_merging tracks a group's merge-eligible clusters in a single
// 32-bit mask, so BakerConfig::clusterGroupSize may not exceed this.
static const uint32_t kTraversalBlasMergingMaxGroupClusters = 32;
// 64 = 2 waves per group, one sector each; must stay a multiple of kWaveSize
// because streaming.cpp dispatches ceil(sectorCount / (threads / kWaveSize))
// groups and the shader derives its sectorID from the same two constants.
static const uint32_t kStreamAllocatorBuildFreegapsThreads  = 64;
static const uint32_t kStreamAllocatorFreegapsInsertThreads = 64;
static const uint32_t kStreamAllocatorSetupInsertionThreads = 64;

// Compaction-allocator thread-group sizes: NEW=128 (one thread per newly built
// CLAS), OLD=64 (one thread per resident group).
static const uint32_t kStreamCompactionNewClasThreads = 128;
static const uint32_t kStreamCompactionOldClasThreads =  64;

// stream_dispatch_setup.hlsl branch IDs.  Selected via the push constant at
// dispatch time.
enum class StreamSetup
{
    CompactionOldNoUnloads = 0,
    CompactionStatus       = 1,
    AllocatorFreeInsert    = 2,
    AllocatorStatus        = 3,
    UpdateGeometryIndices  = 4,
};

// One wave per geometry-indices task, looping over its task's triangles in
// steps of kWaveSize so sceneMaxClusterTriangles > kWaveSize works.
static const uint32_t kStreamUpdateClasGeometryIndicesThreads = 64;

static const uint32_t kGeometryIndicesTasksPerGroup =
    kStreamUpdateClasGeometryIndicesThreads / kWaveSize;

}  // namespace shaderio
