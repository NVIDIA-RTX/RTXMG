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

// feature_gates.hlsli
// Compile-time feature toggles shared by every cluster_lod shader.  Include it
// wherever one of these macros is branched on so the shader never relies on the
// preprocessor's implicit "undefined == 0"; override any #ifndef gate from
// shaders.cfg (-D USE_BLAS_SHARING=1, ...).

#pragma once

// --- Culling ---------------------------------------------------------------
// Frustum culling has no gate here: it is always compiled and switched at
// runtime (g_Constants.useCulling / useHardCull / hardCullForcesInvisible).
// HiZ occlusion is a PSO permutation: the =1 variant carries the space-1 HiZ
// binding layout, and only that variant is built (shaders.cfg) and loaded
// (pass.cpp).  The runtime off-switch is g_Constants.hizNumLODs == 0.
#ifndef CLUSTER_LOD_HIZ_OCCLUSION
#  define CLUSTER_LOD_HIZ_OCCLUSION 0
#endif

// --- BLAS reduction (also driven per-shader via -D / PSO permutation) ------
#ifndef USE_BLAS_SHARING
#  define USE_BLAS_SHARING 0
#endif
#ifndef USE_BLAS_CACHING
#  define USE_BLAS_CACHING 0
#endif

// --- Per-frame render/memory-stats readback --------------------------------
// Single switch for all DISPLAY-ONLY counters/atomics: CLAS-pool byte
// accounting, BLAS-sharing provider/consumer counts, and the triangle tallies.
// None affect correctness -- they only feed the Profiler "Streaming" tab + HUD
// -- so -D TRACK_RENDER_STATS=0 measures the cost of the stats atomics.
// The traversal_setup desired* overflow signals are deliberately outside the gate:
// the test harness asserts on them.
#ifndef TRACK_RENDER_STATS
#  define TRACK_RENDER_STATS 1
#endif

// --- Streaming debug validation ---------------------------------------------
// Cross-checks the unload patch's host-carried resident IDs against the
// still-resident group blob in stream_allocator_free_groups; a mismatch means
// the free would hit the WRONG CLAS region and raises a negative (GPU-side)
// errorClasDealloc.  Default OFF — costs a bindless blob read per unload.
#ifndef STREAMING_DEBUG_UNLOAD_PATCH_IDS
#  define STREAMING_DEBUG_UNLOAD_PATCH_IDS 0
#endif

// --- Per-scene alpha-mask permutation (host passes -D HAS_ALPHA_TEST={0,1}) -
#ifndef HAS_ALPHA_TEST
#  define HAS_ALPHA_TEST 0
#endif

// --- Clusters per group (host passes -D GROUP_CLUSTER_COUNT={32,64,96,128}) -
// Sizes the per-group cluster bitmask in traversal_blas_merging; 128 is the
// widest a uint4 mask can carry.  Must be >= BakerConfig::clusterGroupSize.
#ifndef GROUP_CLUSTER_COUNT
#  define GROUP_CLUSTER_COUNT 32
#endif


// --- 64-bit atomic add on a RWByteAddressBuffer byte offset ----------------
// The un-suffixed InterlockedAdd binds the 32-bit overload and wraps at 4 GB.
// DXC's SPIR-V backend lacks InterlockedAdd64 (microsoft/DirectXShaderCompiler#5965),
// so Vulkan adds through a uint64_t view aliased on the same binding; declare it
// with RTXMG_BYTEBUFFER_U64_ALIAS.  `off` must be 8-byte aligned.
#if defined(TARGET_VULKAN)
#  define RTXMG_BYTEBUFFER_U64_ALIAS(buf, reg) RWStructuredBuffer<uint64_t> buf##_U64 : reg;
#  define RTXMG_BYTEBUFFER_ATOMIC_ADD_I64(buf, off, val, old)                     \
      do {                                                                        \
          uint64_t rtxmgAtomicOld64_;                                             \
          InterlockedAdd(buf##_U64[(off) >> 3], uint64_t(val), rtxmgAtomicOld64_); \
          (old) = int64_t(rtxmgAtomicOld64_);                                     \
      } while (false)
#else
#  define RTXMG_BYTEBUFFER_U64_ALIAS(buf, reg)
#  define RTXMG_BYTEBUFFER_ATOMIC_ADD_I64(buf, off, val, old) (buf).InterlockedAdd64((off), (val), (old))
#endif
