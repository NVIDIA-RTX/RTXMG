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

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#endif

enum class ShadingMode { PRIMARY_RAYS = 0, AO, PT, SHADING_MODE_COUNT };

enum class ColorMode
{
    BASE_COLOR = 0,
    COLOR_BY_SHADING_NORMAL,
    // Shading modes that only work for true cluster builds start here
    COLOR_BY_TEXCOORD,
    COLOR_BY_MATERIAL,
    COLOR_BY_SURFACE_INDEX,
    COLOR_BY_CLUSTER_ID,
    COLOR_BY_MICROTRI_ID,
    COLOR_BY_CLUSTER_UV,
    COLOR_BY_MICROTRI_AREA,
    COLOR_BY_TOPOLOGY,
    // cluster-LOD-specific debug modes
    COLOR_BY_LOD_LEVEL,
    COLOR_BY_CLUSTER_GROUP,
    COLOR_BY_BLAS_SOURCE,
    COLOR_BY_BLAS_CACHED,
    COLOR_MODE_COUNT
};

#ifdef __cplusplus
// Which geometry path can actually render each colour mode.  A mode outside the
// scene's live paths shades flat grey, so the UI groups on this and the cycle
// key skips modes no live path serves.  Host-only: the shader switches on
// colorMode directly.
enum ColorModePath : uint32_t
{
    kColorModeClusterLod  = 1u,
    kColorModeClusterTess = 2u,
    kColorModeAnyPath     = kColorModeClusterLod | kColorModeClusterTess,
};

inline uint32_t GetColorModePaths(ColorMode mode)
{
    switch (mode)
    {
    case ColorMode::COLOR_BY_SURFACE_INDEX:
    case ColorMode::COLOR_BY_MICROTRI_AREA:
    case ColorMode::COLOR_BY_TOPOLOGY:
        return kColorModeClusterTess;
    case ColorMode::COLOR_BY_LOD_LEVEL:
    case ColorMode::COLOR_BY_CLUSTER_GROUP:
    case ColorMode::COLOR_BY_BLAS_SOURCE:
    case ColorMode::COLOR_BY_BLAS_CACHED:
        return kColorModeClusterLod;
    default:
        return kColorModeAnyPath;
    }
}
#endif  // __cplusplus

enum class TonemapOperator
{
    Linear = 0,
    Srgb,
    Aces,  // Academy Color Encoding System
    Hable, // Uncharted 2
    Count
};

enum class BlitDecodeMode
{
    None,
    SingleChannel,
    Depth,
    Normals,
    MotionVectors,
    InstanceId,
    SurfaceIndex,
    SurfaceUv,
    Texcoord
};

enum class MvecDisplacement
{
    FromSubdEval,
    FromMaterial,
    Count
};

enum class DenoiserMode
{
    None,
    DlssSr,
    DlssRr
};

#ifdef __cplusplus
#include <array>
constexpr auto kColorModeNames = std::to_array<const char *>(
{
    "Base Color",
    "Shading Normal",
    "Tex Coord",
    "Material",
    "Surface Index",
    "Cluster ID",
    "MicroTri ID",
    "Cluster UV",
    "MicroTri Area",
    "Topology Quality",
    "LOD Level",
    "Cluster Group",
    "BLAS Source",
    "BLAS Cached"
});
static_assert(kColorModeNames.size() == size_t(ColorMode::COLOR_MODE_COUNT));

constexpr auto kToneMapOperatorNames = std::to_array<const char*>(
{
    "Linear",
    "sRGB",
    "ACES",
    "Hable"
});
static_assert(kToneMapOperatorNames.size() == size_t(TonemapOperator::Count));

constexpr auto kShadingModeNames = std::to_array<const char*>(
{
    "Primary Rays",
    "Ambient Occlusion",
    "Path Tracing"
});
static_assert(kShadingModeNames.size() == size_t(ShadingMode::SHADING_MODE_COUNT));

constexpr auto kMvecDisplacementNames = std::to_array<const char*>(
{
    "From Subd Eval",
    "From Material"
});
static_assert(kMvecDisplacementNames.size() == size_t(MvecDisplacement::Count));
#endif