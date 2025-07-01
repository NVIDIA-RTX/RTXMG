/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#pragma once

enum class ShadingMode { PRIMARY_RAYS = 0, AO, PT, SHADING_MODE_COUNT };

enum class ColorMode
{
    BASE_COLOR = 0,
    COLOR_BY_NORMAL,
    // Shading modes that only work for true cluster builds start here
    COLOR_BY_TEXCOORD,
    COLOR_BY_MATERIAL,
    COLOR_BY_GEOMETRY_INDEX,
    COLOR_BY_SURFACE_INDEX,
    COLOR_BY_CLUSTER_ID,
    COLOR_BY_MICROTRI_ID,
    COLOR_BY_CLUSTER_UV,
    COLOR_BY_MICROTRI_AREA,
    COLOR_BY_TOPOLOGY,
    COLOR_MODE_COUNT
};

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
#include <cstdlib>

constexpr auto kColorModeNames = std::to_array<const char *>(
{
    "Base Color",
    "Surface Normal",
    "Tex Coord",
    "Material",
    "Geometry Index",
    "Surface Index",
    "Cluster ID",
    "MicroTri ID",
    "Cluster UV",
    "MicroTri Area",
    "Topology Quality"
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