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

// culling.hlsli — frustum / screen-size / HiZ culling helpers.
//
// IntersectHiz takes the depth pyramid + sampler as PARAMETERS rather than
// reading a global, so this header stays binding-agnostic; each consumer
// declares the pyramid itself (see traversal_init.hlsl).
// Include AFTER traversal_common.hlsli — this uses its ToFloat4x4() to lift the
// row-major float3x4 world matrix to float4x4.

#pragma once

#pragma pack_matrix(row_major)

#include "rtxmg/hiz/hiz_buffer_constants.h"  // HIZ_MAX_LODS, HIZ_LOD0_TILE_SIZE

// HiZ depth-comparison slack, RELATIVE to the occluder distance.  The pyramid
// stores LINEAR view-space depth, so a fixed NDC epsilon would be sub-micron
// here — effectively zero slack, which makes surfaces on an occlusion boundary
// flip visible<->occluded on float noise and LoD wobble.  2% keeps those
// borderline cases conservatively visible.
static const float c_hizRelDepthSlack = 0.02f;
// Clip-w near-zero guard.
static const float c_epsilon    = 1.2e-07f;

// One of the 8 corners of an AABB (n's low 3 bits pick min/max per axis).
float4 CullBoxCorner(float3 bboxMin, float3 bboxMax, int n)
{
    bool3 useMax = bool3((n & 1) != 0, (n & 2) != 0, (n & 4) != 0);
    return float4(lerp(bboxMin, bboxMax, useMax), 1.0f);
}

// Per-corner outside-plane bits.  ANDing the bits over all 8 corners is non-zero
// iff every corner is outside the SAME plane → reject.
uint CullBits(float4 hPos)
{
    uint cullBits = 0u;
    cullBits |= hPos.x < -hPos.w ?  1u : 0u;
    cullBits |= hPos.x >  hPos.w ?  2u : 0u;
    cullBits |= hPos.y < -hPos.w ?  4u : 0u;
    cullBits |= hPos.y >  hPos.w ?  8u : 0u;
    cullBits |= hPos.z <  0.0f   ? 16u : 0u;
    cullBits |= hPos.z >  hPos.w ? 32u : 0u;
    cullBits |= hPos.w <= 0.0f   ? 64u : 0u;
    return cullBits;
}

// Perspective-divide to clip space, flagging a near-degenerate w.
// Divides by abs(w) so behind-eye corners keep their sign.
float4 GetClip(float4 hPos, out bool valid)
{
    valid = !(-c_epsilon < hPos.w && hPos.w < c_epsilon);
    return float4(hPos.xyz / abs(hPos.w), hPos.w);
}

// Frustum test.  Returns true if the object-space AABB, transformed by the
// row-major float3x4 worldMatrix, is at least partially inside the frustum of
// viewProjMatrix (world→clip).  Also outputs the clip-space AABB bounds (xy
// clamped to [-1,1]) + a validity flag for the downstream size / HiZ tests.
bool IntersectFrustum(float4x4 viewProjMatrix, float3 bboxMin, float3 bboxMax,
                      float3x4 worldMatrix,
                      out float4 oClipMin, out float4 oClipMax, out bool oClipValid)
{
    float4x4 worldViewProj = mul(viewProjMatrix, ToFloat4x4(worldMatrix));

    bool   valid;
    float4 hPos      = mul(worldViewProj, CullBoxCorner(bboxMin, bboxMax, 0));
    float4 clip      = GetClip(hPos, valid);
    uint   andBits   = CullBits(hPos);
    uint   orBits    = andBits;
    float4 clipMin   = clip;
    float4 clipMax   = clip;
    bool   clipValid = valid;

    [unroll]
    for (int n = 1; n < 8; n++)
    {
        hPos = mul(worldViewProj, CullBoxCorner(bboxMin, bboxMax, n));
        clip = GetClip(hPos, valid);
        uint bits = CullBits(hPos);
        andBits &= bits;
        orBits  |= bits;
        clipMin   = min(clipMin, clip);
        clipMax   = max(clipMax, clip);
        clipValid = clipValid && valid;
    }

    // Any corner crossing the near plane (bit 16) or the eye plane (bit 64)
    // makes the projected clip-space AABB unreliable — flag it invalid so
    // callers skip the size/HiZ occlusion tests (which would otherwise falsely
    // cull, e.g. large geometry the camera dollies into).
    clipValid = clipValid && (orBits & (16u | 64u)) == 0u;

    oClipValid = clipValid;
    oClipMin   = float4(clamp(clipMin.xy, float2(-1.0f, -1.0f), float2(1.0f, 1.0f)), clipMin.zw);
    oClipMax   = float4(clamp(clipMax.xy, float2(-1.0f, -1.0f), float2(1.0f, 1.0f)), clipMax.zw);
    return andBits == 0u;
}

// Screen-size test: true if the clip-space AABB covers more than `threshold`
// pixels in either axis.  viewportPx = render viewport size in pixels.
bool IntersectSize(float4 clipMin, float4 clipMax, float threshold, float2 viewportPx)
{
    float2 rect = (clipMax.xy - clipMin.xy) * 0.5f * viewportPx;
    return any(rect > float2(threshold, threshold));
}

// HiZ-occlusion test.  Returns true if the instance is potentially VISIBLE,
// false if it is fully occluded by the previous frame's depth.  Consumes the
// clip-space AABB from IntersectFrustum: .xy = NDC bounds in [-1,1], and (from
// GetClip) .w = the per-corner view-space depth, so clipMin.w / clipMax.w are the
// nearest / farthest view-space depth of the AABB.
//
// The HiZ pyramid stores the MAX (farthest) view-space depth per tile, level 0 at
// (screen / HIZ_LOD0_TILE_SIZE) resolution (the Z pre-pass writes it, hiz_pass*
// reduces it), and is passed as an ARRAY because the HiZBuffer is one texture per
// level, not mips.  An AABB is occluded when its NEAREST point (clipMin.w) is
// farther than the farthest occluder in its screen footprint.  hizInvSize = 1 /
// HiZ-base-texel-size, viewportPx = render viewport in pixels.  Conservative: any
// uncertainty returns true, since a false occlusion would punch holes.
bool IntersectHiz(float4 clipMin, float4 clipMax,
                  Texture2D<float> hizFar[HIZ_MAX_LODS], SamplerState hizSampler,
                  uint hizNumLODs, float2 hizInvSize, float2 viewportPx)
{
    float nearViewZ = clipMin.w;   // nearest view-space depth of the AABB
    if (!(nearViewZ > 0.0f))
        return true;               // straddles / behind eye → don't occlusion-cull

    // NDC [-1,1] → screen pixels, Y flipped to texture space (NDC y-up → tex y-down).
    float2 sx = float2(clipMin.x, clipMax.x) * 0.5f + 0.5f;            // [0,1] x (min,max)
    float2 sy = 0.5f - float2(clipMax.y, clipMin.y) * 0.5f;           // [0,1] y (top,bottom)
    float2 sMin = float2(sx.x, sy.x) * viewportPx;
    float2 sMax = float2(sx.y, sy.y) * viewportPx;

    const float invTileSize = 1.0f / float(HIZ_LOD0_TILE_SIZE);
    float sizeInTiles = max(sMax.x - sMin.x, sMax.y - sMin.y) * invTileSize;
    uint  level = (uint)ceil(log2(max(sizeInTiles, 1.0f)));
    if (level >= hizNumLODs)
        return true;               // footprint larger than the pyramid → assume visible

    float2 uv = (sMin + sMax) * 0.5f * invTileSize * hizInvSize;
    // Gather + max over 4 texels in case the footprint straddles a tile boundary.
    float4 z4   = hizFar[NonUniformResourceIndex(level)].Gather(hizSampler, uv, 0);
    float  zFar = max(max(z4.x, z4.y), max(z4.z, z4.w));

    // Visible if the nearest part of the AABB is at/in front of the farthest
    // occluder (with distance-relative slack); occluded only if entirely and
    // meaningfully behind it.
    return nearViewZ <= zFar * (1.0f + c_hizRelDepthSlack);
}
