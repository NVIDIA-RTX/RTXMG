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

#include <donut/shaders/binding_helpers.hlsli>
#include "lerp_keyframes_params.h"

StructuredBuffer<float3>            keyframes : register(t0);
StructuredBuffer<LerpKeyFramesPose> poses     : register(t1);

RWStructuredBuffer<float3> posed     : register(u0);
RWStructuredBuffer<float3> posedPrev : register(u1);

DECLARE_PUSH_CONSTANTS(LerpKeyFramesConstants, g_constants, 0, 0);

// x: vertex, y: pose.  Every pose of one mesh in a single dispatch.
[numthreads(64, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    const uint32_t vertexIndex = threadIdx.x;
    if (vertexIndex >= g_constants.numVertices)
        return;

    const LerpKeyFramesPose pose = poses[threadIdx.y];
    const float3 v0 = keyframes[pose.f0 * g_constants.numVertices + vertexIndex];
    const float3 v1 = keyframes[pose.f1 * g_constants.numVertices + vertexIndex];

    const uint32_t dst = threadIdx.y * g_constants.poseStride + vertexIndex;
    posedPrev[dst] = posed[dst];
    posed[dst]     = lerp(v0, v1, pose.blend);
}
