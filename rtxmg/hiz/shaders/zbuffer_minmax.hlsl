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

#include "rtxmg/utils/wave_size.hlsli"

Texture2D<float> zbuffer: register(t0);
RWStructuredBuffer<uint> minmax: register(u0); // floats-as-uints because depth values are non-negative.

RTXMG_WAVE_SIZE_32
[numthreads(16, 16, 1)]
void main(uint2 threadIdx : SV_GroupThreadID, uint2 dispatchThreadId : SV_DispatchThreadID)
{
    uint32_t width, height;
    zbuffer.GetDimensions(width, height);

    uint32_t x = dispatchThreadId.x;
    uint32_t y = dispatchThreadId.y;

    if ((x >= width) || (y >= height))
    {
        return;
    }

    float depth = zbuffer[dispatchThreadId];
    if (depth < 0) depth = 0;

    uint wmin = asuint(depth), wmax = asuint(depth);
    if (isinf(depth))
    {
        wmin = 0xffffffff;
        wmax = 0;
    }

    for (int i = 16; i >= 1; i /= 2)
    {
        uint targetLane = WaveGetLaneIndex() ^ i;

        wmin = min(wmin, WaveReadLaneAt(wmin, targetLane));
        wmax = max(wmax, WaveReadLaneAt(wmax, targetLane));
    }
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        uint orig;
        InterlockedMin(minmax[0], asuint(wmin), orig);
        InterlockedMax(minmax[1], asuint(wmax), orig);
    }
}