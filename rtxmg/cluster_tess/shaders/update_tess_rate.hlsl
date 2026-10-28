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

#include <donut/shaders/binding_helpers.hlsli>
#include "rtxmg/cluster_tess/tessellation_counters.h"
#include "rtxmg/cluster_tess/update_tess_rate_params.h"

RWStructuredBuffer<TessellationCounters> u_TessellationCounters : register(u0);

DECLARE_PUSH_CONSTANTS(UpdateTessRateParams, g_Params, 0, 0);

// Demand scales with rate^2, so one step lands the load on kTargetLoad.  Coarsen
// at once above kHighLoad; refine at most kRecoverStep per build below kLowLoad.
static const float kTargetLoad  = 0.75f;
static const float kHighLoad    = 0.85f;
static const float kLowLoad     = 0.6f;
static const float kRecoverStep = 1.03f;
static const float kMaxGrowth   = 1.35f;
// Past this nearly every edge is one segment; patches have a floor of one cluster.
static const float kMinScale    = 1.f / 64.f;

float3 Loads(TessellationCounters c)
{
    return float3(float(c.desiredClusters) / float(g_Params.maxClusters),
                  float(c.desiredClasBlocks) / float(g_Params.maxClasBlocks),
                  float(c.desiredVertices) / float(g_Params.maxVertices));
}

// The scale one budget asks for, given its load at prevScale and its growth since.
float BudgetScale(float prevScale, float load, float growth)
{
    const float predicted = load * growth;
    float scale = prevScale;
    if (predicted > kHighLoad)
        scale *= sqrt(kTargetLoad / predicted);
    else if (load < kLowLoad && growth == 1.f)
        scale *= kRecoverStep;
    return clamp(scale, kMinScale, 1.f);
}

// Picks this build's tessellation rate scale from the previous build's demand,
// which is complete on the GPU by now, unlike its readback.
[numthreads(1, 1, 1)]
void main()
{
    float3 scales = 1.f;
    float prevScale = 1.f;
    if (g_Params.enable && g_Params.numPrevBuilds > 0)
    {
        const TessellationCounters prev = u_TessellationCounters[g_Params.prevSlot];
        prevScale = prev.rateScale > 0.f ? prev.rateScale : 1.f;
        const float3 loads = Loads(prev);

        // Growth is only measured between builds at nearly the same rate:
        // rate^2 holds locally, not near the per-patch floor.
        float3 growth = 1.f;
        if (g_Params.numPrevBuilds > 1)
        {
            const TessellationCounters prev2 = u_TessellationCounters[g_Params.prev2Slot];
            const float step = prev2.rateScale > 0.f ? prevScale / prev2.rateScale : 0.f;
            const float3 prev2Loads = Loads(prev2) * step * step;
            if (step > 0.9f && step < 1.1f)
                growth = select(prev2Loads > 0.f, clamp(loads / max(prev2Loads, 1e-6f), 1.f, kMaxGrowth), 1.f);
        }

        scales = float3(BudgetScale(prevScale, loads.x, growth.x),
                        BudgetScale(prevScale, loads.y, growth.y),
                        BudgetScale(prevScale, loads.z, growth.z));
    }

    const float scale = min(scales.x, min(scales.y, scales.z));
    u_TessellationCounters[g_Params.slot].rateScale = scale;
    u_TessellationCounters[g_Params.slot].rateAdjusted = scale != prevScale ? 1u : 0u;
    u_TessellationCounters[g_Params.slot].clusterRateScale = scales.x;
    u_TessellationCounters[g_Params.slot].clasRateScale = scales.y;
    u_TessellationCounters[g_Params.slot].vertexRateScale = scales.z;
}
