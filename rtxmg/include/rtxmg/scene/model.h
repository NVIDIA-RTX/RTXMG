#pragma once
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

class SubdivisionSurface;

#include <donut/core/math/math.h>
#include "rtxmg/scene/scene_graph.h"

using namespace donut::math;

struct Instance
{
    std::shared_ptr<RTXMGSceneNode> node;
    affine3 localToWorld = affine3::identity();

    box3 aabb;

    float3 translation = { 0.f, 0.f, 0.f };
    quat rotation = { 1.f, 0.f, 0.f, 0.f };
    float3 scaling = { 1.f, 1.f, 1.f };

    uint32_t meshID = ~uint32_t(0);

    // Keyframe-track offset in frames; instances of an animated mesh with the
    // same offset share one pose (~0 for static meshes).
    float    frameOffset = 0.f;
    uint32_t poseID      = ~uint32_t(0);

    void UpdateLocalTransform();
};

struct Model
{
    int2 frameRange = { std::numeric_limits<int>::max(),
                       std::numeric_limits<int>::min() };
    std::unique_ptr<SubdivisionSurface> subd;
    std::vector<Instance> instances;
};
