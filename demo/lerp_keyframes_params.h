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
//

#ifndef LERP_KEYFRAMES_PARAMS_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define LERP_KEYFRAMES_PARAMS_H

struct LerpKeyFramesConstants
{
    uint32_t numVertices;
    uint32_t poseStride;  // elements between consecutive pose slices
};

// One per pose of the mesh; keyframe indices into the packed keyframe buffer.
struct LerpKeyFramesPose
{
    uint32_t f0;
    uint32_t f1;
    float    blend;
    uint32_t _pad;
};

#endif // LERP_KEYFRAMES_PARAMS_H