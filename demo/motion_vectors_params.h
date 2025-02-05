//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IFclust ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#ifdef __cplusplus
#include <donut/core/math/math.h>
using namespace donut::math;
#endif

static const uint32_t kMotionVectorsNumThreadsX = 8;
static const uint32_t kMotionVectorsNumThreadsY = 8;

static const uint32_t kInvalidBindlessIndex = ~0u;

struct SubdInstance
{
    // Bindless buffer indices
    uint32_t plansBindlessIndex;
    uint32_t stencilMatrixBindlessIndex;
    uint32_t subpatchTreesBindlessIndex;
    uint32_t patchPointIndicesBindlessIndex;

    uint32_t vertexSurfaceDescriptorBindlessIndex;
    uint32_t vertexControlPointIndicesBindlessIndex;
    uint32_t positionsBindlessIndex;
    uint32_t positionsPrevBindlessIndex;

    uint32_t surfaceToGeometryIndexBindlessIndex;
    uint32_t isolationLevel;

    float3x4 prevLocalToWorld;
    float3x4 worldToLocal;

#ifdef __cplusplus
    SubdInstance()
        : plansBindlessIndex(kInvalidBindlessIndex)
        , stencilMatrixBindlessIndex(kInvalidBindlessIndex)
        , subpatchTreesBindlessIndex(kInvalidBindlessIndex)
        , patchPointIndicesBindlessIndex(kInvalidBindlessIndex)
        , vertexSurfaceDescriptorBindlessIndex(kInvalidBindlessIndex)
        , vertexControlPointIndicesBindlessIndex(kInvalidBindlessIndex)
        , positionsBindlessIndex(kInvalidBindlessIndex)
        , positionsPrevBindlessIndex(kInvalidBindlessIndex)
        , surfaceToGeometryIndexBindlessIndex(kInvalidBindlessIndex)
        , isolationLevel(0)
    {
    }

    bool operator==(const SubdInstance& other) const
    {
        return memcmp(this, &other, sizeof(*this)) == 0;
    }
#endif
};