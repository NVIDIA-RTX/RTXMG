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
#pragma pack_matrix(row_major)

#include "rtxmg/cluster_builder/compute_cluster_tiling_params.h"

// TESS_MODE
#define TESS_MODE_SPHERICAL_PROJECTION 0
#define TESS_MODE_WORLD_SPACE_EDGE_LENGTH 1
#define TESS_MODE_UNIFORM 2

// VIS_MODE
#define VIS_MODE_LIMIT_EDGES 0
#define VIS_MODE_SURFACE 1

#define PATCH_POINTS_WRITEABLE


#include <donut/shaders/material_cb.h>
#include <donut/shaders/binding_helpers.hlsli>
#include <donut/shaders/bindless.h>


#include "rtxmg/cluster_builder/cluster.h"
#include "rtxmg/subdivision/subdivision_eval.hlsli"
#include "rtxmg/cluster_builder/tessellation_counters.h"
#include "rtxmg/cluster_builder/tessellator_constants.h"
#include "rtxmg/cluster_builder/tilings.h"
#include "rtxmg/subdivision/osd_ports/tmr/surfaceDescriptor.h"
#include "rtxmg/subdivision/osd_ports/tmr/subdivisionNode.h"
#include "rtxmg/subdivision/osd_ports/tmr/nodeDescriptor.h"
#include "rtxmg/utils/box3.h"
#include "rtxmg/utils/debug.h"
#include "rtxmg/hiz/hiz_buffer_constants.h"

#include "rtxmg/cluster_builder/displacement.hlsli"

static const uint32_t kComputeClusterTilingThreadsX = 32;

ConstantBuffer<ComputeClusterTilingParams> g_Params : register(b0);

StructuredBuffer<float3> t_VertexControlPoints : register(t0);
StructuredBuffer<GeometryData> t_GeometryData : register(t1);
StructuredBuffer<MaterialConstants> t_MaterialConstants: register(t2);
StructuredBuffer<uint16_t> t_SurfaceToGeometryIndex: register(t3);
StructuredBuffer<SurfaceDescriptor> t_VertexSurfaceDescriptors : register(t4);
StructuredBuffer<Index> t_VertexControlPointIndices : register(t5);
StructuredBuffer<uint32_t> t_VertexPatchPointsOffsets : register(t6);
StructuredBuffer<SubdivisionPlanHLSL> t_Plans : register(t7);
StructuredBuffer<uint32_t> t_SubpatchTrees : register(t8);
StructuredBuffer<Index> t_PatchPointIndices : register(t9);
StructuredBuffer<float> t_StencilMatrix : register(t10);
StructuredBuffer<uint32_t> t_ClasInstantiationBytes : register(t11);
StructuredBuffer<nvrhi::GpuVirtualAddress> t_TemplateAddresses : register(t12);
StructuredBuffer<LinearSurfaceDescriptor> t_TexCoordSurfaceDescriptors : register(t13);
StructuredBuffer<Index> t_TexCoordControlPointIndices : register(t14);
StructuredBuffer<uint32_t> t_TexCoordPatchPointsOffsets : register(t15);
StructuredBuffer<float2> t_TexCoords : register(t16);

RWStructuredBuffer<GridSampler> u_GridSamplers : register(u0);
RWStructuredBuffer<TessellationCounters> u_TessellationCounters : register(u1);
RWStructuredBuffer<Cluster> u_Clusters : register(u2);
RWStructuredBuffer<ClusterShadingData> u_ClusterShadingData : register(u3);
RWStructuredBuffer<nvrhi::rt::cluster::IndirectInstantiateTemplateArgs> u_IndirectArgData : register(u4);
RWStructuredBuffer<nvrhi::GpuVirtualAddress> u_ClasAddresses : register(u5);
RWStructuredBuffer<float3> u_VertexPatchPoints : register(u6);
RWStructuredBuffer<float2> u_TexCoordPatchPoints : register(u7);

RWStructuredBuffer<float4> u_Debug : register(u8);

SamplerState s_DisplacementSampler : register(s0);
SamplerState s_HizSampler : register(s1);

VK_BINDING(1, 1) Texture2D t_BindlessTextures[] : register(t0, space2);
Texture2D<float> t_HiZBuffer[HIZ_MAX_LODS]: register(t0, space3);


const static uint32_t nSamples = kComputeClusterTilingWavesPerSurface * kNumWaveSurfaceUVSamples;
groupshared LimitFrame samples[nSamples];
static uint g_debugOutputSlot = 0;

bool HasLimit(uint32_t iSurface)
{
    return t_VertexSurfaceDescriptors[iSurface].HasLimit();
}

bool HiZIsVisible(Box3 aabb)
{
    // tests visibility of a screen-space aligned aabb against hi-z buffer
    if (aabb.m_min.z > 0.f)
    {
        float oldminy = aabb.m_min.y;
        float oldmaxy = aabb.m_max.y;

        aabb.m_max.y = g_Params.viewportSize.y - oldminy;
        aabb.m_min.y = g_Params.viewportSize.y - oldmaxy;

        float invTileSize = 1.f / float(HIZ_LOD0_TILE_SIZE);

        float sizeInTiles = max(aabb.m_max.x - aabb.m_min.x, aabb.m_max.y - aabb.m_min.y) * invTileSize;
        uint32_t level = (uint32_t)ceil(log2(sizeInTiles));

        if (level < g_Params.numHiZLODs)
        {
            float2 uv = float2(aabb.m_min.x + aabb.m_max.x, aabb.m_min.y + aabb.m_max.y) * .5f;
            uv *= invTileSize * g_Params.invHiZSize;

            int lw, lh;
            t_HiZBuffer[level].GetDimensions(lw, lh);

            // gather & reduce 4 texels in case the aabb straddles pixel boundaries
            float4 z4 = t_HiZBuffer[level].Gather(s_HizSampler, uv, 0);
            float zfar = max(max(z4.x, z4.y), max(z4.z, z4.w));

            if (aabb.m_min.z > zfar)
            {
                return false;
            }
        }
    }
    return true;
}

float FrustumVisibility(SubdivisionEvaluatorHLSL subd, uint3 threadIdx)
{
    SurfaceDescriptor desc = subd.GetSurfaceDesc();
    uint32_t numControlPoints = subd.GetPlan().m_data.numControlPoints;

    Box3 aabb;
    aabb.Init();

    // if bit 0 is set, then X value is > +1
    // if bit 1 is set, then X value is < -1
    // bits 2, 3: same for Y
    // bit 4: behind eye
    uint signBits = 0xFF;

    for (uint32_t i = threadIdx.x; i < numControlPoints; i += 32)
    {
        Index index = subd.m_vertexControlPointIndices[desc.firstControlPoint + i];
        float3 cp = subd.m_vertexControlPoints[index];

        float4 p = { cp.x, cp.y, cp.z, 1.0f };
        float3 pWorld = mul(g_Params.localToWorld, p);
        float4 pClip = mul(g_Params.matWorldToClip, float4(pWorld, 1.0f));

        uint16_t bits = 0u;
        bits |= (uint16_t(pClip.x > pClip.w) << 0);
        bits |= (uint16_t(pClip.x < -pClip.w) << 1);
        bits |= (uint16_t(pClip.y > pClip.w) << 2);
        bits |= (uint16_t(pClip.y < -pClip.w) << 3);
        bits |= (uint16_t(pClip.w < 0) << 4); // behind eye

        signBits &= bits;

        // accumulate screen-space AABB

        float3 pScreen = float3(
            (.5f + pClip.x / pClip.w * .5f) * g_Params.viewportSize.x,
            (.5f + pClip.y / pClip.w * .5f) * g_Params.viewportSize.y,
            pClip.w);
        aabb.Include(pScreen);
    }

    // reduce sign bits across lanes
    uint surfaceSignBits = WaveActiveBitAnd(signBits);

    bool visible = (surfaceSignBits == 0);
    bool surfaceVisible = WaveReadLaneAt(visible, 0);

    if (!surfaceVisible || !g_Params.enableHiZVisibility)
    {
        return surfaceVisible;
    }

    // butterfly reduction of AABB across lanes
    for (int i = 16; i >= 1; i /= 2)
    {
        uint targetLane = WaveGetLaneIndex() ^ i;
        aabb.m_min.x = min(aabb.m_min.x, WaveReadLaneAt(aabb.m_min.x, targetLane));
        aabb.m_min.y = min(aabb.m_min.y, WaveReadLaneAt(aabb.m_min.y, targetLane));
        aabb.m_min.z = min(aabb.m_min.z, WaveReadLaneAt(aabb.m_min.z, targetLane));

        aabb.m_max.x = max(aabb.m_max.x, WaveReadLaneAt(aabb.m_max.x, targetLane));
        aabb.m_max.y = max(aabb.m_max.y, WaveReadLaneAt(aabb.m_max.y, targetLane));
        aabb.m_max.z = max(aabb.m_max.z, WaveReadLaneAt(aabb.m_max.z, targetLane));
    }


    if (threadIdx.x == 0 && aabb.Valid())
    {
        aabb.m_min.x = clamp(aabb.m_min.x, 0.f, g_Params.viewportSize.x);
        aabb.m_max.x = clamp(aabb.m_max.x, 0.f, g_Params.viewportSize.x);
        aabb.m_min.y = clamp(aabb.m_min.y, 0.f, g_Params.viewportSize.y);
        aabb.m_max.y = clamp(aabb.m_max.y, 0.f, g_Params.viewportSize.y);

        surfaceVisible = HiZIsVisible(aabb);
    }

    surfaceVisible = WaveReadLaneAt(surfaceVisible, 0);

    return (float)surfaceVisible;
}

float CalculateVisibility(SubdivisionEvaluatorHLSL subd, uint3 threadIdx)
{
    float visibility = 1.0; // fully visible
#if VIS_MODE == VIS_MODE_SURFACE
    if (g_Params.enableFrustumVisibility)
    {
        visibility = FrustumVisibility(subd, threadIdx);
    }
#elif VIS_MODE == VIS_MODE_LIMIT_EDGES
    // for edge visibility, all the work IsBSplinePatch done below int `calculateEdgeVisibility`
#else
#error UNKNOWN VIS MODE VIS_MODE
#endif
    return visibility;
}

float CalculateEdgeVisibility(float visibility, uint32_t waveSampleOffset, uint3 threadIdx)
{
#if VIS_MODE == VIS_MODE_SURFACE
    return visibility;
#elif VIS_MODE == VIS_MODE_LIMIT_EDGES
    //
    // expected limitFrames layout:
    //
    //          e2
    //
    //     p6---p5---p4
    //     |          |
    // e3  p7        p3  e1
    //     |          |
    //     p0---p1---p2
    //
    //          e0
    //

    uint32_t iLane = threadIdx.x;

    // each lane is assigned to one of the 4 surface edges
    if (iLane > 3)
        return 1.f;

    visibility = 1.f;

    // compute a normalized visibility factor for the 3 limit samples locations
    // along the edge against the frustum, hi-z and back-facing criteria.
    float3 pworld[3];
    float4 pclip[3];

    float frustumFactor = 0.f;
    for (int i = 0; i < 3; ++i)
    {
        pworld[i] = samples[waveSampleOffset + (iLane * 2 + i) % kNumWaveSurfaceUVSamples].p;
        pclip[i] = mul(g_Params.matWorldToClip, float4(pworld[i], 1.0));

        float dist = (1.f / pclip[i].w) * sqrt(pclip[i].x * pclip[i].x + pclip[i].y * pclip[i].y);
        frustumFactor = max(frustumFactor, pclip[i].w < 0.f ? 1.f : smoothstep(1.3f, 2.5f, dist));
    }

    if (g_Params.enableFrustumVisibility)
    {
        visibility *= (1.f - frustumFactor);
    }

    if (visibility == 0.f)
        return visibility;

    if (g_Params.enableHiZVisibility)
    {
        float3 pscreen[3];
        for (uint16_t i = 0; i < 3; ++i)
        {
            pscreen[i].x = (.5f + pclip[i].x / pclip[i].w * .5f) * g_Params.viewportSize.x;
            pscreen[i].y = (.5f + pclip[i].y / pclip[i].w * .5f) * g_Params.viewportSize.y;
            pscreen[i].z = pclip[i].w;
        }

        Box3 aabb; 
        aabb.Init(pscreen[0], pscreen[1], pscreen[2]);
        aabb.m_min.x = clamp(aabb.m_min.x, 0.f, g_Params.viewportSize.x);
        aabb.m_max.x = clamp(aabb.m_max.x, 0.f, g_Params.viewportSize.x);
        aabb.m_min.y = clamp(aabb.m_min.y, 0.f, g_Params.viewportSize.y);
        aabb.m_max.y = clamp(aabb.m_max.y, 0.f, g_Params.viewportSize.y);

        if (!HiZIsVisible(aabb))
            return 0.f;
    }

    if (g_Params.enableBackfaceVisibility)
    {
        for (uint16_t i = 0; i < 3; ++i)
        {
            float3 t0 = samples[waveSampleOffset + (iLane * 2 + i) % kNumWaveSurfaceUVSamples].deriv1;
            float3 t1 = samples[waveSampleOffset + (iLane * 2 + i) % kNumWaveSurfaceUVSamples].deriv2;
            float3 nobj = cross(t0, t1);
            float3 nworld = normalize(mul(g_Params.localToWorld, float4(nobj, 0.f)).xyz);

            float cosTheta = dot(normalize(pworld[i] - g_Params.cameraPos), nworld);

            float backfaceFactor = smoothstep(.6f, 1.f, cosTheta);

            visibility *= (1.f - backfaceFactor);
        }
    }
    return visibility;
#else
#error UNKNOWN VIS MODE VIS_MODE
#endif
}

void WaveEvaluateBSplinePatch8(SubdivisionEvaluatorHLSL subd, uint3 threadIdx, uint32_t waveSampleOffset)
{
    const int iLane = threadIdx.x;

    // always do the non-displaced evaluation first.  Displacement maps will perturb this calculation below
    
    if (subd.IsPureBSplinePatch())
    {
        LimitFrame limit = subd.WaveEvaluatePureBsplinePatch8(iLane);
        if (iLane < kNumWaveSurfaceUVSamples)
            samples[waveSampleOffset + iLane] = limit;
    }
    else if (subd.IsBSplinePatch())
    {
        LimitFrame limit = subd.WaveEvaluateBsplinePatch(iLane);
        if (iLane < kNumWaveSurfaceUVSamples)
        {
            samples[waveSampleOffset + iLane] = limit;
        }
    }
    else
    {
        // there is no wave parallel implementation for non-bspline patches falling back to single thread
        subd.WaveEvaluatePatchPoints(iLane);
        if (iLane < kNumWaveSurfaceUVSamples)
        {
            LimitFrame limit = subd.EvaluateLimitSurface(kWaveSurfaceUVSamples[iLane]);
            samples[waveSampleOffset + iLane] = limit;
        }
    }
    
#if DISPLACEMENT_MAPS
    if (iLane < kNumWaveSurfaceUVSamples)
    {
        uint32_t geometryIndex = t_SurfaceToGeometryIndex[subd.m_surfaceIndex] + g_Params.firstGeometryIndex;
        GeometryData geometry = t_GeometryData[geometryIndex];
        MaterialConstants material = t_MaterialConstants[geometry.materialIndex];

        float displacementScale;
        int displacementTexIndex;
        GetDisplacement(material, g_Params.globalDisplacementScale, displacementTexIndex, displacementScale);
        LimitFrame displaced = DoDisplacement(t_TexCoordSurfaceDescriptors,
                t_TexCoordControlPointIndices, t_TexCoordPatchPointsOffsets,
                u_TexCoordPatchPoints, t_TexCoords,
            samples[waveSampleOffset + iLane], subd.m_surfaceIndex, kWaveSurfaceUVSamples[iLane],
            t_BindlessTextures, displacementTexIndex,
            s_DisplacementSampler, displacementScale);
        samples[waveSampleOffset + iLane] = displaced;
    }
#endif
}

void WaveEvaluateTexCoordPatchPoints(uint3 threadIdx,
    StructuredBuffer<LinearSurfaceDescriptor> surfaceDescriptors,
    StructuredBuffer<Index> controlPointIndices,
    StructuredBuffer<uint32_t> patchPointsOffsets,
    TEXCOORD_PATCH_POINTS_TYPE patchPoints,
    StructuredBuffer<float2> controlPoints,
    uint32_t iSurface)
{
    LinearSurfaceDescriptor desc = surfaceDescriptors[iSurface];
    LocalIndex subface = desc.GetQuadSubfaceIndex();

    if (subface == LOCAL_INDEX_INVALID)
        return;

    uint16_t faceSize = desc.GetFaceSize();
    float2 center = (threadIdx.x < faceSize) ?
        controlPoints[controlPointIndices[desc.firstControlPoint + threadIdx.x]] : float2(0, 0);

    for (int offset = 16; offset > 0; offset /= 2)
    {
        center.x += WaveReadLaneAt(center.x, WaveGetLaneIndex() + offset);
        center.y += WaveReadLaneAt(center.y, WaveGetLaneIndex() + offset);
    }

    if (threadIdx.x == 0)
    {
        patchPoints[patchPointsOffsets[iSurface] + 0] = center / faceSize;
    }

    if (threadIdx.x < faceSize)
    {
        float2 a = controlPoints[controlPointIndices[desc.firstControlPoint + threadIdx.x]];
        float2 b = controlPoints[controlPointIndices[desc.firstControlPoint + (threadIdx.x + 1) % faceSize]];
        patchPoints[patchPointsOffsets[iSurface] + threadIdx.x + 1] = 0.5f * (a + b);
    }
}

float CalculateEdgeRates(LimitFrame limitFrame)
{
#if TESS_MODE == TESS_MODE_SPHERICAL_PROJECTION
    const float3 poi = mul(g_Params.localToWorld, float4(limitFrame.p, 1.0f)).xyz;
    const float distance = max(length(poi - g_Params.cameraPos), 0.01f);
    float edgeRate = float(g_Params.viewportSize.y) * g_Params.fineTessellationRate / distance;
    return edgeRate;
#elif TESS_MODE == TESS_MODE_WORLD_SPACE_EDGE_LENGTH
    float diagonalLength = length(g_Params.aabb.Extent());
    return g_Params.fineTessellationRate * 1000.f / diagonalLength;
#endif
    return -1; // should not Get here; uniform tess mode doesn't call this function
}

uint16_t EvaluateEdgeSegments(uint3 threadIdx, float visibility, float visibilityRateMultiplier, uint32_t waveSampleOffset)
{
#if TESS_MODE == TESS_MODE_UNIFORM
    const int iLane = threadIdx.x;
    if (iLane < 4)
    {
        uint32_t segments = g_Params.edgeSegments[iLane];
        float segmentVisibility = CalculateEdgeVisibility(visibility, waveSampleOffset, threadIdx);

        segments = float(segments) * (visibilityRateMultiplier + segmentVisibility * (1.f - visibilityRateMultiplier));
        return (uint16_t)clamp(segments, 1u, 1024u);
    }
    return 0;
#else
    const int iLane = threadIdx.x;

    float segmentRate = iLane < 4 ? CalculateEdgeRates(samples[waveSampleOffset + 2 * iLane + 1]) : .0f;
    if (iLane < kNumWaveSurfaceUVSamples)
    {
        samples[waveSampleOffset + iLane].p = mul(g_Params.localToWorld, float4(samples[waveSampleOffset + iLane].p, 1.0)).xyz;
    }
    float edgeLength = length(samples[waveSampleOffset + (iLane + 1) % kNumWaveSurfaceUVSamples].p - samples[waveSampleOffset + iLane % kNumWaveSurfaceUVSamples].p);

    float finalEdgeLength = edgeLength;
    float edgeLength2 = edgeLength;
    if (iLane < 8)
    {
        edgeLength2 = edgeLength + WaveReadLaneAt(edgeLength, iLane + 1);

        // Get the edge lengths starting at the corner vertices (in 2*iLane threads).
        // if (2 * iLane < kNumWaveSurfaceUVSamples)
        {
            finalEdgeLength = WaveReadLaneAt(edgeLength2, 2 * iLane);
        }
    }

    float edgeVisibility = CalculateEdgeVisibility(visibility, waveSampleOffset, threadIdx);
    segmentRate *= (visibilityRateMultiplier + edgeVisibility * (1.f - visibilityRateMultiplier));

#ifdef MAX_EDGES
    return iLane < 4 ? min(MAX_EDGES, max(1u, (uint32_t)(round(edgeLength * segmentRate)))) : 0;
#else
    float edgeSegments = round(finalEdgeLength * segmentRate);

    return iLane < 4 ? ((uint16_t)max(1.0, edgeSegments)) : 0;
#endif

#endif
}

void GathererWriteCluster(Cluster cluster, uint32_t clusterIndex, GridSampler gridSampler, uint32_t localGeometryIndex, nvrhi::GpuVirtualAddress templateAddress)
{
    const nvrhi::GpuVirtualAddress vertexBufferAddress = g_Params.clusterVertexPositionsBaseAddress + nvrhi::GpuVirtualAddress(cluster.nVertexOffset * sizeof(float3));

    u_IndirectArgData[clusterIndex].clusterIdOffset = clusterIndex;
    u_IndirectArgData[clusterIndex].geometryIndexOffset = localGeometryIndex;
    u_IndirectArgData[clusterIndex].clusterTemplate = templateAddress;
    u_IndirectArgData[clusterIndex].vertexBuffer.startAddress = vertexBufferAddress;
    u_IndirectArgData[clusterIndex].vertexBuffer.strideInBytes = sizeof(float3);

    u_ClusterShadingData[clusterIndex].m_edgeSegments = gridSampler.edgeSegments;
    u_ClusterShadingData[clusterIndex].m_surfaceId = cluster.iSurface;
    u_ClusterShadingData[clusterIndex].m_vertexOffset = cluster.nVertexOffset;
    u_ClusterShadingData[clusterIndex].m_clusterOffset = cluster.offset;
    u_ClusterShadingData[clusterIndex].m_clusterSizeX = cluster.sizeX;
    u_ClusterShadingData[clusterIndex].m_clusterSizeY = cluster.sizeY;
}

void WriteSurfaceWave(uint3 threadIdx, uint3 groupIdx, uint32_t iSurface, uint16_t edgeSegments, uint32_t waveSampleOffset)
{
    // PatchGatherer
    const int iLane = threadIdx.x;
    if (iLane < 4)
    {
        u_GridSamplers[iSurface].edgeSegments[iLane] = edgeSegments;
    }

    GridSampler rSampler = u_GridSamplers[iSurface];
    uint16_t2 surfaceSize = rSampler.GridSize();
    SurfaceTiling surfaceTiling = MakeSurfaceTiling(surfaceSize);

    
    // Get the amount of sizes to enable the templates
#define ENABLE_USE_WAVE_INTRINSICS 1
#if ENABLE_USE_WAVE_INTRINSICS
    _Static_assert(SurfaceTiling::N_SUB_TILINGS <= kComputeClusterTilingThreadsX, "Must have enough lanes to use wave ops");
    uint32_t clasBlocks = 0;
    uint32_t clusterCount = 0;
    uint32_t vertexCount = 0;
    if (iLane < SurfaceTiling::N_SUB_TILINGS)
    {
        ClusterTiling clusterTiling = surfaceTiling.subTilings[iLane];
        uint32_t templateIndex = GetTemplateIndex(clusterTiling.clusterSize);
        clasBlocks += (t_ClasInstantiationBytes[templateIndex] / nvrhi::rt::cluster::kClasByteAlignment) * clusterTiling.ClusterCount();
        clasBlocks += WaveReadLaneAt(clasBlocks, WaveGetLaneIndex() + 2); // Lanes: [0+2], [1+3]
        clasBlocks += WaveReadLaneAt(clasBlocks, WaveGetLaneIndex() + 1); // Lanes: [0+1]

        clusterCount = clusterTiling.ClusterCount();
        clusterCount += WaveReadLaneAt(clusterCount, WaveGetLaneIndex() + 2);
        clusterCount += WaveReadLaneAt(clusterCount, WaveGetLaneIndex() + 1);

        vertexCount = clusterTiling.VertexCount();
        vertexCount += WaveReadLaneAt(vertexCount, WaveGetLaneIndex() + 2);
        vertexCount += WaveReadLaneAt(vertexCount, WaveGetLaneIndex() + 1);
    }
#endif

    bool allocationSucceeded = false;
    // compute cluster and vertex offsets into linear storage using global counters
    uint32_t surfaceClusterOffset, surfaceVertexOffset, clasBlocksOffset;
    if (iLane == 0)
    {   
#if !ENABLE_USE_WAVE_INTRINSICS
        uint32_t clusterCount = surfaceTiling.ClusterCount();
        uint32_t vertexCount = surfaceTiling.VertexCount();

        // Use clas alignment to avoid using 64-bit atomic
        uint32_t clasBlocks = 0;
        for (uint16_t iTiling = 0; iTiling < surfaceTiling.N_SUB_TILINGS; ++iTiling)
        {
            ClusterTiling clusterTiling = surfaceTiling.subTilings[iTiling];
            uint32_t templateIndex = GetTemplateIndex(clusterTiling.clusterSize);
            clasBlocks += (t_ClasInstantiationBytes[templateIndex] / nvrhi::rt::cluster::kClasByteAlignment) * clusterTiling.ClusterCount();
        }
#endif

        InterlockedAdd(u_TessellationCounters[0].desiredClasBlocks, clasBlocks, clasBlocksOffset);
        InterlockedAdd(u_TessellationCounters[0].desiredClusters, clusterCount, surfaceClusterOffset);
        InterlockedAdd(u_TessellationCounters[0].desiredVertices, vertexCount, surfaceVertexOffset);

        uint32_t clusterTris = 2 * (uint32_t)surfaceSize.x * (uint32_t)surfaceSize.y;
        uint32_t dummy;
        InterlockedAdd(u_TessellationCounters[0].desiredTriangles, clusterTris, dummy);

        allocationSucceeded = ((clasBlocksOffset + clasBlocks) <= g_Params.maxClasBlocks) &&
            ((surfaceClusterOffset + clusterCount) <= g_Params.maxClusters) &&
            ((surfaceVertexOffset + vertexCount) <= g_Params.maxVertices);


        // If we passed all allocations, get real offsets here
        if (allocationSucceeded)
        {
            InterlockedAdd(u_TessellationCounters[0].clasBlocks, clasBlocks, clasBlocksOffset);
            InterlockedAdd(u_TessellationCounters[0].clusters, clusterCount, surfaceClusterOffset);
            InterlockedAdd(u_TessellationCounters[0].vertices, vertexCount, surfaceVertexOffset);
            InterlockedAdd(u_TessellationCounters[0].triangles, clusterTris, dummy);
        }        
    }

    allocationSucceeded = WaveReadLaneAt(allocationSucceeded, 0);
    if (!allocationSucceeded)
    {
        return;
    }

    surfaceClusterOffset = WaveReadLaneAt(surfaceClusterOffset, 0);
    surfaceVertexOffset = WaveReadLaneAt(surfaceVertexOffset, 0);
    clasBlocksOffset = WaveReadLaneAt(clasBlocksOffset, 0);

    uint32_t tilingClusterOffset = surfaceClusterOffset;
    uint32_t tilingVertexOffset = surfaceVertexOffset;
    
    nvrhi::GpuVirtualAddress tilingClusterBaseAddress = g_Params.clasDataBaseAddress + clasBlocksOffset * nvrhi::rt::cluster::kClasByteAlignment;
    uint32_t localGeometryIndex = t_SurfaceToGeometryIndex[iSurface];

    for (uint16_t iTiling = 0; iTiling < surfaceTiling.N_SUB_TILINGS; ++iTiling)
    {
        const ClusterTiling clusterTiling = surfaceTiling.subTilings[iTiling];

        const uint16_t2 clusterResolution = uint16_t2(clusterTiling.clusterSize.x, clusterTiling.clusterSize.y);
        const uint32_t templateIndex = GetTemplateIndex(clusterResolution);
        nvrhi::GpuVirtualAddress templateAddress = t_TemplateAddresses[templateIndex];
        uint32_t clasBytes = t_ClasInstantiationBytes[templateIndex];

        // make clusters with tilingSize
        for (uint32_t iCluster = iLane;
            iCluster < clusterTiling.ClusterCount();
            iCluster += 32)
        {
            Cluster cluster = MakeCluster(iSurface, tilingVertexOffset + clusterTiling.ClusterVertexCount() * iCluster,
                    surfaceTiling.ClusterOffset(iTiling, iCluster), clusterTiling.clusterSize.x, clusterTiling.clusterSize.y);
            uint32_t clusterIndex = tilingClusterOffset + iCluster;
            u_Clusters[clusterIndex] = cluster;

            GathererWriteCluster(cluster, clusterIndex, rSampler, localGeometryIndex, templateAddress);

            u_ClasAddresses[clusterIndex] = tilingClusterBaseAddress + clasBytes * iCluster;
        }
        tilingClusterOffset += clusterTiling.ClusterCount();
        tilingVertexOffset += clusterTiling.VertexCount();
        tilingClusterBaseAddress += clasBytes * clusterTiling.ClusterCount();
    }
}

[numthreads(kComputeClusterTilingThreadsX, kComputeClusterTilingWavesPerSurface, 1)]
void main(uint3 threadIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID)
{
    const uint32_t iWave = threadIdx.y;
    const uint32_t iSurface = kComputeClusterTilingWavesPerSurface * groupIdx.x + iWave;

    uint numSurfaceDescriptors, surfaceDescriptorStride;

    t_VertexSurfaceDescriptors.GetDimensions(numSurfaceDescriptors, surfaceDescriptorStride);

    if (!(iSurface < numSurfaceDescriptors))
    {
        return; // early out waves beyond cluster array end
    }

    if (!HasLimit(iSurface))
    {
        return; // don't process surfaces that have no limit
    }

    SubdivisionEvaluatorHLSL subd;
    subd.m_surfaceIndex = iSurface;
    subd.m_isolationLevel = (uint16_t)g_Params.isolationLevel;
    subd.m_surfaceDescriptors = t_VertexSurfaceDescriptors;
    subd.m_plans = t_Plans;
    subd.m_subpatchTrees = t_SubpatchTrees;
    subd.m_patchPointsIndices = t_PatchPointIndices;
    subd.m_stencilMatrix = t_StencilMatrix;

    subd.m_vertexControlPointIndices = t_VertexControlPointIndices;
    subd.m_vertexControlPoints = t_VertexControlPoints;
    subd.m_vertexPatchPointsOffsets = t_VertexPatchPointsOffsets;
    subd.m_vertexPatchPoints = u_VertexPatchPoints;

    WaveEvaluateTexCoordPatchPoints(threadIdx,
        t_TexCoordSurfaceDescriptors, t_TexCoordControlPointIndices,
        t_TexCoordPatchPointsOffsets, u_TexCoordPatchPoints, t_TexCoords,
        iSurface);

    // Frustum "culling"
    float visibility = CalculateVisibility(subd, threadIdx);

    // -------------------------------------------------------------------------
    // Evaluate corner and mid points for surface quad
    // -------------------------------------------------------------------------
    //
    // sample locations:
    //
    //          e2
    //
    //     p6---p5---p4
    //     |          |
    // e3  p7        p3  e1
    //     |          |
    //     p0---p1---p2
    //
    //          e0
    //

    uint32_t waveSampleOffset = kNumWaveSurfaceUVSamples * iWave;
    WaveEvaluateBSplinePatch8(subd, threadIdx, waveSampleOffset);

    const float tessFactor = g_Params.coarseTessellationRate / g_Params.fineTessellationRate;

    uint16_t edgeSegments = EvaluateEdgeSegments(threadIdx, visibility, tessFactor, waveSampleOffset);
    WriteSurfaceWave(threadIdx, groupIdx, iSurface, edgeSegments, waveSampleOffset);
}