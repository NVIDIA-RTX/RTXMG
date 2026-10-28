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

#pragma pack_matrix(row_major)

#include "rtxmg/utils/shader_debug.h"
#include "rtxmg/utils/pixel_pick.h"

#include "render_params.h"
#include "lighting_cb.h"
#include <donut/shaders/lighting.hlsli>
#include <donut/shaders/packing.hlsli>
#include <donut/shaders/surface.hlsli>
#include <donut/shaders/utils.hlsli>
#include <donut/shaders/binding_helpers.hlsli>
#include "rtxmg/scene/material_constants.h"

#include "rtxmg/cluster_tess/cluster_tess.h"
#include "rtxmg/cluster_lod/shaders/cluster_lod_payload.hlsli"
#include "rtxmg/hiz/hiz_buffer_constants.h"
#include "rtxmg/scene/instance_data.h"
#include "rtxmg/utils/constants.h"

#include "ray_payload.h"
#include "color.hlsli"
#include "gbuffer.h"
#include "brdf.hlsli"
#include "utils.hlsli"
#include "envmap/shaders/envmap.hlsli"

// Cluster-LoD shading ladder (CLUSTER_LOD_SHADING permutation).  A ladder rather
// than independent flags: normal mapping perturbs the interpolated vertex normal,
// so it has nothing to stand on at level 0.  Mirrors
// RTXMGRenderer::RayTracingPermutation::ClusterLodShading.
#define CLUSTER_LOD_SHADING_FLAT           0
#define CLUSTER_LOD_SHADING_VERTEX_NORMALS 1
#define CLUSTER_LOD_SHADING_NORMAL_MAPPED  2

// Texture samples for material evaluation
struct MaterialTextureSample
{
    float4 baseOrDiffuse;
    float4 roughness;    // roughness in .r
    float4 metalness;    // metalness in .r
    float4 specularF0;   // specularF0 in .rgb
    float4 emissive;     // emissive in .rgb, multiplies the material emissiveColor
};

#if defined(TARGET_D3D12)
#define CONCAT(a,b) a##b
#define CONCAT_UAV(x) CONCAT(u,x)
#define NV_SHADER_EXTN_SLOT CONCAT_UAV(RTXMG_NVAPI_SHADER_EXT_SLOT)
#define NV_SHADER_EXTN_REGISTER_SPACE space0
#include "nvHLSLExtns.h"

#if DONUT_CLUSTER_API_DXR
uint32_t GetClusterID()
{
    return ClusterID();
}

#define CLUSTER_LOD_AS_POSITION_FETCH 1
float3x3 GetHitTriangleObjectPositions()
{
    const BuiltInTrianglePositions p = TriangleObjectPositions();
    return float3x3(p.p0, p.p1, p.p2);
}
#else
uint32_t GetClusterID()
{
    return NvRtGetClusterID();
}

// AS position fetch (D3D12/NVAPI equivalent of VK's
// gl_HitTriangleVertexPositionsEXT). Rows = object-space triangle vertices.
#define CLUSTER_LOD_AS_POSITION_FETCH 1
float3x3 GetHitTriangleObjectPositions()
{
    return NvRtTriangleObjectPositions();
}
#endif

#elif defined(TARGET_VULKAN)

// Note that `vk::RayTracingPipelineClusterAccelerationStructureCreateInfoNV::allowClusterAccelerationStructures` must
// be set to `true` to make this valid.
[[vk::ext_extension("SPV_NV_cluster_acceleration_structure")]]
[[vk::ext_capability(5437)]]
[[vk::ext_builtin_input(5436)]]
static int const g_ClusterIDNV_;

uint32_t GetClusterID()
{
    return (uint32_t)g_ClusterIDNV_;
}

// AS position fetch, SPIR-V side: HitTriangleVertexPositionsKHR, the counterpart
// of NvRtTriangleObjectPositions().  Needs the device-level
// VK_KHR_ray_tracing_position_fetch feature.
//
// No data-access build flag is needed, unlike a classic triangle BLAS: a CLAS
// stores vertex positions as part of its format, and the only retention control
// is per-CLAS position truncation (StreamingConfig::clasPositionTruncateBits).
[[vk::ext_extension("SPV_KHR_ray_tracing_position_fetch")]]
[[vk::ext_capability(5336)]]
[[vk::ext_builtin_input(5335)]]
static float3 const g_HitTriangleVertexPositionsKHR_[3];

#define CLUSTER_LOD_AS_POSITION_FETCH 1
float3x3 GetHitTriangleObjectPositions()
{
    // Rows = object-space triangle vertices, matching NvRtTriangleObjectPositions().
    return float3x3(g_HitTriangleVertexPositionsKHR_[0],
                    g_HitTriangleVertexPositionsKHR_[1],
                    g_HitTriangleVertexPositionsKHR_[2]);
}

#endif

ConstantBuffer<LightingConstants> g_Const : register(b0);
ConstantBuffer<RenderParams> g_RenderParams : register(b1);

RWTexture2D<float4> u_Accum     : register(u0);

// GBuffer
VK_IMAGE_FORMAT_UNKNOWN RWTexture2D<DepthFormat>    u_Depth         : register(u1);
VK_IMAGE_FORMAT_UNKNOWN RWTexture2D<NormalFormat>   u_Normal        : register(u2);
VK_IMAGE_FORMAT_UNKNOWN RWTexture2D<AlbedoFormat>   u_Albedo        : register(u3);
VK_IMAGE_FORMAT_UNKNOWN RWTexture2D<SpecularFormat> u_Specular      : register(u4);
VK_IMAGE_FORMAT_UNKNOWN RWTexture2D<SpecularHitTFormat> u_SpecularHitT  : register(u5);
VK_IMAGE_FORMAT_UNKNOWN RWTexture2D<RoughnessFormat>    u_Roughness     : register(u6);

RWStructuredBuffer<HitResult>   u_HitResult     : register(u7);

#if ENABLE_DUMP_FLOAT
RWTexture2D<float4> u_DebugTex1 : register(u8);
RWTexture2D<float4> u_DebugTex2 : register(u9);
RWTexture2D<float4> u_DebugTex3 : register(u10);
RWTexture2D<float4> u_DebugTex4 : register(u11);
#endif

#if ENABLE_SHADER_DEBUG
RWStructuredBuffer<ShaderDebugElement> u_PixelDebug : register(u12);
#define SHADER_DEBUG_BUFFER u_PixelDebug
#endif

#if ENABLE_PIXEL_PICK
RWStructuredBuffer<PixelPickResult> u_PixelPick : register(u14);
#define PIXEL_PICK_BUFFER u_PixelPick
#endif

#ifndef TARGET_VULKAN
RWBuffer<uint32_t>  u_TimeviewBuffer : register(u13);
#endif

RaytracingAccelerationStructure SceneBVH : register(t0);
StructuredBuffer<RTXMGInstanceData> t_InstanceData : register(t1);
StructuredBuffer<RTXMGMaterialConstants> t_MaterialConstants : register(t2);
Texture2D<float4> t_EnvMap : register(t3);
StructuredBuffer<float> t_EnvMapConditionalCDF: register(t4);
StructuredBuffer<float> t_EnvMapMarginalCDF : register(t5);
StructuredBuffer<float> t_EnvMapConditionalFunc: register(t6);
StructuredBuffer<float> t_EnvMapMarginalFunc : register(t7);
StructuredBuffer<ClusterTessShadingData> t_ClusterTessShadingData : register(t8);
StructuredBuffer<float3> t_ClusterVertexPositions : register(t9);
#if VERTEX_NORMALS
StructuredBuffer<float3> t_ClusterVertexNormals : register(t10);
#endif
StructuredBuffer<SubdInstance> t_SubdInstances : register(t11);

// Cluster LOD hit shader resources
#include "rtxmg/cluster_lod/shaderio.h"
StructuredBuffer<shaderio::RenderInstance> t_ClusterLodInstances  : register(t12);
StructuredBuffer<shaderio::Geometry>       t_ClusterLodGeometries : register(t13);
// Scene-global resident cluster-range table, indexed by the clusterResidentID
// that traversal emits and the CLAS carries into GetClusterID().
StructuredBuffer<shaderio::ClusterAddress> t_ResidentClusters      : register(t14);
// Flat scene-wide local->global material indirection, sliced by
// shaderio::Geometry::localMaterialsOffset / localMaterialsCount.
StructuredBuffer<uint>                     t_ClusterLodLocalMaterialIDs : register(t15);
#include "rtxmg/cluster_lod/shaders/cluster_lod_material_resolve.hlsli"

SamplerState s_MaterialSampler : register(s0);

#include "self_intersection_avoidance.hlsli"

// Cluster look up
uint16_t2 ClusterGetEdgeSize(uint32_t clusterId)
{
    ClusterTessShadingData clusterShadingData = t_ClusterTessShadingData[clusterId];
    return uint16_t2(clusterShadingData.m_clusterSizeX, clusterShadingData.m_clusterSizeY);
}

uint3 ClusterGetVertexIndices(uint32_t primId)
{
    const uint32_t      clusterId = GetClusterID();
    const uint16_t      triID = (uint16_t)primId;

    // vertex quad ordering: 
    // 23
    // 01
    // triangle ordering: left edge first -- 032+013 (diagonal:03) or 012+132 (diagonal:12)
    // 21 .5    or   2. 54
    // 0. 34    or   01 .3
    // vx,vy are row-major vertex indices in range [0..sx][0..sy] sx,sy are cluster edge m_size
    // if vx,vy are the lower left corner vtx idxs, then diagonal:03 == ((vx & 1) == (vy & 1))

    uint16_t2 clusterEdgeSize = ClusterGetEdgeSize(clusterId);

    const uint16_t qs = clusterEdgeSize.x;      // quad stride
    const uint16_t vs = clusterEdgeSize.x + 1;  // vert stride
    const uint16_t qid = triID >> 1;             // quad id
    const uint16_t qx = qid % qs;               // quad x
    const uint16_t qy = qid / qs;               // quad y
    const uint16_t vid = qy * vs + qx;           // lower-left vertex id
    const bool    diag03 = ((qx & 1) == (qy & 1));       // is diag 0-3 (true) or 1-2 (false)

    const uint16_t df = uint16_t(diag03) << 1 | uint16_t(triID & 1);

    uint3 indices;
    switch (df)
    {
    case 0b00: indices = uint3(vid, vid + 1, vid + vs); break;
    case 0b01: indices = uint3(vid + 1, vid + 1 + vs, vid + vs); break;
    case 0b10: indices = uint3(vid, vid + 1 + vs, vid + vs); break;
    case 0b11: indices = uint3(vid, vid + 1, vid + 1 + vs); break;
    }

    return indices;
}

inline uint16_t2 Index2D(uint32_t indexLinear, uint16_t lineStride)
{
    return uint16_t2(uint16_t(indexLinear % lineStride), uint16_t(indexLinear / lineStride));
}

// Given a cluster triangle id, find the uv coordinates in the parametric surface
// that generated the triangle's three corners.
//
inline void GetSurfaceUV(out float2 uvs[3], ClusterTessShadingData clusterShadingData, uint primId)
{
    const uint3    uMajorVtxIDs = ClusterGetVertexIndices(primId);

    const uint16_t2 clusterSize = uint16_t2(clusterShadingData.m_clusterSizeX, clusterShadingData.m_clusterSizeY);
    const uint16_t2 clusterOffset = clusterShadingData.m_clusterOffset;
    const uint16_t4 edgeSegments = clusterShadingData.m_edgeSegments;

    const GridSampler sampler = { edgeSegments };

    // offset local i,j index to surface index
    uint16_t2 vertexIndex2d = Index2D(uMajorVtxIDs.x, clusterSize.x + 1) + clusterOffset;
    uvs[0] = sampler.UV(vertexIndex2d, (ClusterTessPattern)g_RenderParams.clusterPattern);
    vertexIndex2d = Index2D(uMajorVtxIDs.y, clusterSize.x + 1) + clusterOffset;
    uvs[1] = sampler.UV(vertexIndex2d, (ClusterTessPattern)g_RenderParams.clusterPattern);
    vertexIndex2d = Index2D(uMajorVtxIDs.z, clusterSize.x + 1) + clusterOffset;
    uvs[2] = sampler.UV(vertexIndex2d, (ClusterTessPattern)g_RenderParams.clusterPattern);
}

struct IntersectionRecord
{
    float3 p;             // world space intersection point
    float3 n;             // world space shading normal
    float3 gn;            // world space geometry normal
    float2 texcoord;      // user-assigned texcoord from base mesh
    float3 barycentrics;  // barycentrics
    float3 distToEdge;
    float  hitT;

    // Gbuffer output needed for motion vecs
    uint32_t surfaceIndex;
    float2   surfaceUV;

    MaterialSample ms;
};

void SetupPrimaryRay(uint2 pixelPosition, float2 subPixelJitter, out float3 rayOrigin, out float3 rayDirection)
{
    float2 d = ((float2(pixelPosition) + 0.5f + subPixelJitter) *
        g_RenderParams.camera.dimsInv) *
        2.f -
        1.f;

    d *= float2(1, -1);

    RayDesc ray;
    rayOrigin = g_RenderParams.eye;
    rayDirection = normalize(d.x * g_RenderParams.U + d.y * g_RenderParams.V +
        g_RenderParams.W);
}

RayDesc SetupShadowRay(float3 surfacePos, float3 L)
{
    RayDesc ray;
    ray.Origin = surfacePos - WorldRayDirection() * 0.001;
    ray.Direction = L;
    ray.TMin = 0;
    ray.TMax = 1.#INF;
    return ray;
}

[shader("miss")] void ShadowMiss(inout ShadowRayPayload payload : SV_RayPayload)
{
    payload.missed = true;
}

bool IsOccluded(float3 worldPos, float3 towardsLight)
{
    ShadowRayPayload shadowPayload = (ShadowRayPayload)0;
    shadowPayload.missed = false;

    RayDesc shadowRay = SetupShadowRay(worldPos, towardsLight);

    // Any occluder will do; the any-hit still runs, so blend cards pass through.
    TraceRay(SceneBVH,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xFF,
        1, // shadow hit group
        0,
        1, // shadow miss shader
        shadowRay, shadowPayload);

    return !shadowPayload.missed;
}

enum GeometryAttributes
{
    GeomAttr_Position = 0x01,
    GeomAttr_TexCoord = 0x02,
    GeomAttr_Normal = 0x04,
    GeomAttr_Tangents = 0x08,

    GeomAttr_All = 0x0F
};

struct GeometrySample
{
    RTXMGInstanceData instance;
    RTXMGMaterialConstants material;

    float3 vertexPositions[3];
    float3 vertexNormals[3];
    float2 vertexTexcoords[3];

    float3 barycentrics;
    float2 texcoord;
    float3x4 objectToWorld;
    float3x4 worldToObject;

    uint clusterId;
    uint surfaceIndex;
    float2 surfaceUV;
};

GeometrySample
GetGeometryFromHit(RayPayload payload)
{
    GeometrySample gs = (GeometrySample)0;

    RTXMGInstanceData instance = t_InstanceData[payload.instanceID];

    gs.instance = instance;
    gs.objectToWorld = instance.transform;
    gs.clusterId = ~0u;

    float3x3 w2oRotation = transpose((float3x3)gs.objectToWorld);
    float3 w2oTranslation = -mul(w2oRotation, float3(gs.objectToWorld[0][3], gs.objectToWorld[1][3], gs.objectToWorld[2][3]));
    gs.worldToObject = float3x4(
        w2oRotation[0], w2oTranslation.x,
        w2oRotation[1], w2oTranslation.y,
        w2oRotation[2], w2oTranslation.z);

    gs.barycentrics.yz = payload.barycentrics;
    gs.barycentrics.x = 1.0 - (gs.barycentrics.y + gs.barycentrics.z);


    // Look up cluster geometry data

    uint32_t clusterId = GetClusterID();
    gs.clusterId = clusterId;
    ClusterTessShadingData clusterShadingData = t_ClusterTessShadingData[clusterId];

    // Per-surface lookup, so one mesh can carry several materials.
    {
        SubdInstance subdInstance = t_SubdInstances[payload.instanceID];
        StructuredBuffer<uint32_t> surfaceToMaterialIndex = ResourceDescriptorHeap[NonUniformResourceIndex(subdInstance.surfaceToMaterialIndexBindlessIndex)];
        uint32_t materialIdx = surfaceToMaterialIndex[clusterShadingData.m_surfaceId];
        gs.material = t_MaterialConstants[materialIdx];
        PIXEL_PICK_SUBD(payload.instanceID, clusterShadingData.m_surfaceId, materialIdx);
    }

    uint3 localVtxIndices = ClusterGetVertexIndices(payload.primitiveIndex);
    uint3 globalVtxIndices = localVtxIndices + clusterShadingData.m_vertexOffset;

    // Load vertex positions.
    gs.vertexPositions[0] = t_ClusterVertexPositions[globalVtxIndices[0]];
    gs.vertexPositions[1] = t_ClusterVertexPositions[globalVtxIndices[1]];
    gs.vertexPositions[2] = t_ClusterVertexPositions[globalVtxIndices[2]];

#if VERTEX_NORMALS
    // Load vertex normals.
    gs.vertexNormals[0] = t_ClusterVertexNormals[globalVtxIndices[0]];
    gs.vertexNormals[1] = t_ClusterVertexNormals[globalVtxIndices[1]];
    gs.vertexNormals[2] = t_ClusterVertexNormals[globalVtxIndices[2]];
#endif


    // Use this for a stable cluster ID
    uint linearClusterOffset = (clusterShadingData.m_clusterOffset.y * clusterShadingData.m_clusterSizeX) + clusterShadingData.m_clusterOffset.x;
    SHADER_DEBUG(uint2(clusterShadingData.m_surfaceId, linearClusterOffset));
    SHADER_DEBUG(localVtxIndices);

    // Texcoords
    // Bilinear texcoords
    float2 uvs[3];
    GetSurfaceUV(uvs, clusterShadingData, payload.primitiveIndex);
    float2 uv = gs.barycentrics.x * uvs[0] + gs.barycentrics.y * uvs[1] + gs.barycentrics.z * uvs[2];

    gs.surfaceUV = uv;
    gs.surfaceIndex = clusterShadingData.m_surfaceId;

    // bilerp from 4 corner attributes
    const float u = uv.x;
    const float v = uv.y;

    gs.texcoord = clusterShadingData.m_texcoords[0] * (1.0f - u) * (1.0f - v)
        + clusterShadingData.m_texcoords[1] * u * (1.0f - v)
        + clusterShadingData.m_texcoords[2] * u * v
        + clusterShadingData.m_texcoords[3] * (1.0f - u) * v;

    return gs;
}

// Blinking function that returns true for "on" state based on subframe index
bool GetBlinkState(uint subFrameIndex, uint blinkPeriod = 60)
{
    return (subFrameIndex / blinkPeriod) % 2 == 0;
}

// Project three object-space positions to clip-space NDC.  Shared by the subd and
// cluster-LOD wireframe / micro-triangle area visualizations.
void GetClipPointsFromObjectSpace(
    out float3   outClipPoints[3],
    in  float3   p0, in float3 p1, in float3 p2,
    in  float3x4 objectToWorld)
{
    float3 wp[3];
    wp[0] = mul(objectToWorld, float4(p0, 1.0)).xyz;
    wp[1] = mul(objectToWorld, float4(p1, 1.0)).xyz;
    wp[2] = mul(objectToWorld, float4(p2, 1.0)).xyz;

    float4 pp[3];
    pp[0] = mul(g_RenderParams.viewProjectionMatrix, float4(wp[0], 1.0));
    pp[1] = mul(g_RenderParams.viewProjectionMatrix, float4(wp[1], 1.0));
    pp[2] = mul(g_RenderParams.viewProjectionMatrix, float4(wp[2], 1.0));

    outClipPoints[0] = pp[0].xyz / pp[0].w;
    outClipPoints[1] = pp[1].xyz / pp[1].w;
    outClipPoints[2] = pp[2].xyz / pp[2].w;
}

void GetClipPoints(out float3 outClipPoints[3], in GeometrySample gs)
{
    GetClipPointsFromObjectSpace(outClipPoints,
        gs.vertexPositions[0], gs.vertexPositions[1], gs.vertexPositions[2],
        gs.instance.transform);
}

// Per-barycentric distance-to-opposite-edge scale factor in clip space:
// distToEdge * barycentrics is the screen-space edge distance WireframeWeight wants.
float3 ComputeTriangleDistToEdge(in float3 clipPoints[3], in float3 bary)
{
    float3 e01 = clipPoints[0] - clipPoints[1];
    float3 e12 = clipPoints[1] - clipPoints[2];
    float3 e20 = clipPoints[2] - clipPoints[0];

    float area = 0.5f * length(cross(e01, e20));

    float3 dte = 2.f * area * bary;
    dte.x /= length(e12);
    dte.y /= length(e20);
    dte.z /= length(e01);
    return dte;
}

MaterialSample RTXMG_EvaluateSceneMaterial(GeometrySample gs, float3 normal, MaterialTextureSample textures)
{
    MaterialSample result = DefaultMaterialSample();
    result.roughness = 1;

    ColorMode colorMode = g_RenderParams.colorMode;

    if (colorMode == ColorMode::COLOR_BY_SHADING_NORMAL)
    {
        // `normal` is the shading normal, so this visualizes the interpolated
        // vertex normal when VERTEX_NORMALS is on and the geometric one otherwise.
        result.baseColor = 0.5f * (float3(1, 1, 1) + normal);
        result.diffuseAlbedo = 0.5f * (float3(1, 1, 1) + normal);
    }
    else if (colorMode == ColorMode::COLOR_BY_TOPOLOGY)
    {
        uint32_t clusterId = GetClusterID();
        uint32_t surfaceId = t_ClusterTessShadingData[clusterId].m_surfaceId;

        SubdInstance subdInstance = t_SubdInstances[InstanceID()];
        StructuredBuffer<uint16_t> topologyQuality = ResourceDescriptorHeap[NonUniformResourceIndex(subdInstance.topologyQualityBindlessIndex)];
        uint16_t surfaceValue = topologyQuality[surfaceId];

        float value = float(surfaceValue) / 255.f;

        result.baseColor = lerp(float3(0.f, 1.f, 0.f), float3(1.f, 0.f, 0.f), value);
    }
    else if (colorMode == ColorMode::COLOR_BY_TEXCOORD)
    {
        result.baseColor = float3(frac(gs.texcoord), 0);
        result.diffuseAlbedo = float3(frac(gs.texcoord), 0);
    }
    else if (colorMode == ColorMode::COLOR_BY_MATERIAL)
    {
        float3 hashedColor = UintToColor(gs.material.materialID);
        result.baseColor = hashedColor;
        result.diffuseAlbedo = hashedColor;
    }
    else if (colorMode == ColorMode::BASE_COLOR)
    {
        if (g_RenderParams.shadingMode == ShadingMode::AO)
        {
            result.baseColor = 0.8;
        }
        else
        {
            result.baseColor = lerp(gs.material.baseOrDiffuseColor.rgb, textures.baseOrDiffuse.rgb, textures.baseOrDiffuse.a);

            result.emissiveColor = gs.material.emissiveColor * textures.emissive.rgb;

            result.roughness = gs.material.roughness;
            if (g_RenderParams.roughnessOverride > 0.f)
            {
                result.roughness = g_RenderParams.roughnessOverride;
            }
            else if ((gs.material.flags & RTXMGMaterialFlags_UseRoughnessTexture) != 0)
            {
                result.roughness = textures.roughness.r;
            }
            result.roughness = max(result.roughness, 1e-4f);

            result.metalness = gs.material.metalness;
            result.hasMetalRoughParams = true;
            // Compute the BRDF inputs for the metal-rough model
            // https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#metal-brdf-and-dielectric-brdf
            if (g_RenderParams.shadingMode == ShadingMode::PT)
            {
                if ((gs.material.flags & RTXMGMaterialFlags_UseSpecularF0Texture) != 0)
                {
                    result.metalness = -1.f;
                    result.specularF0 = textures.specularF0.rgb;
                }
                else if ((gs.material.flags & RTXMGMaterialFlags_UseMetalnessTexture) != 0)
                {
                    result.metalness = textures.metalness.r;
                }
            }
            else
            {
                result.diffuseAlbedo = lerp(result.baseColor * (1.0 - c_DielectricSpecular), 0.0, result.metalness);
                result.specularF0 = lerp(c_DielectricSpecular, result.baseColor.rgb, result.metalness);
            }
        }
    }
    else if (colorMode == ColorMode::COLOR_BY_CLUSTER_ID)
    {
        ClusterTessShadingData clusterShadingData = t_ClusterTessShadingData[gs.clusterId];
        uint linearClusterOffset = (clusterShadingData.m_clusterOffset.y * clusterShadingData.m_clusterSizeX) + clusterShadingData.m_clusterOffset.x;

        uint hash = 0;
        hash = MurmurAdd(hash, clusterShadingData.m_surfaceId);
        hash = MurmurAdd(hash, linearClusterOffset);
        float3 hashedColor = UintToColor(hash);
        result.baseColor = hashedColor;
        result.diffuseAlbedo = hashedColor;
    }
    else if (colorMode == ColorMode::COLOR_BY_MICROTRI_ID)
    {
        uint primitiveIndex = PrimitiveIndex();

        ClusterTessShadingData clusterShadingData = t_ClusterTessShadingData[gs.clusterId];
        uint linearClusterOffset = (clusterShadingData.m_clusterOffset.y * clusterShadingData.m_clusterSizeX) + clusterShadingData.m_clusterOffset.x;

        uint hash = 0;
        hash = MurmurAdd(hash, clusterShadingData.m_surfaceId);
        hash = MurmurAdd(hash, linearClusterOffset);
        hash = MurmurAdd(hash, primitiveIndex);
        float3 hashedColor = UintToColor(hash);
        result.baseColor = hashedColor;
        result.diffuseAlbedo = hashedColor;
    }
    else if (colorMode == ColorMode::COLOR_BY_SURFACE_INDEX)
    {
        if (g_RenderParams.debugSurfaceIndex >= 0)
        {
            // Highlighting mode enabled
            if (gs.surfaceIndex == (uint)g_RenderParams.debugSurfaceIndex)
            {
                // This is the highlighted surface - blink between red and dark gray
                bool blinkOn = GetBlinkState(g_RenderParams.subFrameIndex);
                float3 highlightColor = blinkOn ? float3(1.0, 0.0, 0.0) : float3(0.1, 0.1, 0.1);
                result.baseColor = highlightColor;
                result.diffuseAlbedo = highlightColor;
            }
            else
            {
                // All other surfaces are dark gray
                float3 darkGray = float3(0.1, 0.1, 0.1);
                result.baseColor = darkGray;
                result.diffuseAlbedo = darkGray;
            }
        }
        else
        {
            // Normal hashed color scheme when no debug surface is selected
            float3 hashedColor = UintToColor(gs.surfaceIndex);
            result.baseColor = hashedColor;
            result.diffuseAlbedo = hashedColor;
        }
    }
    else if (colorMode == ColorMode::COLOR_BY_CLUSTER_UV)
    {
        result.baseColor = float3(gs.surfaceUV, 0);
        result.diffuseAlbedo = float3(gs.surfaceUV, 0);
    }
    else if (colorMode == ColorMode::COLOR_BY_MICROTRI_AREA)
    {
        float3 clipPoints[3];
        GetClipPoints(clipPoints, gs);

        const float uTriScreenArea = .5f * length(cross(clipPoints[0] - clipPoints[1], clipPoints[2] - clipPoints[0]));
        const float uTriAreaInPixels = g_RenderParams.camera.dims.x * g_RenderParams.camera.dims.y * uTriScreenArea / 4.f;  // the area of the screen is 4.f since the screen vertices range from [-1, 1]

        const float normUTriAreaInPixels = Remap(clamp(uTriAreaInPixels, 0.f, 2.f), 0.f, 2.f, 1.0f, 0.0f);
        result.baseColor = Temperature(normUTriAreaInPixels);
        result.diffuseAlbedo = result.baseColor;
    }
    else
    {
        // Cluster-LOD-only modes have no cluster_tess equivalent.  Match the
        // grey the cluster-LOD path returns for the reverse case; falling
        // through left a zeroed sample, i.e. black, which under the path tracer
        // is indistinguishable from missing geometry.
        result.baseColor     = float3(0.8f, 0.8f, 0.8f);
        result.diffuseAlbedo = float3(0.8f, 0.8f, 0.8f);
    }

    result.occlusion = 1.0;

    // if you need to highlight a particular surface
    if (gs.surfaceIndex == DEBUG_SURFACE)
    {
        result.baseColor = float3(1, 0, 0);
    }

    return result;
}

MaterialTextureSample RTXMG_DefaultMaterialTextures()
{
    MaterialTextureSample values;
    values.baseOrDiffuse = float4(1.0, 1.0, 1.0, 0.0); // fully transparent texture
    values.roughness  = float4(0.8, 0, 0, 0);
    values.metalness  = float4(0.0, 0.0, 0.0, 0.0);
    values.emissive   = float4(1.0, 1.0, 1.0, 1.0); // multiplier: untextured emitters keep emissiveColor
    return values;
}

MaterialSample
SampleGeometryMaterial(GeometrySample gs,
    float3 normal,
    float2 texGradX,
    float2 texGradY,
    float mipLevel, // <-- Use a compile time constant for mipLevel, < 0 for aniso filtering
    SamplerState materialSampler)
{
    MaterialTextureSample textures = RTXMG_DefaultMaterialTextures();

    if ((gs.material.baseOrDiffuseTextureIndex >= 0) &&
        (gs.material.flags & RTXMGMaterialFlags_UseBaseOrDiffuseTexture) != 0)
    {
        Texture2D<float4> diffuseTexture = ResourceDescriptorHeap[NonUniformResourceIndex(
            gs.material.baseOrDiffuseTextureIndex)];

        if (mipLevel >= 0)
            textures.baseOrDiffuse =
            diffuseTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
        else
        {
            textures.baseOrDiffuse = diffuseTexture.SampleGrad(
                materialSampler, gs.texcoord, texGradX, texGradY);
        }
    }

    if ((gs.material.flags & RTXMGMaterialFlags_UseRoughnessTexture) != 0)
    {
        Texture2D<float4> roughnessTex = ResourceDescriptorHeap[NonUniformResourceIndex(
            gs.material.roughnessTextureIndex)];

        if (mipLevel >= 0)
            textures.roughness =
            roughnessTex.SampleLevel(materialSampler, gs.texcoord, mipLevel);
        else
            textures.roughness = roughnessTex.SampleGrad(
                materialSampler, gs.texcoord, texGradX, texGradY);
    }

    if ((gs.material.flags & RTXMGMaterialFlags_UseMetalnessTexture) != 0)
    {
        Texture2D<float4> metalnessTex = ResourceDescriptorHeap[NonUniformResourceIndex(
            gs.material.metalnessTextureIndex)];

        if (mipLevel >= 0)
            textures.metalness =
            metalnessTex.SampleLevel(materialSampler, gs.texcoord, mipLevel);
        else
            textures.metalness = metalnessTex.SampleGrad(
                materialSampler, gs.texcoord, texGradX, texGradY);
    }

    if ((gs.material.flags & RTXMGMaterialFlags_UseSpecularF0Texture) != 0)
    {
        Texture2D<float4> specularF0Tex = ResourceDescriptorHeap[NonUniformResourceIndex(
            gs.material.specularF0TextureIndex)];

        if (mipLevel >= 0)
            textures.specularF0 =
            specularF0Tex.SampleLevel(materialSampler, gs.texcoord, mipLevel);
        else
            textures.specularF0 = specularF0Tex.SampleGrad(
                materialSampler, gs.texcoord, texGradX, texGradY);
    }

    if ((gs.material.flags & RTXMGMaterialFlags_UseEmissiveTexture) != 0)
    {
        Texture2D<float4> emissiveTex = ResourceDescriptorHeap[NonUniformResourceIndex(
            gs.material.emissiveTextureIndex)];

        if (mipLevel >= 0)
            textures.emissive =
            emissiveTex.SampleLevel(materialSampler, gs.texcoord, mipLevel);
        else
            textures.emissive = emissiveTex.SampleGrad(
                materialSampler, gs.texcoord, texGradX, texGradY);
    }

    return RTXMG_EvaluateSceneMaterial(gs, normal, textures);
}

MaterialSample GetMaterialSample(GeometrySample gs, float3 normal)
{
    uint2 pixelPosition = DispatchRaysIndex().xy;

    float2 noJitter = float2(0.f, 0.f);

    float3 ray0Origin, ray0Direction;
    float3 rayXOrigin, rayXDirection;
    float3 rayYOrigin, rayYDirection;

    SetupPrimaryRay(pixelPosition, noJitter, ray0Origin, ray0Direction);
    SetupPrimaryRay(pixelPosition + uint2(1, 0), noJitter, rayXOrigin, rayXDirection);
    SetupPrimaryRay(pixelPosition + uint2(0, 1), noJitter, rayYOrigin, rayYDirection);
    float3 worldSpacePositions[3];
    worldSpacePositions[0] =
        mul(gs.instance.transform, float4(gs.vertexPositions[0], 1.0)).xyz;
    worldSpacePositions[1] =
        mul(gs.instance.transform, float4(gs.vertexPositions[1], 1.0)).xyz;
    worldSpacePositions[2] =
        mul(gs.instance.transform, float4(gs.vertexPositions[2], 1.0)).xyz;
    float3 bary_0 = computeRayIntersectionBarycentrics(
        worldSpacePositions, ray0Origin, ray0Direction);
    float3 bary_x = computeRayIntersectionBarycentrics(
        worldSpacePositions, rayXOrigin, rayXDirection);
    float3 bary_y = computeRayIntersectionBarycentrics(
        worldSpacePositions, rayYOrigin, rayYDirection);
    float2 texCoord0 = interpolate(gs.vertexTexcoords, bary_0);
    float2 texCoordX = interpolate(gs.vertexTexcoords, bary_x);
    float2 texCoordY = interpolate(gs.vertexTexcoords, bary_y);
    float2 texGradX = texCoordX - texCoord0;
    float2 texGradY = texCoordY - texCoord0;

    MaterialSample ms = SampleGeometryMaterial(gs, normal, texGradX, texGradY, -1, s_MaterialSampler);

    return ms;
}

IntersectionRecord GetIntersectionRecord(RayPayload payload)
{
    IntersectionRecord ir = (IntersectionRecord)0;
    ir.hitT = RayTCurrent();

    GeometrySample gs = GetGeometryFromHit(payload);

    float3 objP, objN, wldP;
    float wldOffset;

    SafeSpawnPoint(objP, wldP, objN, ir.gn, wldOffset,
        gs.vertexPositions[0], gs.vertexPositions[1], gs.vertexPositions[2],
        gs.barycentrics.yz, gs.objectToWorld, gs.worldToObject);

    if (dot(ir.gn, WorldRayDirection()) > 0.0f)
    {
        ir.gn = -ir.gn;
    }

    ir.p = SafeSpawnPoint(wldP, ir.gn, wldOffset);

    ir.n = ir.gn;
    
#if VERTEX_NORMALS
    // Use interpolated vertex normals when available
    float3 objInterpolatedNormal = gs.barycentrics.x * gs.vertexNormals[0] + 
                                   gs.barycentrics.y * gs.vertexNormals[1] + 
                                   gs.barycentrics.z * gs.vertexNormals[2];
    
    // Transform to world space and normalize
    float3 worldInterpolatedNormal = normalize(mul((float3x3)gs.objectToWorld, objInterpolatedNormal));
    
    // Handle front-facing (ensure normal faces towards camera)
    if (dot(worldInterpolatedNormal, WorldRayDirection()) > 0.0f)
    {
        worldInterpolatedNormal = -worldInterpolatedNormal;
    }
    
    ir.n = worldInterpolatedNormal;
#endif
    
    ir.texcoord = gs.texcoord;
    ir.barycentrics = gs.barycentrics;
    ir.surfaceIndex = gs.surfaceIndex;
    ir.surfaceUV = gs.surfaceUV;

    ir.ms = GetMaterialSample(gs, ir.n);
    ir.ms.geometryNormal = ir.gn;
    ir.ms.shadingNormal = ir.n;

    if (g_RenderParams.enableWireframe || g_RenderParams.selectedSubdMesh >= 0)
    {
        float3 clipPoints[3];
        GetClipPoints(clipPoints, gs);
        ir.distToEdge = ComputeTriangleDistToEdge(clipPoints, gs.barycentrics);
    }
    return ir;
}

float3 ShadeSurface(IntersectionRecord ir)
{
    float3 diffuseTerm = 0, specularTerm = 0;

    if (!IsOccluded(ir.p, -g_Const.light.direction))
    {
        ShadeSurface(g_Const.light, ir.ms, ir.p, WorldRayDirection(), diffuseTerm,
                       specularTerm);
    }

    return (diffuseTerm + specularTerm +
        ir.ms.diffuseAlbedo * g_Const.ambientColor.rgb);
}

struct Attributes
{
    float2 uv;
};

[shader("miss")] void Miss(inout RayPayload payload
    : SV_RayPayload)
{
    bool hasEnvMap = g_RenderParams.hasEnvironmentMap;

    float3 d = WorldRayDirection();
    float2 u = convertDirToTexCoords(d, g_RenderParams.envmapRotation);

    float3 lightContribution;
    if (g_RenderParams.shadingMode == ShadingMode::PT)
    {
        if (hasEnvMap)
        {
            lightContribution = envMapEvaluate(u, t_EnvMap, g_RenderParams.envmapIntensity, s_MaterialSampler);
        }
        else
        {
            lightContribution = ((float3(d.y, d.y, d.y) + 1.f) / 2.f) * g_RenderParams.missColor;
        }
    }
    else
    {
        lightContribution = g_RenderParams.missColor;
    }

    if (g_RenderParams.shadingMode == ShadingMode::PT)
    {
        float brdfPdf = payload.pdf;
        uint bounce = payload.bounce;

        if (bounce > 0)
        {
            // Calculate pdf if texture environment map is present and when not present,
            // uniformly sample the hemisphere for the gradient environment map
            const float lightPdf = hasEnvMap ? envMapPdf(u, t_EnvMap, t_EnvMapConditionalFunc, t_EnvMapMarginalCDF, s_MaterialSampler) : 1.f / (4.f * M_PIf);
            const float misWeight = PowerHeuristic(1.f, brdfPdf, 1.f, lightPdf);
            const float3 pathWeight = FromRGBe9995(payload.pathWeight);
            lightContribution *= pathWeight * misWeight;
        }
        else
        {
            if (g_RenderParams.enableEnvmapHeatmap)
            {
                float pdf = hasEnvMap ? envMapPdf(u, t_EnvMap, t_EnvMapConditionalFunc, t_EnvMapMarginalCDF, s_MaterialSampler) : 0.0f;
                lightContribution = hasEnvMap ? Temperature(pdf) : Temperature((d.y + 1.f) / 2.f);
            }
        }
        payload.pathContribution = ToRGBe9995(lightContribution);
    }
    else
    {
        payload.pathWeight = ToRGBe9995(lightContribution);
    }
    payload.hitT = 1.#INF;
    payload.instanceID = ~0u;

    bool clearGBuffer = g_RenderParams.denoiserMode != DenoiserMode::None;
    if (clearGBuffer)
    {
        // Write no hit
        uint2 dispatchDims = DispatchRaysDimensions().xy;
        uint2 dispatchPixel = DispatchRaysIndex().xy;
        uint dispatchIndex = dispatchPixel.x + dispatchDims.x * dispatchPixel.y;

        if (g_RenderParams.shadingMode != ShadingMode::PT || payload.bounce == 0)
        {
            u_HitResult[dispatchIndex] = DefaultHitResult();

            // Clear gbuffer
            // using linear depth
            u_Depth[dispatchPixel] = g_RenderParams.zFar;
            u_Normal[dispatchPixel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            u_Albedo[dispatchPixel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            u_Specular[dispatchPixel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            u_Roughness[dispatchPixel] = 0.0f;
        }

        // If PT mode, then if we miss on bounce 0 or 1, then clear to zFar
        // If non-PT mode, then only bounce 0, clear to ZFar
        if (payload.bounce <= 1)
        {
            u_SpecularHitT[dispatchPixel] = g_RenderParams.zFar;
        }
    }
}

float WireframeWeight(IntersectionRecord ir)
{
    float thickness = g_RenderParams.wireframeThickness * 1e-4f;
    float smoothness = 1e-7f;
    float3 b = ir.barycentrics * ir.distToEdge;

    float minBary = min(min(b.x, b.y), b.z);
    return smoothstep(thickness, thickness + smoothness, minBary);
}

float3 AOSample(const float3 normal, const float2 u)
{
    const Onb onb = MakeOnb(normal);
    float3    dir = CosineSampleHemisphere(u);
    dir = onb.ToWorld(dir);
    return normalize(dir);
}

float3 SampleDirect(MaterialSample material, float3 p, float3 gN, float3 N, float3 V, inout uint32_t seed)
{
    float2 u = float2(Rnd(seed), Rnd(seed));

    float lightPdf = 1.f;
    float3 envMapColor;
    float3 L = envMapImportanceSample(u, g_RenderParams.envmapRotationInv,
        lightPdf, envMapColor, t_EnvMap, g_RenderParams.envmapIntensity, t_EnvMapConditionalFunc, t_EnvMapMarginalFunc, t_EnvMapConditionalCDF, t_EnvMapMarginalCDF, s_MaterialSampler);
    float3 lightContribution = 0.f;
    if (IsOccluded(p, L))
    {
        return lightContribution;
    }
    lightContribution = envMapColor;
    float brdfPdf = 1.f;
    const float3 brdfWeight = BRDFEval(material, gN, N, V, L, brdfPdf);
    const float misWeight = PowerHeuristic(1.f, lightPdf, 1.f, brdfPdf);
    // brdfWeight includes scaling by dot( N, L )
    float3 result = lightContribution * brdfWeight * misWeight / lightPdf;
    return result;
}

// Thin-walled dielectric glass BSDF sample (KHR_materials_transmission): a
// stochastic choice between Fresnel reflection and straight-through transmission.
// pdf comes back large so this delta lobe wins MIS (glass is not NEE-sampled), and
// didTransmit tells the caller to push the continuation ray past the surface.
float3 DielectricSampleThin(float3 N, float3 V, float ior, float transmissionFactor,
                            float3 tint, out float3 L, out float pdf,
                            inout uint32_t seed, out bool didTransmit)
{
    // Face the normal toward the incoming ray (handles back-side hits).
    if (dot(N, V) < 0.f) N = -N;
    const float NdotV = saturate(dot(N, V));

    // Schlick Fresnel from the dielectric F0 = ((ior-1)/(ior+1))^2.
    const float r0 = (ior - 1.f) / (ior + 1.f);
    const float F0 = r0 * r0;
    const float F  = F0 + (1.f - F0) * pow(1.f - NdotV, 5.f);

    pdf = 1e9f; // delta lobe

    if (Rnd(seed) < F)
    {
        // Uncolored specular reflection, sampled with probability F, carrying F
        // — so weight 1.
        L = reflect(-V, N);
        didTransmit = false;
        return float3(1.f, 1.f, 1.f);
    }
    // Thin-walled transmission: ray continues straight through, tinted by the
    // glass color. Sampled with probability (1-F); weight = tint * factor.
    L = -V;
    didTransmit = true;
    return tint * transmissionFactor;
}

[shader("closesthit")]void ClosestHit(inout RayPayload payload
    : SV_RayPayload, in Attributes attrib
    : SV_IntersectionAttributes)
{
    SHADER_DEBUG_INIT(g_RenderParams.debugPixel, DispatchRaysIndex().xy);
    PIXEL_PICK_INIT(g_RenderParams.debugPixel, DispatchRaysIndex().xy);

    payload.instanceID = InstanceID();
    payload.primitiveIndex = PrimitiveIndex();
    payload.geometryIndex = GeometryIndex();
    payload.barycentrics = attrib.uv;
    payload.hitT = RayTCurrent();

    SHADER_DEBUG(uint4(payload.instanceID, payload.primitiveIndex, payload.geometryIndex, GetClusterID()));
    SHADER_DEBUG(float3(payload.barycentrics, payload.hitT));

    IntersectionRecord ir = GetIntersectionRecord(payload);

    float3 pathWeight = 1.f;

    SubdInstance subdInstance = t_SubdInstances[InstanceID()];
    const bool isSelectedSubd = g_RenderParams.selectedSubdMesh >= 0
                             && subdInstance.meshID == uint(g_RenderParams.selectedSubdMesh);
    float wfWeight = (g_RenderParams.enableWireframe || isSelectedSubd) ? WireframeWeight(ir) : 1.0f;

    const bool onSelectionWireframe = isSelectedSubd && wfWeight == 0.f;
    const float3 selectionRed = float3(1.f, 0.05f, 0.05f);
    if (onSelectionWireframe)
    {
        ir.ms.baseColor          = selectionRed;
        ir.ms.diffuseAlbedo      = selectionRed;
        ir.ms.emissiveColor      = 0.f;
        ir.ms.metalness          = 0.f;
        ir.ms.hasMetalRoughParams = false;
    }

    if (g_RenderParams.denoiserMode != DenoiserMode::None)
    {
        uint2 dispatchDims = DispatchRaysDimensions().xy;
        uint2 dispatchPixel = DispatchRaysIndex().xy;
        uint dispatchIndex = dispatchPixel.x + dispatchDims.x * dispatchPixel.y;

        const bool writePrimarySurfaceGBuffer = (g_RenderParams.shadingMode != ShadingMode::PT || payload.bounce == 0);

        if (writePrimarySurfaceGBuffer)
        {
            HitResult hitResult;
            hitResult.instanceId = payload.instanceID;
            hitResult.surfaceIndex = ir.surfaceIndex;
            hitResult.surfaceUV = ir.surfaceUV;
            hitResult.texcoord = ir.texcoord;
            u_HitResult[dispatchIndex] = hitResult;

            float depth = dot(normalize(g_RenderParams.W), ir.p - g_RenderParams.eye);
            u_Depth[dispatchPixel] = depth;
            u_Normal[dispatchPixel] = float4(ir.n, 0.f);

            const float wfGuideWeight = onSelectionWireframe ? 1.f : wfWeight;
            float3 V = -WorldRayDirection();
            FresnelBlend brdf = MakeFresnelBlend(ir.ms.baseColor, ir.ms.specularF0, ir.ms.metalness, ir.ms.roughness);
            u_Albedo[dispatchPixel] = float4(brdf.m_diffuse.m_albedo * wfGuideWeight, 1.0f);
            float3 spec = BRDFEnvApprox(brdf, ir.n, V);
            u_Specular[dispatchPixel] = float4(spec * wfGuideWeight, 1.0f);
            u_Roughness[dispatchPixel] = ir.ms.roughness;
        }

        if (g_RenderParams.shadingMode != ShadingMode::PT)
        {
            // Non-PT mode clear to far Z
            u_SpecularHitT[dispatchPixel] = g_RenderParams.zFar;
        }
        else if (payload.bounce == 1)
        {
            // We want to write the hit distance from primary surface to specular hit
            u_SpecularHitT[dispatchPixel] = ir.hitT;
        }
    }

    if (wfWeight == 0.f && !onSelectionWireframe)
    {
        payload.pathWeight = 0;
        return;
    }


    if (g_RenderParams.shadingMode == ShadingMode::PRIMARY_RAYS)
    {
        pathWeight = ir.ms.baseColor;
        payload.pathWeight = ToRGBe9995(pathWeight);
    }
    else if (g_RenderParams.shadingMode == ShadingMode::AO)
    {
        uint spp = g_RenderParams.denoiserMode != DenoiserMode::None ? 1 : g_RenderParams.spp;

        uint32_t seed = payload.seed;
        float2 subPixelJitter = RandomStrat(payload.multipurposeField, sqrt(spp), seed);
        float3 L = AOSample(ir.n, subPixelJitter);
        bool occluded = IsOccluded(ir.p, L);
        pathWeight *= ir.ms.baseColor * (occluded ? 0.f : 1.f);
        payload.pathWeight = ToRGBe9995(pathWeight);
        payload.seed = seed;
    }
    else if (g_RenderParams.shadingMode == ShadingMode::PT)
    {
        uint32_t seed = payload.seed;
        float3 V = -WorldRayDirection();

        float3 pw = FromRGBe9995(payload.pathWeight);
        pathWeight *= pw;
        float3 lightContribution = 0;
        float samplePdf = payload.pdf;

        if (g_RenderParams.hasEnvironmentMap)
        {
            float3 directLighting = SampleDirect(ir.ms, ir.p, ir.gn, ir.n, V, seed);
            lightContribution = pathWeight * directLighting;
        }

        // Only the envmap is NEE-sampled, so adding the full emission at every
        // path vertex is unbiased — nothing to double count against.
        lightContribution += pathWeight * ir.ms.emissiveColor;

        if (payload.bounce < g_RenderParams.ptMaxBounces - 1)
        {
            // No need to do this on the final hit, since we won't trace 
            // another ray anyway
            float3 L = 0;

            float3 indirectWeight;

            indirectWeight = BRDFSample(ir.ms, ir.gn, ir.n, V, L, samplePdf, seed);
            pathWeight *= indirectWeight;

            payload.pathWeight = ToRGBe9995(pathWeight);
            payload.multipurposeField = PackNormalizedVector(L);
            payload.rayOrigin = ir.p;

            payload.pdf = samplePdf;
            payload.seed = seed;
        }
        payload.pathContribution = ToRGBe9995(lightContribution);
    }
}

void TraceRadiancePT(uint           bounce,
                     inout uint     seed,
                     const uint32_t subPixelIndex,
                     inout float3   rayOrigin,
                     inout float3   rayDirection,
                     inout float3   pathWeight,
                     inout float3   pathContribution,
                     inout float    pdf,
                     out float      hitT)
{
    RayPayload payload = (RayPayload)0;
    payload.instanceID = ~0u;
    payload.pathWeight = ToRGBe9995(pathWeight);
    payload.bounce = bounce;
    payload.multipurposeField = subPixelIndex;
    payload.pathContribution = ToRGBe9995(pathContribution);
    payload.pdf = pdf;
    payload.seed = seed;
    payload.blendState = 0u;              // pass 1: opaque search, only flags FOUND
    const float3 blendPwIn = pathWeight;  // throughput entering this segment (scales blend emissive)

    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.Direction = rayDirection;
    ray.TMin = 0;
    ray.TMax = 1.#INF;

    TraceRay(SceneBVH,
        RAY_FLAG_NONE,
        0xFF,
        0, // which hit group to use
        0,
        0, // which miss shader to use (0 = regular, 1 = shadow)
        ray, payload);

    // Pass 2, only when an alphaMode BLEND surface was actually crossed: re-trace
    // bounded by the opaque depth to gather their emissive.  That TMax excludes
    // blend surfaces behind the closest opaque hit, so they can't leak on top of
    // nearer geometry; the gather itself is additive (see ClusterLodAnyHit), so unsorted.
    float  backgroundVisibility = 1.0f;
    float3 blendEmissive = 0.f;
    if ((payload.blendState & RTXMG_BLEND_FOUND) != 0u)
    {
        RayPayload blendPayload = (RayPayload)0;
        blendPayload.blendState    = RTXMG_BLEND_ACCUMULATE;
        blendPayload.backgroundVisibility = 1.0f;
        // Pass 2 ends in a miss (no opaque within TMax), and bounce > 1 keeps that
        // Miss from clearing the primary G-buffer.
        blendPayload.bounce        = 2u;

        RayDesc blendRay = ray;         // same origin/direction as pass 1
        blendRay.TMax = payload.hitT;   // opaque depth (1.#INF on a miss)

        TraceRay(SceneBVH,
            RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
            0xFF,
            0, 0, 0,
            blendRay, blendPayload);

        backgroundVisibility = blendPayload.backgroundVisibility;
        blendEmissive = FromRGBe9995(blendPayload.blendEmissive);
    }

    pathWeight = FromRGBe9995(payload.pathWeight) * backgroundVisibility;
    hitT = payload.hitT;
    seed = payload.seed;
    rayDirection = UnpackNormalizedVector(payload.multipurposeField); // multipurposeField gets re-used for normal
    rayOrigin = payload.rayOrigin;
    float3 pc = FromRGBe9995(payload.pathContribution) * backgroundVisibility;
    if (g_RenderParams.denoiserMode == DenoiserMode::None)
    {
        pathContribution += bounce > 0 ? FireflyFiltering(pc, g_RenderParams.fireflyMaxIntensity) : pc;
    }
    else
    {
        pathContribution += pc;
    }
    pathContribution += blendPwIn * blendEmissive; // deterministic, noise-free
    pdf = payload.pdf;
}

void TraceRadiancePR(float3 rayOrigin, float3 rayDirection, out float3 pathWeight, out float hitT)
{
    RayPayload payload = (RayPayload)0;
    payload.instanceID = ~0u;

    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.Direction = rayDirection;
    ray.TMin = 0;
    ray.TMax = 1.#INF;

    TraceRay(SceneBVH,
        RAY_FLAG_NONE,
        0xFF,
        0, // which hit group to use
        0,
        0, // which miss shader to use (0 = regular, 1 = shadow)
        ray, payload);

    pathWeight = FromRGBe9995(payload.pathWeight);
    hitT = payload.hitT;
}

void TraceRadianceAO(inout uint32_t seed,
                    const uint32_t         subPixelIndex,
                    float3                 rayOrigin,
                    float3                 rayDirection,
                    out float3 pathWeight,
                    out float hitT)
{
    RayPayload payload = (RayPayload)0;
    payload.instanceID = ~0u;
    payload.multipurposeField = subPixelIndex;
    payload.seed = seed;

    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.Direction = rayDirection;
    ray.TMin = 0;
    ray.TMax = 1.#INF;

    TraceRay(SceneBVH,
               RAY_FLAG_NONE,
               0xFF,
               0, // which hit group to use
               0,
               0, // which miss shader to use (0 = regular, 1 = shadow)
               ray, payload);

    pathWeight = FromRGBe9995(payload.pathWeight);
    hitT = payload.hitT;
    seed = payload.seed;
}

uint TimeDiff(uint startTime, uint endTime)
{
    // Account for (at most one) overflow
    return endTime >= startTime ? (endTime - startTime) : (~0u - (startTime - endTime));
}

[shader("raygeneration")]void RayGen()
{
#if defined(TARGET_D3D12)
    // The timer goes through the NVAPI extension UAV, which is null without NVAPI.
    uint startTime = 0;
    if (g_RenderParams.enableTimeView)
        startTime = NvGetSpecial(NV_SPECIALOP_GLOBAL_TIMER_LO);
#endif
    DUMP_FLOAT4(1, float4(0, 0, 0, 1)); // Clear debug buffer
    DUMP_FLOAT4(2, float4(0, 0, 0, 1)); // Clear debug buffer
    DUMP_FLOAT4(3, float4(0, 0, 0, 1)); // Clear debug buffer
    DUMP_FLOAT4(4, float4(0, 0, 0, 1)); // Clear debug buffer

    uint2 pixelPosition = DispatchRaysIndex().xy;
    uint imageIndex = GetImageIndex();

    float3 result = 0;
    float3 diffuseResult = 0;
    float3 specularResult = 0;
    unsigned int seed = TEA(16, imageIndex, g_RenderParams.subFrameIndex);

    float3 rayOrigin, rayDirection;

    float hitT = 1.#INF;

    uint spp = g_RenderParams.denoiserMode != DenoiserMode::None ? 1 : g_RenderParams.spp;

    const uint32_t shflIdxRndOffset = Rnd(seed) * spp;
    const int strataCount = sqrt(spp);

    for (int i = 0; i < spp; i++)
    {
        float2 subPixelJitter = g_RenderParams.denoiserMode != DenoiserMode::None ? g_RenderParams.jitter :
            (RandomStrat(i, strataCount, seed) - 0.5f);
        int subpixelIndex2nd = (i + shflIdxRndOffset) % spp;

        SetupPrimaryRay(pixelPosition, subPixelJitter, rayOrigin, rayDirection);

        float3 pathWeight = 1.f;

        if (g_RenderParams.shadingMode == ShadingMode::PT)
        {
            float3 pathContribution = 0.f;
            float pdf = 0.f;

            for (uint32_t bounce = 0; bounce < g_RenderParams.ptMaxBounces; bounce++)
            {
                TraceRadiancePT(bounce, seed, subpixelIndex2nd,
                    rayOrigin, rayDirection, pathWeight, pathContribution, pdf, hitT);

                if (isinf(hitT) || !any(pathWeight))
                {
                    break;
                }

                if (g_RenderParams.denoiserMode == DenoiserMode::None)
                {
                    // Russian Roulette 
                    if (bounce > 1)
                    {
                        float rrProbability = min(0.95f, Luminance(pathWeight));
                        if (rrProbability < Rnd(seed))
                            break;
                        else
                            pathWeight /= rrProbability;
                    }
                }
            }
            result += pathContribution;
        }
        else if (g_RenderParams.shadingMode == ShadingMode::AO)
        {
            TraceRadianceAO(seed, subpixelIndex2nd, rayOrigin, rayDirection, pathWeight, hitT);
            result += pathWeight;
        }
        else
        {
            TraceRadiancePR(rayOrigin, rayDirection, pathWeight, hitT);
            result += pathWeight;
        }
    }
    result /= spp;

    float4 accumVal;
    if (g_RenderParams.denoiserMode != DenoiserMode::None)
    {
        accumVal = float4(result, 1.f);
    }
    else
    {
        accumVal = u_Accum[pixelPosition];
        accumVal = lerp(accumVal, float4(result, 1.f), 1.f / float(g_RenderParams.subFrameIndex + 1));
    }

#if defined(TARGET_D3D12)
    if (g_RenderParams.enableTimeView)
    {
        uint endTime = NvGetSpecial(NV_SPECIALOP_GLOBAL_TIMER_LO);
        uint deltaTime = TimeDiff(startTime, endTime);

        if (g_RenderParams.subFrameIndex == 0)
        {
            if (pixelPosition.x == 0 && pixelPosition.y == 0)
            {
                u_TimeviewBuffer[0] = 0xffffffff;
                u_TimeviewBuffer[1] = 0;
            }
        }
        else if (g_RenderParams.subFrameIndex == 1)
        {
            int orig;
            InterlockedMin(u_TimeviewBuffer[0], deltaTime, orig);
            InterlockedMax(u_TimeviewBuffer[1], deltaTime, orig);
        }
        else
        {
            uint minValue = u_TimeviewBuffer[0];
            uint maxValue = u_TimeviewBuffer[1];

            float3 result = Temperature(((float)deltaTime) / ((float)maxValue - minValue));
            accumVal = u_Accum[pixelPosition];
            accumVal = lerp(accumVal, float4(result, 1.f), 1.f / float(g_RenderParams.subFrameIndex - 1));
        }
    }
#endif

    u_Accum[pixelPosition] = accumVal;
}

// ---------------------------------------------------------------------------
// Cluster LOD hit shaders
// ---------------------------------------------------------------------------

// Shadow hit — registered as "ClusterLodShadowHitGroup" (hit group index 3 =
// instanceContribution 2 + SBTOffset 1).  The any-hit decides what occludes;
// reaching the closest hit means something did.
[shader("closesthit")]
void ClusterLodShadowClosestHit(inout ShadowRayPayload payload : SV_RayPayload,
                           in Attributes attrib          : SV_IntersectionAttributes)
{
    payload.missed = false;
}

// What occludes a shadow ray: BLEND cards are decals with no opaque substance,
// and MASK needs its cutoff or a leaf card casts the shadow of its whole quad.
// Only non-opaque triangles get here — the CLAS build clears the Opaque geometry
// flag for exactly the alpha materials (Clas_encode*GeometryIndexAndFlags), so
// the hardware skips this shader everywhere else.  Measured at ~0.02 ms of a
// 12.6 ms trace on dense foliage.
[shader("anyhit")]
void ClusterLodShadowAnyHit(inout ShadowRayPayload payload : SV_RayPayload,
                      in    Attributes       attrib  : SV_IntersectionAttributes)
{
    const uint primIdx = PrimitiveIndex();

    const shaderio::RenderInstance inst = t_ClusterLodInstances[InstanceID()];
    const shaderio::Geometry       geom = t_ClusterLodGeometries[inst.geometryID];

    const shaderio::ClusterAddress clusterAddress = t_ResidentClusters[GetClusterID()];
    ByteAddressBuffer groupData =
        ResourceDescriptorHeap[NonUniformResourceIndex(clusterAddress.srvIndex)];

    const shaderio::Cluster cluster =
        groupData.Load<shaderio::Cluster>(clusterAddress.byteOffset);
    const uint triBase  = ClusterGetTriangleMaterialsByteOffset(cluster, clusterAddress.byteOffset);
    const uint localMat = ResolveClusterLocalMaterialID(cluster, groupData, triBase, primIdx);
    const uint matSlot  = ResolveMaterialIDFromLocal(geom, t_ClusterLodLocalMaterialIDs, localMat);
    const RTXMGMaterialConstants clusterLodMaterial = t_MaterialConstants[matSlot];

    if ((clusterLodMaterial.flags & RTXMGMaterialFlags_AlphaBlend) != 0)
    {
        IgnoreHit();
        return;
    }

    if ((clusterLodMaterial.flags & RTXMGMaterialFlags_AlphaMask) == 0)
        return;

    if ((clusterLodMaterial.flags & RTXMGMaterialFlags_UseBaseOrDiffuseTexture) == 0
        || clusterLodMaterial.baseOrDiffuseTextureIndex < 0)
        return;

    float2 uv0, uv1, uv2;
    ClusterLodLoadTriangleTex0(groupData, clusterAddress, primIdx, uv0, uv1, uv2);
    const float3 bary  = float3(1.f - attrib.uv.x - attrib.uv.y, attrib.uv.x, attrib.uv.y);
    const float2 hitUV = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;

    Texture2D<float4> baseTex =
        ResourceDescriptorHeap[NonUniformResourceIndex(clusterLodMaterial.baseOrDiffuseTextureIndex)];
    if (baseTex.SampleLevel(s_MaterialSampler, hitUV, 0).a < clusterLodMaterial.alphaCutoff)
        IgnoreHit();
}

// ---- cluster-LOD debug helpers -------------------------------------------

// The LOD level comes from the Cluster header (cluster.lodLevel).  Do NOT map the
// hit shader's clusterID against the baked per-LOD [clusterOffset, clusterCount)
// ranges: that ID is the allocator-issued scene-global RESIDENT id, not the baked
// per-geometry cluster index, so a range lookup returns garbage.

// Smooth color ramp for VISUALIZE_LOD: white at v=0 (finest), rolling through
// cyan/green/yellow to red at v=1 (coarsest).
static float3 LodMix(float v)
{
    const float low = 0.15f;
    if (v == 0.f)
        return float3(1.f, 1.f, 1.f);

    if (v < low)
    {
        // hue 0.5 = cyan
        const float h = 0.5f;
        float3 cyan = saturate(float3(
            abs(h * 6.f - 3.f) - 1.f,
            2.f - abs(h * 6.f - 2.f),
            2.f - abs(h * 6.f - 4.f)));
        return lerp(float3(1.f, 1.f, 1.f), cyan, v / low);
    }
    else
    {
        v = (v - low) / (1.f - low);
        float h = 0.5f - v * 0.5f;  // cyan → red
        return saturate(float3(
            abs(h * 6.f - 3.f) - 1.f,
            2.f - abs(h * 6.f - 2.f),
            2.f - abs(h * 6.f - 4.f)));
    }
}

// Picks a base color for a cluster-LOD hit based on the active colorMode.
// matSlot comes in already resolved because COLOR_BY_MATERIAL has to show the
// same slot the shading path uses, indirection included.
static float3 ClusterLodEvaluateBaseColor(
    uint                       clusterID,
    uint                       instanceID,
    uint                       geometryID,
    uint                       primitiveIndex,
    float3                     shadingNormal,
    float3                     bary,
    in shaderio::Geometry      geom,
    in shaderio::Cluster       cluster,
    shaderio::ClusterAddress   clusterAddress,
    uint                       localMat,
    uint                       matSlot)
{
    ColorMode colorMode = g_RenderParams.colorMode;

    if (colorMode == ColorMode::COLOR_BY_SHADING_NORMAL)
    {
        return 0.5f * (float3(1.f, 1.f, 1.f) + shadingNormal);
    }
    else if (colorMode == ColorMode::COLOR_BY_CLUSTER_ID)
    {
        uint hash = 0;
        hash = MurmurAdd(hash, geometryID);
        hash = MurmurAdd(hash, clusterID);
        return UintToColor(hash);
    }
    else if (colorMode == ColorMode::COLOR_BY_MICROTRI_ID)
    {
        uint hash = 0;
        hash = MurmurAdd(hash, geometryID);
        hash = MurmurAdd(hash, clusterID);
        hash = MurmurAdd(hash, primitiveIndex);
        return UintToColor(hash);
    }
    else if (colorMode == ColorMode::COLOR_BY_MATERIAL)
    {
        // Hash of the resolved slot, so two surfaces sharing a material also
        // share a colour here.
        return UintToColor(matSlot);
    }
    else if (colorMode == ColorMode::COLOR_BY_CLUSTER_UV)
    {
        // Triangle barycentrics double as a simple per-cluster UV viz.
        return float3(bary.yz, 0.f);
    }
    else if (colorMode == ColorMode::COLOR_BY_TEXCOORD)
    {
        // Handled at the ClusterLodClosestHit call site, which has the interpolated
        // hitUV this runs too early to see; barycentrics keep the other uses defined.
        return float3(bary.yz, 0.f);
    }
    else if (colorMode == ColorMode::COLOR_BY_LOD_LEVEL)
    {
        // Header level normalized by this geometry's max level, then biased
        // into albedo range.
        uint  maxLvl  = max(1u, geom.lodLevelsCount - 1u);
        float v       = float(cluster.lodLevel) / float(maxLvl);
        return LodMix(v) * 0.7f + 0.2f;
    }
    else if (colorMode == ColorMode::COLOR_BY_CLUSTER_GROUP)
    {
        // Hash the GROUP's base address so all its clusters share a color.  Cluster
        // headers are contiguous 16-byte entries (static_assert in shaderio.h), so
        // backing up groupChildIndex of them is a stable per-group identity.
        uint hash = 0;
        hash = MurmurAdd(hash, clusterAddress.srvIndex);
        hash = MurmurAdd(hash, clusterAddress.byteOffset - cluster.groupChildIndex * 16u);
        return UintToColor(hash);
    }
    else if (colorMode == ColorMode::COLOR_BY_BLAS_SOURCE)
    {
        // The low-detail BLAS contains a single cluster (lowDetailClusterID).
        // Hits on that cluster are most likely served from the fallback BLAS;
        // anything else is from a per-frame dynamic BLAS.
        if (clusterID == geom.lowDetailClusterID)
            return float3(0.9f, 0.1f, 0.1f);
        uint hash = 0;
        hash = MurmurAdd(hash, geometryID);
        hash = MurmurAdd(hash, clusterID);
        return UintToColor(hash);
    }
    else if (colorMode == ColorMode::COLOR_BY_BLAS_CACHED)
    {
        // The two static (not-built-this-frame) BLAS sources get distinct colors:
        // blue = the persistent low-detail fallback, green = the per-geometry
        // cached-BLAS pool, red = a per-frame dynamic build.
        if (clusterID == geom.lowDetailClusterID)
            return float3(0.1f, 0.4f, 0.9f);        // low-detail fallback
        bool cached = (geom.cachedBlasLodLevel != shaderio::kTraversalInvalidLodLevel)
                   && (cluster.lodLevel == geom.cachedBlasLodLevel);
        return cached ? float3(0.1f, 0.9f, 0.1f)    // cached-BLAS pool
                      : float3(0.9f, 0.1f, 0.1f);   // dynamic build
    }

    // BASE_COLOR / COLOR_BY_TOPOLOGY / COLOR_BY_SURFACE_INDEX /
    // COLOR_BY_MICROTRI_AREA → white-diffuse default.
    return float3(0.8f, 0.8f, 0.8f);
}

#if CLUSTER_LOD_SHADING >= CLUSTER_LOD_SHADING_NORMAL_MAPPED
// Tangent frame from the hit triangle's own du/dv, so the geometry pool carries no
// baked tangents.  Returns wNormal unchanged when the UVs are degenerate.
float3 ClusterLodPerturbNormal(float2 sampledXY, float scale,
                               float3 p0, float3 p1, float3 p2,
                               float2 uv0, float2 uv1, float2 uv2,
                               float3 wNormal, float3x4 o2w)
{
    const float2 duv1 = uv1 - uv0;
    const float2 duv2 = uv2 - uv0;
    const float  det  = duv1.x * duv2.y - duv2.x * duv1.y;
    if (abs(det) < 1e-16f)
        return wNormal;
    const float r = 1.f / det;

    // dP/du and dP/dv, object space then world (true vectors, so o2w and not its
    // inverse transpose).
    const float3 dp1 = p1 - p0;
    const float3 dp2 = p2 - p0;
    const float3 wTu = mul((float3x3)o2w, (dp1 * duv2.y - dp2 * duv1.y) * r);
    const float3 wTv = mul((float3x3)o2w, (dp2 * duv1.x - dp1 * duv2.x) * r);

    float3 T = wTu - wNormal * dot(wNormal, wTu);
    const float tLen = length(T);
    if (tLen < 1e-8f)
        return wNormal;
    T /= tLen;

    // Orthonormal by construction; dP/dv only supplies the handedness.
    const float3 B = cross(wNormal, T) * (dot(cross(wNormal, T), wTv) < 0.f ? -1.f : 1.f);

    // Z is reconstructed rather than read, so a two-channel (BC5) map and a
    // three-channel one both work.
    const float2 nxy = (sampledXY * 2.f - 1.f) * scale;
    const float  nz  = sqrt(saturate(1.f - dot(nxy, nxy)));
    return normalize(T * nxy.x + B * nxy.y + wNormal * nz);
}
#endif

// Primary hit — shades the cluster LOD triangle with geometric normal + default material.
// Registered as "ClusterLodHitGroup" (hit group index 2 = instanceContribution 2 + SBTOffset 0).
[shader("closesthit")]
void ClusterLodClosestHit(inout RayPayload payload : SV_RayPayload,
                    in Attributes attrib     : SV_IntersectionAttributes)
{
    payload.instanceID     = InstanceID();          // = instanceCustomIndex from TLAS
    payload.primitiveIndex = PrimitiveIndex();
    payload.geometryIndex  = GeometryIndex();
    payload.barycentrics   = attrib.uv;
    payload.hitT           = RayTCurrent();

    const uint clusterID  = GetClusterID();
    const uint instanceID = payload.instanceID;

    shaderio::RenderInstance inst = t_ClusterLodInstances[instanceID];
    shaderio::Geometry       geom = t_ClusterLodGeometries[inst.geometryID];

    // Cluster vertices/indices, out of the same ByteAddressBuffer the CLAS-build
    // hardware reads.  ClusterAddress gives the header address and the bindless
    // heap slot of the buffer holding it; the header carries the data offsets.
    shaderio::ClusterAddress clusterAddress = t_ResidentClusters[clusterID];
    ByteAddressBuffer groupData =
        ResourceDescriptorHeap[NonUniformResourceIndex(clusterAddress.srvIndex)];

    // Positions come from the acceleration structure where available: bit-exact to
    // what the CLAS build consumed (no pool-vs-AS mismatch for SafeSpawnPoint), and
    // what lets the persistent geometry pool drop positions entirely.
    float3 p0, p1, p2;
#if CLUSTER_LOD_AS_POSITION_FETCH
    {
        const float3x3 triPositions = GetHitTriangleObjectPositions();
        p0 = triPositions[0];
        p1 = triPositions[1];
        p2 = triPositions[2];
    }
#else
    ClusterLodLoadTrianglePositions(groupData, clusterAddress, payload.primitiveIndex, p0, p1, p2);
#endif

    float3 bary = float3(1.f - attrib.uv.x - attrib.uv.y, attrib.uv.x, attrib.uv.y);

    // Transforms needed by SafeSpawnPoint().
    float3x4 o2w = ObjectToWorld3x4();
    float3x4 w2o = WorldToObject3x4();

    // Self-intersection-safe spawn point for secondary rays, same as the subd path.
    float3 objP, objN, wldP, wldN;
    float  wldOffset;
    SafeSpawnPoint(objP, wldP, objN, wldN, wldOffset,
                   p0, p1, p2, attrib.uv, o2w, w2o);

    // Ensure the normal faces the incoming ray (back-face friendly).
    if (dot(wldN, WorldRayDirection()) > 0.f)
        wldN = -wldN;

    // Offset along the world-space normal by the floating-point error bound, so the
    // secondary ray can't hit this triangle again.
    float3 wPosSafe = SafeSpawnPoint(wldP, wldN, wldOffset);

    float3 wNormal = wldN;

    // UVs and the material resolve both come before the shading normal, because
    // the normal map needs them and every downstream reader of wShadingNormal
    // (ir.n, COLOR_BY_SHADING_NORMAL) must see the perturbed value.
    float2 uv0, uv1, uv2;
    ClusterLodLoadTriangleTex0(groupData, clusterAddress, payload.primitiveIndex, uv0, uv1, uv2);
    const float2 hitUV = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;

    // Resolve the cluster's local material to a t_MaterialConstants slot: a
    // mixed-material cluster keys off the per-triangle byte, a uniform one off
    // cluster.localMaterialID.
    const shaderio::Cluster cluster =
        groupData.Load<shaderio::Cluster>(clusterAddress.byteOffset);
    const uint triBase   = ClusterGetTriangleMaterialsByteOffset(cluster, clusterAddress.byteOffset);
    const uint localMat  = ResolveClusterLocalMaterialID(cluster, groupData, triBase, payload.primitiveIndex);
    const uint matSlot   = ResolveMaterialIDFromLocal(geom, t_ClusterLodLocalMaterialIDs, localMat);
    const RTXMGMaterialConstants clusterLodMaterial = t_MaterialConstants[matSlot];

    // Interpolate the baked per-vertex normals when present, else stay geometric.
    // ir.gn is geometric either way.
    float3 wShadingNormal = wNormal;
#if CLUSTER_LOD_SHADING >= CLUSTER_LOD_SHADING_VERTEX_NORMALS
    {
        float3 n0, n1, n2;
        ClusterLodLoadTriangleNormals(groupData, clusterAddress, payload.primitiveIndex, n0, n1, n2);
        if (any(n0 != 0.f) || any(n1 != 0.f) || any(n2 != 0.f))
        {
            // object->world for a normal is the inverse-transpose, i.e. a
            // row-vector multiply by worldToObject.
            const float3 objN = bary.x * n0 + bary.y * n1 + bary.z * n2;
            float3 wN = normalize(mul(objN, (float3x3)w2o));
            if (dot(wN, WorldRayDirection()) > 0.f)
                wN = -wN;
            wShadingNormal = wN;
        }
    }
#endif
#if CLUSTER_LOD_SHADING >= CLUSTER_LOD_SHADING_NORMAL_MAPPED
    if ((clusterLodMaterial.flags & RTXMGMaterialFlags_UseNormalTexture) != 0
        && clusterLodMaterial.normalOrDisplacementTextureIndex >= 0)
    {
        Texture2D<float4> normalTex =
            ResourceDescriptorHeap[NonUniformResourceIndex(clusterLodMaterial.normalOrDisplacementTextureIndex)];
        wShadingNormal = ClusterLodPerturbNormal(
            normalTex.SampleLevel(s_MaterialSampler, hitUV, 0).xy,
            clusterLodMaterial.normalOrDisplacementTextureScale,
            p0, p1, p2, uv0, uv1, uv2, wShadingNormal, o2w);
    }
#endif

    // Geometry half of the IntersectionRecord; ir.ms is filled in once the
    // material resolves below.
    IntersectionRecord ir = (IntersectionRecord)0;
    ir.hitT        = payload.hitT;
    ir.p           = wPosSafe;      // nudged spawn point for next bounce / shadow
    ir.n           = wShadingNormal;
    ir.gn          = wNormal;
    ir.barycentrics = bary;
    ir.texcoord    = float2(0.f, 0.f);
    // surfaceIndex = ~0u → motion_vectors treats this as non-subd (camera motion only).
    ir.surfaceIndex = ~0u;
    ir.surfaceUV    = float2(0.f, 0.f);

    // Viewport pick, carrying the geometryID + LOD level that drive the
    // right-click material readout and the Inspector's select-mesh-under-cursor.
    PIXEL_PICK_INIT(g_RenderParams.debugPixel, DispatchRaysIndex().xy);
    PIXEL_PICK_CLUSTER_LOD(instanceID, inst.geometryID, matSlot, cluster.lodLevel);

    // Base color for the visualization modes.  The shading normal goes in so
    // COLOR_BY_SHADING_NORMAL shows the interpolated vertex normal, as subd does.
    float3 baseColor = ClusterLodEvaluateBaseColor(
        clusterID, instanceID, inst.geometryID, payload.primitiveIndex,
        wShadingNormal, bary, geom, cluster, clusterAddress, localMat, matSlot);

    // Also computed for the Inspector selection, which draws as a red wireframe
    // overlay even with global wireframe off.  A viewport pick narrows the
    // selection to one instance and one LOD level; a geometry-row selection
    // leaves both unset and so highlights every instance at every level.
    const bool isSelectedClusterLod = g_RenderParams.selectedClusterLodGeometry >= 0 &&
                                uint(g_RenderParams.selectedClusterLodGeometry) == inst.geometryID &&
                                (g_RenderParams.selectedClusterLodInstance < 0 ||
                                 uint(g_RenderParams.selectedClusterLodInstance) == instanceID) &&
                                (g_RenderParams.selectedClusterLodLevel < 0 ||
                                 uint(g_RenderParams.selectedClusterLodLevel) == cluster.lodLevel);
    float wfWeight = 1.f;
    if (g_RenderParams.enableWireframe || isSelectedClusterLod)
    {
        float3 clipPoints[3];
        GetClipPointsFromObjectSpace(clipPoints, p0, p1, p2, o2w);
        ir.distToEdge = ComputeTriangleDistToEdge(clipPoints, bary);
        wfWeight = WireframeWeight(ir);
    }

    float3 sampledBaseColor = clusterLodMaterial.baseOrDiffuseColor;
    if ((clusterLodMaterial.flags & RTXMGMaterialFlags_UseBaseOrDiffuseTexture) != 0
        && clusterLodMaterial.baseOrDiffuseTextureIndex >= 0)
    {
        Texture2D<float4> diffuseTex =
            ResourceDescriptorHeap[NonUniformResourceIndex(clusterLodMaterial.baseOrDiffuseTextureIndex)];
        const float4 texel = diffuseTex.SampleLevel(s_MaterialSampler, hitUV, 0);
        sampledBaseColor = sampledBaseColor * texel.rgb;
    }

    float3 sampledEmissive = clusterLodMaterial.emissiveColor;
    if ((clusterLodMaterial.flags & RTXMGMaterialFlags_UseEmissiveTexture) != 0
        && clusterLodMaterial.emissiveTextureIndex >= 0)
    {
        Texture2D<float4> emissiveTex =
            ResourceDescriptorHeap[NonUniformResourceIndex(clusterLodMaterial.emissiveTextureIndex)];
        sampledEmissive = sampledEmissive * emissiveTex.SampleLevel(s_MaterialSampler, hitUV, 0).rgb;
    }

    // glTF packs roughness in .g and metalness in .b of one texture, and both
    // multiply the scalar factors.  NOT the subd path's .r convention, which is
    // for separate OBJ grayscale maps.
    float sampledRoughness = clusterLodMaterial.roughness;
    float sampledMetalness = clusterLodMaterial.metalness;
    if ((clusterLodMaterial.flags & RTXMGMaterialFlags_UseMetalnessTexture) != 0
        && clusterLodMaterial.metalnessTextureIndex >= 0)
    {
        Texture2D<float4> mrTex =
            ResourceDescriptorHeap[NonUniformResourceIndex(clusterLodMaterial.metalnessTextureIndex)];
        const float4 mr = mrTex.SampleLevel(s_MaterialSampler, hitUV, 0);
        sampledRoughness *= mr.g;
        sampledMetalness *= mr.b;
    }
    if (g_RenderParams.roughnessOverride > 0.f)
        sampledRoughness = g_RenderParams.roughnessOverride;

    ir.ms = DefaultMaterialSample();
    ir.ms.roughness       = max(sampledRoughness, 1e-4f);
    ir.ms.geometryNormal  = wNormal;
    ir.ms.shadingNormal   = wShadingNormal;
    ir.texcoord           = hitUV;
    // Branch on the colorMode rather than on baseColor's value: ClusterLodEvaluateBaseColor
    // returns a plain 0.8 grey in BASE_COLOR mode, which no value test can tell from
    // a real visualization colour.
    if (g_RenderParams.colorMode == ColorMode::BASE_COLOR)
    {
        ir.ms.baseColor     = sampledBaseColor;
        ir.ms.diffuseAlbedo = sampledBaseColor;
        ir.ms.emissiveColor = sampledEmissive;
        // The BRDF derives diffuse = (1-m)*base and F0 = lerp(0.04, base, m) from
        // these, same as the subd BASE_COLOR path.
        ir.ms.metalness           = sampledMetalness;
        ir.ms.hasMetalRoughParams = true;
    }
    else if (g_RenderParams.colorMode == ColorMode::COLOR_BY_TEXCOORD)
    {
        // Handled here rather than in ClusterLodEvaluateBaseColor, which runs before
        // hitUV exists.  frac() wraps tiled UVs into [0,1), as the subd path does.
        const float3 uvColor = float3(frac(hitUV), 0.f);
        ir.ms.baseColor     = uvColor;
        ir.ms.diffuseAlbedo = uvColor;
    }
    else
    {
        ir.ms.baseColor     = baseColor;
        ir.ms.diffuseAlbedo = baseColor;
    }

    // Inspector selection highlight: a red wireframe, so the shading stays readable
    // between the lines.  Folded into the material sample before the GBuffer write
    // so the DLSS-RR albedo guide carries it and the denoiser keeps it crisp.
    const bool onSelectedWireframe = isSelectedClusterLod && wfWeight == 0.f;
    const float3 selectionRed = float3(1.f, 0.05f, 0.05f);
    if (onSelectedWireframe)
    {
        ir.ms.baseColor     = selectionRed;
        ir.ms.diffuseAlbedo = selectionRed;
        ir.ms.emissiveColor = 0.f;
        ir.ms.metalness     = 0.f;
        ir.ms.hasMetalRoughParams = false;
    }

    // Write GBuffer.
    if (g_RenderParams.denoiserMode != DenoiserMode::None)
    {
        uint2 dispatchPixel = DispatchRaysIndex().xy;
        uint2 dispatchDims  = DispatchRaysDimensions().xy;
        uint  dispatchIndex = dispatchPixel.x + dispatchDims.x * dispatchPixel.y;

        if (g_RenderParams.shadingMode != ShadingMode::PT || payload.bounce == 0)
        {
            HitResult hitResult;
            hitResult.instanceId   = payload.instanceID;
            hitResult.surfaceIndex = ~0u;
            hitResult.surfaceUV    = float2(0.f, 0.f);
            hitResult.texcoord     = float2(0.f, 0.f);
            u_HitResult[dispatchIndex] = hitResult;

            float depth = dot(normalize(g_RenderParams.W), wldP - g_RenderParams.eye);
            u_Depth[dispatchPixel]   = depth;
            u_Normal[dispatchPixel]  = float4(wShadingNormal, 0.f);
            // Metals need the (1-m) diffuse demodulation and an F0-based specular
            // guide, not raw base color + zero spec, so the denoiser guides go
            // through the same FresnelBlend as the subd hit path.
            const float wfGuideWeight = onSelectedWireframe ? 1.f : wfWeight;
            float3 V = -WorldRayDirection();
            FresnelBlend brdf = MakeFresnelBlend(ir.ms.baseColor, ir.ms.specularF0, ir.ms.metalness, ir.ms.roughness);
            u_Albedo[dispatchPixel]  = float4(brdf.m_diffuse.m_albedo * wfGuideWeight, 1.f);
            float3 spec = BRDFEnvApprox(brdf, ir.n, V);
            u_Specular[dispatchPixel] = float4(spec * wfGuideWeight, 1.f);
            u_Roughness[dispatchPixel] = ir.ms.roughness;
        }
        if (g_RenderParams.shadingMode != ShadingMode::PT)
            u_SpecularHitT[dispatchPixel] = g_RenderParams.zFar;
        else if (payload.bounce == 1)
            u_SpecularHitT[dispatchPixel] = ir.hitT;
    }

    // Wireframe: solid black edges — red on the Inspector-selected geometry.
    if (wfWeight == 0.f)
    {
        if (onSelectedWireframe)
        {
            if (g_RenderParams.shadingMode == ShadingMode::PT)
            {
                // Emit the edge as unlit red (throughput-weighted, like an
                // emitter) and terminate the path.
                const float3 pw = FromRGBe9995(payload.pathWeight);
                payload.pathContribution = ToRGBe9995(pw * selectionRed);
                payload.pathWeight = 0;
            }
            else
            {
                // PRIMARY_RAYS / AO: payload.pathWeight IS the output color.
                payload.pathWeight = ToRGBe9995(selectionRed);
            }
            return;
        }
        payload.pathWeight = 0;
        return;
    }

    // Shade.
    if (g_RenderParams.shadingMode == ShadingMode::PRIMARY_RAYS)
    {
        payload.pathWeight = ToRGBe9995(ir.ms.baseColor);
    }
    else if (g_RenderParams.shadingMode == ShadingMode::AO)
    {
        uint32_t seed = payload.seed;
        float2 u = float2(Rnd(seed), Rnd(seed));
        float3 L = AOSample(wNormal, u);
        bool occluded = IsOccluded(ir.p, L);
        payload.pathWeight = ToRGBe9995(ir.ms.baseColor * (occluded ? 0.f : 1.f));
        payload.seed = seed;
    }
    else
    {
        // PT path — share the same ShadeSurface logic.
        float3 pw = FromRGBe9995(payload.pathWeight);
        float3 lightContribution = 0.f;

        // Glass is a delta lobe with no finite BRDF to weight the envmap against,
        // so it is not NEE-sampled; the envmap arrives via the BSDF continuation.
        const bool isTransmissive =
            (clusterLodMaterial.flags & RTXMGMaterialFlags_Transmissive) != 0;

        if (g_RenderParams.hasEnvironmentMap && !isTransmissive)
        {
            uint32_t seed = payload.seed;
            lightContribution = pw * SampleDirect(ir.ms, ir.p, ir.gn, ir.n, -WorldRayDirection(), seed);
            payload.seed = seed;
        }

        // Emitters are not NEE-sampled, so the full add is unbiased — same
        // reasoning as the subd ClosestHit.
        lightContribution += pw * ir.ms.emissiveColor;

        if (payload.bounce < g_RenderParams.ptMaxBounces - 1)
        {
            float3 L = 0.f;
            float  samplePdf = payload.pdf;
            uint32_t seed = payload.seed;
            float3 indirectW;
            if (isTransmissive)
            {
                bool didTransmit;
                indirectW = DielectricSampleThin(ir.n, -WorldRayDirection(),
                                                 clusterLodMaterial.ior, clusterLodMaterial.transmissionFactor,
                                                 ir.ms.baseColor, L, samplePdf, seed, didTransmit);
                // Push the transmit ray past the thin surface so it does not
                // immediately re-hit the same triangle; reflection stays at ir.p.
                payload.rayOrigin = didTransmit ? ir.p + L * max(1e-3f, ir.hitT * 2e-4f) : ir.p;
            }
            else
            {
                indirectW = BRDFSample(ir.ms, ir.gn, ir.n, -WorldRayDirection(), L, samplePdf, seed);
                payload.rayOrigin = ir.p;
            }
            payload.pathWeight       = ToRGBe9995(pw * indirectW);
            payload.multipurposeField = PackNormalizedVector(L);
            payload.pdf              = samplePdf;
            payload.seed             = seed;
        }
        payload.pathContribution = ToRGBe9995(lightContribution);
    }
}

// Cluster-LoD any-hit shader, run on the alpha triangles the CLAS build left
// without the Opaque geometry flag.  Non-alpha materials just
// accept the hit, AlphaMask does the cutoff test, and AlphaBlend accumulates
// emissive*alpha and always IgnoreHit()s so the ray passes through.  Same material
// resolve as ClusterLodClosestHit, minus the positions/barycentrics it doesn't need.
[shader("anyhit")]
void ClusterLodAnyHit(inout RayPayload    payload : SV_RayPayload,
                in    Attributes    attrib  : SV_IntersectionAttributes)
{
    const uint clusterID  = GetClusterID();
    const uint instanceID = InstanceID();
    const uint primIdx    = PrimitiveIndex();

    const shaderio::RenderInstance inst = t_ClusterLodInstances[instanceID];
    const shaderio::Geometry       geom = t_ClusterLodGeometries[inst.geometryID];

    const shaderio::ClusterAddress clusterAddress = t_ResidentClusters[clusterID];
    ByteAddressBuffer groupData =
        ResourceDescriptorHeap[NonUniformResourceIndex(clusterAddress.srvIndex)];

    const shaderio::Cluster cluster =
        groupData.Load<shaderio::Cluster>(clusterAddress.byteOffset);
    const uint triBase  = ClusterGetTriangleMaterialsByteOffset(cluster, clusterAddress.byteOffset);
    const uint localMat = ResolveClusterLocalMaterialID(cluster, groupData, triBase, primIdx);
    const uint matSlot  = ResolveMaterialIDFromLocal(geom, t_ClusterLodLocalMaterialIDs, localMat);
    const RTXMGMaterialConstants clusterLodMaterial = t_MaterialConstants[matSlot];

    const bool isBlend = (clusterLodMaterial.flags & RTXMGMaterialFlags_AlphaBlend) != 0;
    const bool isMask  = (clusterLodMaterial.flags & RTXMGMaterialFlags_AlphaMask)  != 0;

    // Opaque / non-alpha materials: accept the hit (default behaviour).
    if (!isBlend && !isMask)
        return;

    const bool accumulatePass = (payload.blendState & RTXMG_BLEND_ACCUMULATE) != 0u;

    if (isBlend)
    {
        // Pass 1 is the opaque search: flag that a blend surface was crossed and
        // pass through, so the nearest OPAQUE hit still wins.
        if (!accumulatePass)
        {
            payload.blendState |= RTXMG_BLEND_FOUND;
            IgnoreHit();
            return;
        }
        // else: pass 2 — fall through to sample alpha/emissive and accumulate.
    }
    else // isMask
    {
        // Pass 2 treats masks as transparent so they never block the blend search;
        // a mask's solid part is the opaque hit, already excluded by that pass's TMax.
        if (accumulatePass)
        {
            IgnoreHit();
            return;
        }
    }

    // Remaining paths (blend pass 2, mask pass 1) read the base-color alpha.
    if ((clusterLodMaterial.flags & RTXMGMaterialFlags_UseBaseOrDiffuseTexture) == 0
        || clusterLodMaterial.baseOrDiffuseTextureIndex < 0)
    {
        if (accumulatePass) IgnoreHit(); // don't block the blend search
        return;
    }

    // Resolve the hit UV (same texture-resolve path as ClusterLodClosestHit).
    float2 uv0, uv1, uv2;
    ClusterLodLoadTriangleTex0(groupData, clusterAddress, primIdx, uv0, uv1, uv2);
    const float3 bary  = float3(1.f - attrib.uv.x - attrib.uv.y, attrib.uv.x, attrib.uv.y);
    const float2 hitUV = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;

    Texture2D<float4> baseTex =
        ResourceDescriptorHeap[NonUniformResourceIndex(clusterLodMaterial.baseOrDiffuseTextureIndex)];
    const float alpha = baseTex.SampleLevel(s_MaterialSampler, hitUV, 0).a;

    if (isBlend)
    {
        // The gather is a plain SUM, not an over-composite: anyhit order is
        // unspecified, and only a commutative accumulation stays stable when the
        // streaming rebuild reorders traversal.  So alpha applies against the
        // opaque background but not between blend surfaces — a deliberate
        // deviation from strict alphaMode BLEND, traded for determinism.
        float3 emissive = clusterLodMaterial.emissiveColor;
        if ((clusterLodMaterial.flags & RTXMGMaterialFlags_UseEmissiveTexture) != 0
            && clusterLodMaterial.emissiveTextureIndex >= 0)
        {
            Texture2D<float4> emissiveTex =
                ResourceDescriptorHeap[NonUniformResourceIndex(clusterLodMaterial.emissiveTextureIndex)];
            emissive *= emissiveTex.SampleLevel(s_MaterialSampler, hitUV, 0).rgb;
        }
        const float  a     = saturate(alpha);
        const float3 accum = FromRGBe9995(payload.blendEmissive) + a * emissive;
        payload.blendEmissive  = ToRGBe9995(accum);
        payload.backgroundVisibility *= (1.0f - a);  // opaque background only
        IgnoreHit();
        return;
    }

    // Alpha-mask (MASK) pass 1: binary alpha test against the base-color alpha.
    if (alpha < clusterLodMaterial.alphaCutoff)
        IgnoreHit();
}
