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

#include "rtxmg/cluster_lod/cluster_lod_gltf_importer.h"
#include "rtxmg/cluster_lod/baker.h"
#include "rtxmg/cluster_lod/cache.h"
#include "rtxmg/cluster_lod/baking/gltf_file_mapping.h"
#include "rtxmg/cluster_lod/baking/bake_progress.h"
#include "rtxmg/utils/hash.h"
#include "rtxmg/utils/parallel_for.h"
#include "rtxmg/utils/system_memory.h"
#include "rtxmg/utils/verbosity.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <cassert>
#include <cstring>
#include <string>
#include <unordered_map>

#include <donut/core/log.h>

#include <cgltf.h>
#include <meshoptimizer.h>
#include <mimalloc.h>

using namespace donut;
using namespace donut::math;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

// Unique RAII wrapper for cgltf_data — calls cgltf_free on destruction.
using UniqueCgltfPtr = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;

// Route meshoptimizer's scratch allocations through mimalloc's per-thread heaps;
// on the global CRT heap they serialise every parallel bake worker on the heap
// lock.  meshopt_setAllocator scopes this to meshopt, so no other module's
// new/delete pairing changes.
void* MeshoptMiAlloc(size_t size) { return mi_malloc(size); }
void  MeshoptMiFree(void* ptr)    { mi_free(ptr); }

void EnsureMeshoptScalableAllocator()
{
    // Must run before any meshopt call on the bake threads, so Load() calls it
    // from the main thread.
    static const bool once = [] {
        meshopt_setAllocator(MeshoptMiAlloc, MeshoptMiFree);
        return true;
    }();
    (void)once;
}

// Read a single accessor into a float[] destination, handling non-float
// sources via cgltf_accessor_read_float.  Returns false if any element could
// not be read, leaving the destination partially written.
//
// T        — element type: float3 (positions/normals), float2 (UVs), float4 (tangents).
// doBBox   — if true, update *bboxLo / *bboxHi with element values.
template<typename T, bool doBBox>
inline bool ReadAttributesGLTF(const cgltf_accessor* accessor,
                                float*                dst,
                                size_t                dstStride,
                                cgltf_type            expectedType,
                                T*                    bboxLo = nullptr,
                                T*                    bboxHi = nullptr)
{
    constexpr size_t kFloats = sizeof(T) / sizeof(float);

    // Fast path: source is already packed float3/float2 with no gaps.  A sparse
    // accessor's overrides live outside the view, and the view is null until a
    // meshopt decode fills it in, so neither may take it.
    const uint8_t* packed = nullptr;
    if (accessor->component_type == cgltf_component_type_r_32f
        && accessor->type         == expectedType
        && accessor->stride       == sizeof(T)
        && !accessor->is_sparse
        && accessor->buffer_view)
    {
        packed = cgltf_buffer_view_data(accessor->buffer_view);
    }

    if (packed)
    {
        const T* src = reinterpret_cast<const T*>(packed + accessor->offset);
        for (size_t i = 0; i < accessor->count; ++i)
        {
            const T& val = src[i];
            *reinterpret_cast<T*>(&dst[i * dstStride]) = val;
            if constexpr (doBBox)
            {
                *bboxLo = min(*bboxLo, val);
                *bboxHi = max(*bboxHi, val);
            }
        }
    }
    else
    {
        for (size_t i = 0; i < accessor->count; ++i)
        {
            // Returns 0 without writing `val` for a sparse accessor or an
            // undecoded view; the value would otherwise be stack garbage.
            T val{};
            if (!cgltf_accessor_read_float(accessor, i, &val.x, kFloats))
                return false;
            *reinterpret_cast<T*>(&dst[i * dstStride]) = val;
            if constexpr (doBBox)
            {
                *bboxLo = min(*bboxLo, val);
                *bboxHi = max(*bboxHi, val);
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Import-warning rate limiting.  These fire once per mesh, and the partial-
// triangle one once per primitive, so a systematically bad export buries the
// log.  Log the first few of each kind and tally the rest into one line.
// ---------------------------------------------------------------------------

enum ImportWarningKind
{
    kWarnPartialTriangle,
    kWarnNoTriangles,
    kWarnTooManyMaterials,
    kWarnUnreadableAttribute,
    kWarnUnreadableIndices,
    kWarnIndexOutOfRange,
    kWarnKindCount
};

const char* const kImportWarningNames[kWarnKindCount] = {
    "primitive(s) with an index count that is not a multiple of 3",
    "mesh(es) with no indexed triangles",
    "mesh(es) over the unique-material limit",
    "mesh(es) with an unreadable attribute",
    "mesh(es) with unreadable indices",
    "mesh(es) with an out-of-range index",
};

constexpr size_t          kMaxWarningsPerKind = 4;
std::atomic<size_t>       g_importWarnings[kWarnKindCount];

// True for the first kMaxWarningsPerKind of each kind; runs on bake workers.
bool ShouldLogImportWarning(ImportWarningKind kind)
{
    return g_importWarnings[kind].fetch_add(1, std::memory_order_relaxed) < kMaxWarningsPerKind;
}

void ResetImportWarnings()
{
    for (std::atomic<size_t>& c : g_importWarnings)
        c.store(0, std::memory_order_relaxed);
}

void ReportSuppressedImportWarnings()
{
    for (int k = 0; k < kWarnKindCount; ++k)
    {
        const size_t n = g_importWarnings[k].load(std::memory_order_relaxed);
        if (n > kMaxWarningsPerKind)
            log::warning("ClusterLodGltfImporter: %zu further %s (%zu total, first %zu logged).",
                         n - kMaxWarningsPerKind, kImportWarningNames[k], n, kMaxWarningsPerKind);
    }
}

// ---------------------------------------------------------------------------
// LoadGeometry — extract one mesh's vertices/indices into a GeometryStorage.
//
// Returns false when the mesh yields nothing bakeable; `geometry` is then left
// empty and the caller must drop it rather than bake and cache it.  Warning,
// never error: donut pops a modal box for Error/Fatal and this runs on parallel
// bake workers.
// ---------------------------------------------------------------------------

static bool LoadGeometry(const cgltf_mesh& gltfMesh,
                         const cgltf_data* gltf,
                         GeometryStorage&  geometry)
{
    const char* const meshName = gltfMesh.name ? gltfMesh.name : "<unnamed>";
    // ---- Pass 1: count totals and determine which attributes are present ----
    uint32_t totalVertices  = 0;
    uint32_t totalTriangles = 0;

    for (size_t primIdx = 0; primIdx < gltfMesh.primitives_count; ++primIdx)
    {
        const cgltf_primitive& prim = gltfMesh.primitives[primIdx];
        if (prim.type != cgltf_primitive_type_triangles || prim.attributes_count == 0)
            continue;

        for (size_t a = 0; a < prim.attributes_count; ++a)
        {
            const cgltf_attribute& attr = prim.attributes[a];
            if (strcmp(attr.name, "POSITION") == 0)
                totalVertices += (uint32_t)attr.data->count;
            else if (strcmp(attr.name, "NORMAL") == 0)
                geometry.attributeBits |= shaderio::ClusterAttribute::VertexNormal;
            else if (strcmp(attr.name, "TANGENT") == 0)
                geometry.attributeBits |= shaderio::ClusterAttribute::VertexTangent;
            else if (strcmp(attr.name, "TEXCOORD_0") == 0)
                geometry.attributeBits |= shaderio::ClusterAttribute::VertexTex0;
            else if (strcmp(attr.name, "TEXCOORD_1") == 0)
                geometry.attributeBits |= shaderio::ClusterAttribute::VertexTex1;
        }

        if (prim.indices)
        {
            // cgltf_validate has no count % 3 check, and this sizing floors, so
            // the fill below must copy whole triangles only.
            if (prim.indices->count % 3 != 0 && ShouldLogImportWarning(kWarnPartialTriangle))
                log::warning("ClusterLodGltfImporter: mesh '%s' primitive %zu has %zu indices, "
                             "not a multiple of 3; dropping the trailing partial triangle.",
                             meshName, primIdx, size_t(prim.indices->count));
            totalTriangles += (uint32_t)(prim.indices->count / 3);
        }
    }

    if (totalVertices == 0 || totalTriangles == 0)
    {
        if (ShouldLogImportWarning(kWarnNoTriangles))
            log::warning("ClusterLodGltfImporter: mesh '%s' has no indexed triangles "
                         "(%u vertices, %u triangles); skipping it.",
                         meshName, totalVertices, totalTriangles);
        return false;
    }

    // Tangents require both normals and tex coords.
    if (!(geometry.attributeBits & shaderio::ClusterAttribute::VertexNormal))
        geometry.attributeBits &= ~shaderio::ClusterAttribute::VertexTangent;
    if (!(geometry.attributeBits & shaderio::ClusterAttribute::VertexTex0))
        geometry.attributeBits &= ~shaderio::ClusterAttribute::VertexTangent;
    // TEX_1 requires TEX_0.
    if (!(geometry.attributeBits & shaderio::ClusterAttribute::VertexTex0))
        geometry.attributeBits &= ~shaderio::ClusterAttribute::VertexTex1;

    // ---- Multi-material discovery: dedupe the primitives' materials into
    // localMaterialIDs[local] = glTF-global index, in first-seen order.  The
    // local index is what the baker stores as Cluster::localMaterialID.  A
    // primitive with no material maps to the ~0u sentinel.
    auto materialGlobalIndex = [&](const cgltf_material* mat) -> uint32_t {
        if (!mat || !gltf->materials)
            return ~0u;
        return uint32_t(mat - gltf->materials);
    };

    geometry.localMaterialIDs.clear();
    bool materialOverflow = false;
    auto findOrAddLocalMaterial = [&](uint32_t globalID) -> uint8_t {
        for (size_t i = 0; i < geometry.localMaterialIDs.size(); ++i)
            if (geometry.localMaterialIDs[i] == globalID)
                return uint8_t(i);
        // The per-triangle material byte reserves bit 6 for two-sided and bit 7
        // for alpha-masked, so an ID past the mask aliases into those flags.
        if (geometry.localMaterialIDs.size() >= shaderio::kMaxLocalMaterials)
        {
            materialOverflow = true;
            return uint8_t(0);
        }
        geometry.localMaterialIDs.push_back(globalID);
        return uint8_t(geometry.localMaterialIDs.size() - 1);
    };

    std::vector<uint8_t> primLocalIDPerPrim;
    primLocalIDPerPrim.reserve(gltfMesh.primitives_count);
    for (size_t primIdx = 0; primIdx < gltfMesh.primitives_count; ++primIdx)
    {
        const cgltf_primitive& prim = gltfMesh.primitives[primIdx];
        if (prim.type != cgltf_primitive_type_triangles || prim.attributes_count == 0)
        {
            primLocalIDPerPrim.push_back(0);
            continue;
        }
        const uint32_t globalID = materialGlobalIndex(prim.material);
        primLocalIDPerPrim.push_back(findOrAddLocalMaterial(globalID));
    }
    if (materialOverflow && ShouldLogImportWarning(kWarnTooManyMaterials))
        log::warning("ClusterLodGltfImporter: mesh '%s' uses more than %u unique materials; "
                     "the excess render with the mesh's first material.",
                     meshName, uint32_t(shaderio::kMaxLocalMaterials));
    const bool isMultiMaterial = geometry.localMaterialIDs.size() > 1;

    // ---- Compute per-vertex attribute stride and offsets -------------------
    // Layout, each block a tight run of floats per vertex:
    //   [NORMAL(3)] [TANGENT(4)] [TEX0(2)] [TEX1(2)] [MATERIAL(1)]
    // The material ID goes last so adding it doesn't shift the other offsets.
    size_t   stride     = 0;
    uint32_t offsetEnd  = 0;

    if (geometry.attributeBits & shaderio::ClusterAttribute::VertexNormal)
    {
        geometry.attributeNormalOffset = offsetEnd;
        offsetEnd += 3;
        stride    += 3;
    }
    if (geometry.attributeBits & shaderio::ClusterAttribute::VertexTangent)
    {
        geometry.attributeTangentOffset = offsetEnd;
        offsetEnd += 4;
        stride    += 4;
    }
    if (geometry.attributeBits & shaderio::ClusterAttribute::VertexTex0)
    {
        geometry.attributeTex0offset = offsetEnd;
        offsetEnd += 2;
        stride    += 2;
    }
    if (geometry.attributeBits & shaderio::ClusterAttribute::VertexTex1)
    {
        geometry.attributeTex1offset = offsetEnd;
        offsetEnd += 2;
        stride    += 2;
    }
    if (isMultiMaterial)
    {
        geometry.attributeMaterialOffset = offsetEnd;
        offsetEnd += 1;
        stride    += 1;
    }
    // attributesWithWeights is the simplifier's attribute_count: 0 keeps it on
    // positions only, otherwise it sees the whole attribute block and baker.cpp
    // weights each offset.  Textured single-material meshes must be included too,
    // or edge collapses cross UV seams and coarse LoDs lose their texturing.
    const bool weightAttributes =
        isMultiMaterial || (geometry.attributeBits & shaderio::ClusterAttribute::VertexTex0);
    geometry.attributesWithWeights = weightAttributes ? uint32_t(stride) : 0u;

    // ---- Allocate storage --------------------------------------------------
    geometry.vertexPositions.resize(totalVertices);
    geometry.vertexAttributes.resize(totalVertices * stride, 0.f);
    geometry.triangles.resize(totalTriangles);

    // ---- Pass 2: fill data -------------------------------------------------
    uint32_t vertexOffset   = 0;
    uint32_t triangleOffset = 0;

    for (size_t primIdx = 0; primIdx < gltfMesh.primitives_count; ++primIdx)
    {
        const cgltf_primitive& prim = gltfMesh.primitives[primIdx];
        if (prim.type != cgltf_primitive_type_triangles || prim.attributes_count == 0)
            continue;

        uint32_t numVertices = 0;
        // Tracked for the missing-normal synthesis below.
        bool           primHasNormal = false;
        const uint32_t primTriStart  = triangleOffset;

        // A failed read leaves uninitialized bytes in the destination, which
        // would be baked and cached, so the whole mesh is dropped instead.
        bool readOk = true;
        for (size_t a = 0; a < prim.attributes_count && readOk; ++a)
        {
            const cgltf_attribute& attr = prim.attributes[a];

            if (strcmp(attr.name, "POSITION") == 0)
            {
                float3* dst = geometry.vertexPositions.data() + vertexOffset;
                readOk = ReadAttributesGLTF<float3, true>(
                    attr.data,
                    reinterpret_cast<float*>(dst), 3,
                    cgltf_type_vec3,
                    &geometry.bbox.lo, &geometry.bbox.hi);
                numVertices = (uint32_t)attr.data->count;
            }
            else if (strcmp(attr.name, "NORMAL") == 0)
            {
                primHasNormal = true;
                if (geometry.attributeBits & shaderio::ClusterAttribute::VertexNormal)
                {
                    float* dst = geometry.vertexAttributes.data()
                        + vertexOffset * stride + geometry.attributeNormalOffset;
                    readOk = ReadAttributesGLTF<float3, false>(
                        attr.data, dst, stride, cgltf_type_vec3);
                }
            }
            else if (strcmp(attr.name, "TANGENT") == 0
                     && (geometry.attributeBits & shaderio::ClusterAttribute::VertexTangent))
            {
                float* dst = geometry.vertexAttributes.data()
                    + vertexOffset * stride + geometry.attributeTangentOffset;
                readOk = ReadAttributesGLTF<float4, false>(
                    attr.data, dst, stride, cgltf_type_vec4);
            }
            else if (strcmp(attr.name, "TEXCOORD_0") == 0
                     && (geometry.attributeBits & shaderio::ClusterAttribute::VertexTex0))
            {
                float* dst = geometry.vertexAttributes.data()
                    + vertexOffset * stride + geometry.attributeTex0offset;
                readOk = ReadAttributesGLTF<float2, false>(
                    attr.data, dst, stride, cgltf_type_vec2);
            }
            else if (strcmp(attr.name, "TEXCOORD_1") == 0
                     && (geometry.attributeBits & shaderio::ClusterAttribute::VertexTex1))
            {
                float* dst = geometry.vertexAttributes.data()
                    + vertexOffset * stride + geometry.attributeTex1offset;
                readOk = ReadAttributesGLTF<float2, false>(
                    attr.data, dst, stride, cgltf_type_vec2);
            }
        }

        if (!readOk)
        {
            if (ShouldLogImportWarning(kWarnUnreadableAttribute))
                log::warning("ClusterLodGltfImporter: mesh '%s' primitive %zu has an attribute "
                             "this importer cannot read (sparse accessor, or a buffer view that "
                             "failed to decode); skipping the mesh.",
                             meshName, primIdx);
            geometry = {};
            return false;
        }

        // Indices — always uint32 in our storage; re-base by vertexOffset.
        if (prim.indices)
        {
            uint32_t* dst = reinterpret_cast<uint32_t*>(
                geometry.triangles.data() + triangleOffset);
            const cgltf_accessor* idx = prim.indices;
            // Sized to whole triangles, exactly like pass 1's floor.
            const size_t triCount   = idx->count / 3;
            const size_t indexCount = triCount * 3;

            // cgltf_accessor_read_index silently returns 0 for both cases, so
            // without this the mesh bakes as one degenerate triangle fan.
            if (idx->is_sparse || !idx->buffer_view
                || !cgltf_buffer_view_data(idx->buffer_view))
            {
                if (ShouldLogImportWarning(kWarnUnreadableIndices))
                    log::warning("ClusterLodGltfImporter: mesh '%s' primitive %zu has indices this "
                                 "importer cannot read (sparse accessor, or a buffer view that "
                                 "failed to decode); skipping the mesh.",
                                 meshName, primIdx);
                geometry = {};
                return false;
            }

            if (vertexOffset == 0
                && idx->component_type == cgltf_component_type_r_32u
                && idx->type           == cgltf_type_scalar
                && idx->stride         == sizeof(uint32_t))
            {
                // Fast path: already uint32, tightly packed.
                std::memcpy(dst,
                    static_cast<const uint8_t*>(cgltf_buffer_view_data(idx->buffer_view))
                    + idx->offset,
                    sizeof(uint32_t) * indexCount);
            }
            else
            {
                for (size_t i = 0; i < indexCount; ++i)
                    dst[i] = (uint32_t)cgltf_accessor_read_index(prim.indices, i)
                             + vertexOffset;
            }

            // cgltf_validate's index-bound check is guarded on the buffer already
            // being loaded, and both paths validate before loading, so this is the
            // only thing between a bad index and an OOB read in the baker.
            uint32_t maxIndex = 0;
            for (size_t i = 0; i < indexCount; ++i)
                if (dst[i] > maxIndex)
                    maxIndex = dst[i];
            if (indexCount != 0 && maxIndex >= vertexOffset + numVertices)
            {
                if (ShouldLogImportWarning(kWarnIndexOutOfRange))
                    log::warning("ClusterLodGltfImporter: mesh '%s' primitive %zu indexes vertex %u "
                                 "of %u; skipping the mesh.",
                                 meshName, primIdx, maxIndex - vertexOffset, numVertices);
                geometry = {};  // half-filled arrays must not reach the baker
                return false;
            }

            triangleOffset += (uint32_t)triCount;
        }

        // glTF assigns materials per primitive but the baker reads them per
        // vertex, so stamp this primitive's local ID across its vertex range.
        // Where a vertex is shared with a differently-materialled primitive the
        // last write wins, and the baker's per-triangle mismatch check catches it.
        if (isMultiMaterial && numVertices > 0)
        {
            const float localIDFloat = float(primLocalIDPerPrim[primIdx]);
            float*       dst         = geometry.vertexAttributes.data()
                                       + vertexOffset * stride
                                       + geometry.attributeMaterialOffset;
            for (uint32_t v = 0; v < numVertices; ++v)
                dst[v * stride] = localIDFloat;
        }

        // Synthesize normals for a NORMAL-less primitive in a mixed mesh: its
        // slots are still zero, and the cluster advertises the NORMAL bit, so the
        // shader's geometric fallback is bypassed and oct(0,0,0) would shade every
        // LoD dark.  A mesh with no authored normals at all never sets the bit and
        // is correctly left to that fallback.
        if (!primHasNormal
            && (geometry.attributeBits & shaderio::ClusterAttribute::VertexNormal)
            && numVertices > 0)
        {
            float* const nrmBase = geometry.vertexAttributes.data()
                                   + geometry.attributeNormalOffset;
            for (uint32_t t = primTriStart; t < triangleOffset; ++t)
            {
                const uint3&   tri  = geometry.triangles[t];
                const uint32_t iv[3] = { tri.x, tri.y, tri.z };
                const float3&  p0   = geometry.vertexPositions[iv[0]];
                const float3&  p1   = geometry.vertexPositions[iv[1]];
                const float3&  p2   = geometry.vertexPositions[iv[2]];
                const float3   fn   = cross(p1 - p0, p2 - p0); // area-weighted, CCW-outward
                for (int k = 0; k < 3; ++k)
                {
                    float* d = nrmBase + iv[k] * stride;
                    d[0] += fn.x; d[1] += fn.y; d[2] += fn.z;
                }
            }
            // A degenerate (zero-area) vertex gets a finite default so
            // pack/unpack stays well-defined.
            for (uint32_t v = vertexOffset; v < vertexOffset + numVertices; ++v)
            {
                float*      d   = nrmBase + v * stride;
                const float3 n  = float3(d[0], d[1], d[2]);
                const float len = length(n);
                const float3 nn = (len > 1e-20f) ? (n / len) : float3(0.f, 0.f, 1.f);
                d[0] = nn.x; d[1] = nn.y; d[2] = nn.z;
            }
        }

        vertexOffset += numVertices;
    }

    return true;
}

// ---------------------------------------------------------------------------
// TraverseNodes — walk the GLTF scene graph, collecting ClusterLodInstances.
// ---------------------------------------------------------------------------

// Column-major 4x4 = T * R * S from glTF-convention inputs (quat as x,y,z,w).
// Matches cgltf_node_transform_local's composition for a TRS node.
static void ComposeTrsColMajor(const float t[3], const float q[4], const float s[3],
                               float m[16])
{
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;

    m[0]  = (1.f - 2.f * (yy + zz)) * s[0];
    m[1]  = (2.f * (xy + wz)) * s[0];
    m[2]  = (2.f * (xz - wy)) * s[0];
    m[3]  = 0.f;
    m[4]  = (2.f * (xy - wz)) * s[1];
    m[5]  = (1.f - 2.f * (xx + zz)) * s[1];
    m[6]  = (2.f * (yz + wx)) * s[1];
    m[7]  = 0.f;
    m[8]  = (2.f * (xz + wy)) * s[2];
    m[9]  = (2.f * (yz - wx)) * s[2];
    m[10] = (1.f - 2.f * (xx + yy)) * s[2];
    m[11] = 0.f;
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
    m[15] = 1.f;
}

// out = a * b for column-major 4x4 matrices.
static void MulColMajor(const float a[16], const float b[16], float out[16])
{
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
        {
            float sum = 0.f;
            for (int k = 0; k < 4; ++k)
                sum += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = sum;
        }
}

// cgltf's column-major float[16] and donut's row-major, ROW-VECTOR float4x4 are
// the same 16 floats in the same order (both put the translation last), so this
// is a straight copy — and the result composes with homogeneousToAffine().
static void SetInstanceTransform(ClusterLodInstance& inst, const float colMajor[16])
{
    std::memcpy(&inst.transform, colMajor, 16 * sizeof(float));
}

static void TraverseNodes(const std::vector<size_t>& meshToGeometry,
                          const cgltf_data*          gltf,
                          const cgltf_node*          node,
                          std::vector<ClusterLodInstance>& instances)
{
    if (!node)
        return;

    if (node->mesh)
    {
        const ptrdiff_t meshIndex = node->mesh - gltf->meshes;
        if (meshIndex >= 0 && (size_t)meshIndex < meshToGeometry.size()
            && meshToGeometry[meshIndex] != size_t(-1))
        {
            float raw[16];  // column-major world matrix
            cgltf_node_transform_world(node, raw);

            ClusterLodInstance inst;
            inst.geometryID = (uint32_t)meshToGeometry[meshIndex];
            inst.materialID = 0;
            if (node->mesh->primitives_count > 0
                && node->mesh->primitives[0].material)
            {
                inst.materialID = (uint32_t)(
                    node->mesh->primitives[0].material - gltf->materials);
            }
            if (node->name)
                inst.name = node->name;
            else if (node->mesh->name)
                inst.name = node->mesh->name;

            if (node->has_mesh_gpu_instancing)
            {
                // EXT_mesh_gpu_instancing: the node itself is NOT rendered, only
                // its instance array, each at world = node * T*R*S.  Load()
                // pre-mapped the buffers these accessors read.
                const cgltf_accessor* accT = nullptr;
                const cgltf_accessor* accR = nullptr;
                const cgltf_accessor* accS = nullptr;
                const cgltf_mesh_gpu_instancing& gi = node->mesh_gpu_instancing;
                for (size_t a = 0; a < gi.attributes_count; ++a)
                {
                    const cgltf_attribute& attr = gi.attributes[a];
                    if      (strcmp(attr.name, "TRANSLATION") == 0) accT = attr.data;
                    else if (strcmp(attr.name, "ROTATION")    == 0) accR = attr.data;
                    else if (strcmp(attr.name, "SCALE")       == 0) accS = attr.data;
                }
                const cgltf_accessor* any = accT ? accT : (accR ? accR : accS);
                const size_t instanceCount = any ? any->count : 0;

                for (size_t i = 0; i < instanceCount; ++i)
                {
                    float t[3] = { 0.f, 0.f, 0.f };
                    float q[4] = { 0.f, 0.f, 0.f, 1.f };
                    float s[3] = { 1.f, 1.f, 1.f };
                    if (accT) cgltf_accessor_read_float(accT, i, t, 3);
                    if (accR) cgltf_accessor_read_float(accR, i, q, 4);
                    if (accS) cgltf_accessor_read_float(accS, i, s, 3);

                    float local[16], world[16];
                    ComposeTrsColMajor(t, q, s, local);
                    MulColMajor(raw, local, world);
                    SetInstanceTransform(inst, world);
                    instances.push_back(inst);
                }
            }
            else
            {
                SetInstanceTransform(inst, raw);
                instances.push_back(inst);
            }
        }
    }

    for (size_t i = 0; i < node->children_count; ++i)
        TraverseNodes(meshToGeometry, gltf, node->children[i], instances);
}

// ---------------------------------------------------------------------------
// Shard cache helpers (per-geometry content-hash cache)
// ---------------------------------------------------------------------------

// The shard cache's fast gate, computed without reading any buffer bytes.
// sourceKey is the manifest key; sumFileSize + maxMtime catch in-place edits
// to it without re-hashing.
struct SourceIdentity
{
    uint64_t sourceKey   = 0;
    uint64_t sumFileSize = 0;
    int64_t  maxMtime    = 0;
};

// The storage a bufferView's bytes actually come from.  For a meshopt-compressed
// view that is the compressed source, NOT bufferView.buffer — the latter is the
// uri-less fallback buffer, which every mesh gets its own copy of starting at
// offset 0, so it carries no mesh identity at all.
struct ViewSource
{
    const cgltf_buffer* buffer = nullptr;
    uint64_t            offset = 0;
    uint64_t            size   = 0;
};

ViewSource GetViewSource(const cgltf_buffer_view* bv)
{
    if (bv->has_meshopt_compression)
        return { bv->meshopt_compression.buffer,
                 uint64_t(bv->meshopt_compression.offset),
                 uint64_t(bv->meshopt_compression.size) };
    return { bv->buffer, uint64_t(bv->offset), uint64_t(bv->size) };
}

// The container file a cgltf buffer is loaded from: the external .bin for a
// file URI, or the gltf/glb itself for an embedded / data: buffer.
fs::path ResolveBufferFile(const cgltf_buffer* buf, const fs::path& gltfPath)
{
    if (buf && buf->uri && strncmp(buf->uri, "data:", 5) != 0)
    {
        std::string uri(buf->uri);
        cgltf_decode_uri(uri.data());
        uri.resize(strlen(uri.c_str()));
        return gltfPath.parent_path() / fs::path(uri);
    }
    return gltfPath;  // embedded / .glb binary chunk
}

// The exact path string cgltf_load_buffers passes to the file-read callback
// for a buffer uri: mirror cgltf_combine_paths (gltf path up to its last
// separator + uri) then cgltf_decode_uri on the appended part.  Byte-identical
// matching matters — FileMappingList::subsetNames filters by raw string.
std::string CgltfBufferCallbackPath(const std::string& gltfFilePath, const char* uri)
{
    const size_t slash  = gltfFilePath.find_last_of("/\\");
    std::string  result = (slash == std::string::npos)
                            ? std::string()
                            : gltfFilePath.substr(0, slash + 1);
    const size_t uriStart = result.size();
    result += uri;
    cgltf_decode_uri(result.data() + uriStart);
    result.resize(strlen(result.c_str()));
    return result;
}

// Visit each distinct buffer file referenced by a mesh's accessors.
template <typename Fn>
void ForEachMeshBufferFile(const cgltf_mesh& mesh, const fs::path& gltfPath, Fn&& fn)
{
    std::unordered_set<std::string> seen;
    auto consider = [&](const cgltf_accessor* acc) {
        if (!acc || !acc->buffer_view)
            return;
        const ViewSource src = GetViewSource(acc->buffer_view);
        if (!src.buffer)
            return;
        fs::path p = ResolveBufferFile(src.buffer, gltfPath);
        if (seen.insert(p.string()).second)
            fn(p);
    };
    for (size_t pi = 0; pi < mesh.primitives_count; ++pi)
    {
        const cgltf_primitive& prim = mesh.primitives[pi];
        consider(prim.indices);
        for (size_t ai = 0; ai < prim.attributes_count; ++ai)
            consider(prim.attributes[ai].data);
    }
}

// FNV-1a over the sorted set of (buffer uri, view offset, view size) a mesh's
// accessors touch, plus its per-primitive material assignment — gltf-independent,
// so two gltfs referencing the same mesh data share a manifest entry, while two
// meshes in one .bin differ by offset.  Reads only gltf metadata, never faults
// the .bin bytes in.  Resolved through GetViewSource(), without which every
// meshopt mesh keys on an empty uri at offset 0 and any two with equal view
// sizes silently share a shard.
uint64_t ComputeSourceKey(const cgltf_mesh& mesh, const cgltf_data* gltf)
{
    std::set<std::tuple<std::string, uint64_t, uint64_t>> views;
    auto consider = [&](const cgltf_accessor* acc) {
        if (!acc || !acc->buffer_view)
            return;
        const ViewSource src = GetViewSource(acc->buffer_view);
        if (!src.buffer)
            return;
        std::string uri;
        if (src.buffer->uri && strncmp(src.buffer->uri, "data:", 5) != 0)
        {
            uri = src.buffer->uri;
            cgltf_decode_uri(uri.data());
            uri.resize(strlen(uri.c_str()));
        }
        views.emplace(std::move(uri), src.offset, src.size);
    };
    for (size_t pi = 0; pi < mesh.primitives_count; ++pi)
    {
        const cgltf_primitive& prim = mesh.primitives[pi];
        consider(prim.indices);
        for (size_t ai = 0; ai < prim.attributes_count; ++ai)
            consider(prim.attributes[ai].data);
    }
    uint64_t h = rtxmg::kFnv1aOffsetBasis;
    for (const auto& [uri, off, size] : views)
    {
        h = rtxmg::Fnv1a(uri.data(), uri.size(), h);
        h = rtxmg::Fnv1aValue(off, h);
        h = rtxmg::Fnv1aValue(size, h);
    }

    // A manifest hit skips LoadGeometry and copies the shard's material tables
    // back, so the key has to cover the assignment as well as the bytes: two
    // sibling gltfs over one .bin otherwise share an entry and the second
    // renders with the first's glTF-global material indices and state bits.
    for (size_t pi = 0; pi < mesh.primitives_count; ++pi)
    {
        const cgltf_primitive& prim = mesh.primitives[pi];
        if (prim.type != cgltf_primitive_type_triangles || prim.attributes_count == 0)
            continue;
        const cgltf_material* mat = prim.material;
        h = rtxmg::Fnv1aValue((mat && gltf->materials) ? uint32_t(mat - gltf->materials) : ~0u, h);
        h = rtxmg::Fnv1aValue(uint32_t(mat ? mat->alpha_mode   : 0), h);
        h = rtxmg::Fnv1aValue(uint32_t(mat ? mat->double_sided : 0), h);
    }
    return h;
}

SourceIdentity ComputeSourceIdentity(const cgltf_mesh& mesh, const cgltf_data* gltf,
                                     const fs::path& gltfPath)
{
    SourceIdentity id;
    id.sourceKey = ComputeSourceKey(mesh, gltf);
    ForEachMeshBufferFile(mesh, gltfPath, [&](const fs::path& p) {
        std::error_code ec;
        auto sz = fs::file_size(p, ec);
        if (!ec)
            id.sumFileSize += uint64_t(sz);
        auto t = fs::last_write_time(p, ec);
        if (!ec)
        {
            int64_t ticks = int64_t(t.time_since_epoch().count());
            if (ticks > id.maxMtime)
                id.maxMtime = ticks;
        }
    });
    return id;
}

// Fill localMaterialStateBits (ClusterState AlphaMasked / TwoSided) from the
// model's materials — an input the baker aggregates into cluster/group bits.
void ComputeStateBits(GeometryStorage& geo, const std::vector<ClusterLodMaterialDesc>& materials)
{
    geo.localMaterialStateBits.assign(geo.localMaterialIDs.size(), uint8_t(0));
    for (size_t li = 0; li < geo.localMaterialIDs.size(); ++li)
    {
        const uint32_t globalID = geo.localMaterialIDs[li];
        if (globalID >= materials.size())
            continue;
        const ClusterLodMaterialDesc& m = materials[globalID];
        uint8_t bits = 0;
        if (m.alphaModeGltf != 0) bits |= uint8_t(shaderio::ClusterState::AlphaMasked);
        if (m.doubleSided)        bits |= uint8_t(shaderio::ClusterState::TwoSided);
        geo.localMaterialStateBits[li] = bits;
    }
}

// Offset-independent content hash of a geometry's baker inputs.  Must be called
// after LoadGeometry but before Build (which clears vertexPositions/triangles).
uint64_t HashGeometryInput(const GeometryStorage& s)
{
    uint64_t h = rtxmg::kFnv1aOffsetBasis;
    h = rtxmg::Fnv1aValue(s.attributeBits, h);
    h = rtxmg::Fnv1aValue(s.attributeNormalOffset, h);
    h = rtxmg::Fnv1aValue(s.attributeTex0offset, h);
    h = rtxmg::Fnv1aValue(s.attributeTex1offset, h);
    h = rtxmg::Fnv1aValue(s.attributeTangentOffset, h);
    h = rtxmg::Fnv1aValue(s.attributeMaterialOffset, h);
    auto hashVec = [&](const auto& v) {
        uint64_t n = uint64_t(v.size());
        h = rtxmg::Fnv1a(&n, sizeof(n), h);
        if (n)
            h = rtxmg::Fnv1a(v.data(), v.size() * sizeof(v[0]), h);
    };
    hashVec(s.vertexPositions);
    hashVec(s.vertexAttributes);
    hashVec(s.triangles);
    hashVec(s.localMaterialIDs);
    hashVec(s.localMaterialStateBits);
    return h;
}

std::string HashToHex(uint64_t h)
{
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return std::string(buf);
}

// One index per cache folder, shared by every gltf in it, so warm loads validate
// by size+mtime instead of re-hashing every input.
struct ManifestEntry
{
    uint64_t contentHash = 0;
    uint64_t sourceSize  = 0;
    int64_t  sourceMtime = 0;
};

// On-disk record = key + value (in-memory it's an unordered_map).
struct ManifestRecord
{
    uint64_t      sourceKey = 0;
    ManifestEntry entry;
};

struct ManifestFileHeader
{
    uint64_t magic       = 0x736e6764696e6d76ULL;  // shard-manifest magic
    uint32_t version     = 3;                      // v3: configHash + headerSize
    // The records follow the header, so a header that grew or shrank silently
    // reads them at the wrong offset unless the size is checked.
    uint32_t headerSize  = uint32_t(sizeof(ManifestFileHeader));
    uint64_t recordCount = 0;
    uint64_t configHash  = 0;   // validated: BakerConfig::SemanticHash()
    // Diagnostic only, never compared: lets a re-bake name the fields that
    // changed instead of just announcing that something did.
    BakerConfig config;
};

// "field was->now, field was->now" over the differing fields, for the re-bake
// warning.  Empty when the two structs agree.
std::string DescribeConfigDifferences(const BakerConfig& was, const BakerConfig& now)
{
    std::string s;
    char        buf[128];
    auto sep = [&]() -> const char* { return s.empty() ? "" : ", "; };
    auto addU = [&](const char* name, uint64_t a, uint64_t b) {
        if (a == b) return;
        std::snprintf(buf, sizeof(buf), "%s%s %llu->%llu", sep(), name,
                      (unsigned long long)a, (unsigned long long)b);
        s += buf;
    };
    auto addF = [&](const char* name, float a, float b) {
        if (a == b) return;
        std::snprintf(buf, sizeof(buf), "%s%s %g->%g", sep(), name, double(a), double(b));
        s += buf;
    };

    addU("clusterVertices",           was.clusterVertices,           now.clusterVertices);
    addU("clusterTriangles",          was.clusterTriangles,          now.clusterTriangles);
    addU("clusterGroupSize",          was.clusterGroupSize,          now.clusterGroupSize);
    addU("preferredNodeWidth",        was.preferredNodeWidth,        now.preferredNodeWidth);
    addF("lodErrorMergePrevious",     was.lodErrorMergePrevious,     now.lodErrorMergePrevious);
    addF("lodErrorMergeAdditive",     was.lodErrorMergeAdditive,     now.lodErrorMergeAdditive);
    addF("lodErrorEdgeLimit",         was.lodErrorEdgeLimit,         now.lodErrorEdgeLimit);
    addU("meshoptPreferRayTracing",   was.meshoptPreferRayTracing,   now.meshoptPreferRayTracing);
    addF("meshoptFillWeight",         was.meshoptFillWeight,         now.meshoptFillWeight);
    addF("meshoptSplitFactor",        was.meshoptSplitFactor,        now.meshoptSplitFactor);
    addU("useCompressedData",         was.useCompressedData,         now.useCompressedData);
    addU("compressionPosDropBits",    was.compressionPosDropBits,    now.compressionPosDropBits);
    addU("compressionTexDropBits",    was.compressionTexDropBits,    now.compressionTexDropBits);
    addF("simplifyNormalWeight",      was.simplifyNormalWeight,      now.simplifyNormalWeight);
    addF("simplifyTexCoordWeight",    was.simplifyTexCoordWeight,    now.simplifyTexCoordWeight);
    addF("simplifyTangentWeight",     was.simplifyTangentWeight,     now.simplifyTangentWeight);
    addF("simplifyTangentSignWeight", was.simplifyTangentSignWeight, now.simplifyTangentSignWeight);
    addF("simplifyMaterialWeight",    was.simplifyMaterialWeight,    now.simplifyMaterialWeight);
    addU("quantizeTexCoords",         was.quantizeTexCoords,         now.quantizeTexCoords);
    return s;
}

// The manifest header's copy of the previous bake settings.  It is the only
// record of them anywhere on disk, so the caller reports the diff once it knows
// how many shards it actually invalidated.
struct ManifestConfigInfo
{
    bool        valid   = false;  // header parsed
    bool        matches = false;  // configHash equals the current config
    BakerConfig config;
};

bool LoadManifest(const fs::path& path, const BakerConfig& config,
                  std::unordered_map<uint64_t, ManifestEntry>& out,
                  ManifestConfigInfo& outInfo)
{
    out.clear();
    outInfo = {};
    std::error_code sizeEc;
    const uint64_t  fileSize = uint64_t(fs::file_size(path, sizeEc));
    if (sizeEc || fileSize < sizeof(ManifestFileHeader))
        return false;

    FILE* f = fopen(path.string().c_str(), "rb");
    if (!f)
        return false;

    ManifestFileHeader hdr{};
    ManifestFileHeader ref{};
    const bool headerOk = fread(&hdr, sizeof(hdr), 1, f) == 1
           && hdr.magic == ref.magic && hdr.version == ref.version
           && hdr.headerSize == ref.headerSize
           // A corrupt recordCount would otherwise size a vector the file
           // cannot possibly hold.
           && hdr.recordCount <= (fileSize - sizeof(hdr)) / sizeof(ManifestRecord);

    // Every value in a record — contentHash, sourceSize, sourceMtime — is
    // config-independent, and each shard's own header gates whether its bytes
    // are usable.  So a config change keeps the map and only costs a re-bake of
    // the shards that actually disagree.
    bool ok = headerOk;
    if (headerOk)
    {
        outInfo.valid   = true;
        outInfo.config  = hdr.config;
        outInfo.matches = hdr.configHash == config.SemanticHash();
    }

    if (ok && hdr.recordCount > 0)
    {
        std::vector<ManifestRecord> recs(hdr.recordCount);
        ok = fread(recs.data(), sizeof(ManifestRecord), hdr.recordCount, f) == hdr.recordCount;
        if (ok)
        {
            out.reserve(hdr.recordCount);
            for (const ManifestRecord& r : recs)
                out.emplace(r.sourceKey, r.entry);
        }
    }
    fclose(f);
    if (!ok)
        out.clear();
    return ok;
}

bool SaveManifest(const fs::path& path, const BakerConfig& config,
                  const std::unordered_map<uint64_t, ManifestEntry>& entries)
{
    fs::path tmp = path;
    tmp += ".tmp";

    FILE* f = fopen(tmp.string().c_str(), "wb");
    if (!f)
        return false;

    std::vector<ManifestRecord> recs;
    recs.reserve(entries.size());
    for (const auto& [key, e] : entries)
        recs.push_back(ManifestRecord{ key, e });

    ManifestFileHeader hdr;
    hdr.recordCount = recs.size();
    hdr.configHash  = config.SemanticHash();
    hdr.config      = config;
    bool ok = fwrite(&hdr, sizeof(hdr), 1, f) == 1;
    if (ok && !recs.empty())
        ok = fwrite(recs.data(), sizeof(ManifestRecord), recs.size(), f) == recs.size();
    fclose(f);

    std::error_code ec;
    if (!ok)
    {
        fs::remove(tmp, ec);
        return false;
    }
    fs::rename(tmp, path, ec);
    if (ec)
    {
        fs::remove(path, ec);
        fs::rename(tmp, path, ec);
    }
    return !ec;
}

// ---------------------------------------------------------------------------
// EXT_meshopt_compression decode
//
// cgltf parses the extension's metadata but does not decode it, so
// cgltf_buffer_view_data() returns null until we fill buffer_view->data.
// Decoding is just-in-time per geometry and released
// right after extraction, bounding the decompressed working set to the
// geometries in flight.
// ---------------------------------------------------------------------------

// Grow-only decode storage, one scratch per bake worker.  A per-geometry
// malloc/free of blocks this size serialises every worker on the heap lock,
// because freeing decommits the pages under the heap's critical section.
struct MeshoptDecodeScratch
{
    std::vector<std::vector<uint8_t>> buffers;   // grow-only reused storage
    size_t                            used = 0;  // buffers handed out this mesh
    std::vector<cgltf_buffer_view*>   views;     // views pointed at our storage

    uint8_t* Take(size_t bytes)
    {
        if (used >= buffers.size())
            buffers.emplace_back();
        std::vector<uint8_t>& b = buffers[used++];
        if (b.size() < bytes)
            b.resize(bytes);  // grow-only: reuses capacity across geometries
        return b.data();
    }

    // After extraction: detach the decoded views (so cgltf_free and the next
    // mesh never touch our storage) but KEEP the buffers for reuse.
    void Release()
    {
        for (cgltf_buffer_view* bv : views)
            bv->data = nullptr;
        views.clear();
        used = 0;
    }

    // cgltf_free frees every buffer_view->data, so a view left attached to our
    // storage is a double free.
    ~MeshoptDecodeScratch() { Release(); }
};

// Decode bv's meshopt payload into `dst`, which must hold count * stride bytes.
bool DecodeMeshoptViewInto(const cgltf_buffer_view* bv, void* dst)
{
    const cgltf_meshopt_compression* mc = &bv->meshopt_compression;
    const unsigned char* source = static_cast<const unsigned char*>(mc->buffer->data);
    if (!source)
        return false;
    source += mc->offset;

    int rc = -1;
    switch (mc->mode)
    {
        case cgltf_meshopt_compression_mode_attributes:
            rc = meshopt_decodeVertexBuffer(dst, mc->count, mc->stride, source, mc->size);
            break;
        case cgltf_meshopt_compression_mode_triangles:
            rc = meshopt_decodeIndexBuffer(dst, mc->count, mc->stride, source, mc->size);
            break;
        case cgltf_meshopt_compression_mode_indices:
            rc = meshopt_decodeIndexSequence(dst, mc->count, mc->stride, source, mc->size);
            break;
        default:
            break;
    }
    if (rc != 0)
        return false;

    switch (mc->filter)
    {
        case cgltf_meshopt_compression_filter_octahedral:
            meshopt_decodeFilterOct(dst, mc->count, mc->stride);
            break;
        case cgltf_meshopt_compression_filter_quaternion:
            meshopt_decodeFilterQuat(dst, mc->count, mc->stride);
            break;
        case cgltf_meshopt_compression_filter_exponential:
            meshopt_decodeFilterExp(dst, mc->count, mc->stride);
            break;
        default:
            break;
    }
    return true;
}

bool DecodeOneMeshoptView(cgltf_buffer_view* bv, MeshoptDecodeScratch& scratch)
{
    const cgltf_meshopt_compression* mc = &bv->meshopt_compression;

    void* result = scratch.Take(mc->count * mc->stride);
    if (!DecodeMeshoptViewInto(bv, result))
        return false;

    bv->data = result;
    scratch.views.push_back(bv);
    return true;
}

// Decode the meshopt-compressed buffer views one mesh references into `scratch`.
template <typename F>
void ForEachMeshoptView(const cgltf_mesh& mesh, F&& fn)
{
    auto handle = [&](const cgltf_accessor* acc) {
        if (acc && acc->buffer_view && acc->buffer_view->has_meshopt_compression)
            fn(acc->buffer_view);
    };
    for (size_t pi = 0; pi < mesh.primitives_count; ++pi)
    {
        const cgltf_primitive& prim = mesh.primitives[pi];
        handle(prim.indices);
        for (size_t ai = 0; ai < prim.attributes_count; ++ai)
            handle(prim.attributes[ai].data);
    }
}

// Call scratch.release() after extracting the mesh (LoadGeometry).
bool DecodeMeshoptViewsForMesh(const cgltf_mesh& mesh, MeshoptDecodeScratch& scratch)
{
    bool ok = true;
    ForEachMeshoptView(mesh, [&](cgltf_buffer_view* bv) {
        if (bv->data)
            return;  // already decoded, for this mesh or into the shared scratch
        if (!DecodeOneMeshoptView(bv, scratch))
            ok = false;
    });
    return ok;
}

// Every bake worker holds its own parsed cgltf_data, a decoded-mesh scratch, the
// extracted geometry and the baker's working set at once, and largest-first
// ordering puts the biggest meshes in flight together -- so peak RAM scales with
// the worker count and nothing else.
uint32_t BakeWorkerCount(uint32_t requested)
{
    const uint32_t cores = std::max(1u, std::thread::hardware_concurrency());
    if (requested > 0)
        return requested;

    rtxmg::SystemMemory mem;
    if (!rtxmg::QuerySystemMemory(mem))
        return cores;

    // Leave the source mmap and the rest of the process room to breathe.
    constexpr uint64_t kReserveBytes   = 8ull << 30;
    constexpr uint64_t kBytesPerWorker = 2ull << 30;

    const uint64_t total = mem.totalBytes;
    if (total <= kReserveBytes)
        return 1;

    const uint64_t budget  = (total - kReserveBytes) / kBytesPerWorker;
    const uint32_t workers = uint32_t(std::clamp<uint64_t>(budget, 1, cores));
    if (workers < cores)
        log::info("cluster_lod: capping bake workers at %u of %u cores (%.1f GB RAM).",
                  workers, cores, double(total) / double(1ull << 30));
    return workers;
}

// Remove geometries that produced no LOD hierarchy, along with the instances
// that reference them: a zero-level geometry underflows `lodLevelsCount - 1` at
// every consumer.  Reachable from a mesh the baker cannot build, and from an
// already-cached shard of one.
size_t DropEmptyGeometries(ClusterLodModel& model, std::vector<size_t>& geometryToMesh)
{
    const size_t          geoCount = model.geometries.size();
    std::vector<uint32_t> remap(geoCount, ~0u);
    uint32_t              kept = 0;
    for (size_t gi = 0; gi < geoCount; ++gi)
        if (model.geometries[gi].lodLevelsCount != 0)
            remap[gi] = kept++;

    if (kept == geoCount)
        return 0;

    for (size_t gi = 0; gi < geoCount; ++gi)
    {
        if (remap[gi] == ~0u)
            continue;
        model.geometries[remap[gi]] = std::move(model.geometries[gi]);
        model.storages[remap[gi]]   = std::move(model.storages[gi]);
        geometryToMesh[remap[gi]]   = geometryToMesh[gi];
    }
    model.geometries.resize(kept);
    model.storages.resize(kept);
    geometryToMesh.resize(kept);

    size_t droppedInstances = 0;
    auto   dead = std::remove_if(model.instances.begin(), model.instances.end(),
                                 [&](const ClusterLodInstance& inst) {
                                     return inst.geometryID >= geoCount
                                         || remap[inst.geometryID] == ~0u;
                                 });
    droppedInstances = size_t(model.instances.end() - dead);
    model.instances.erase(dead, model.instances.end());
    for (ClusterLodInstance& inst : model.instances)
        inst.geometryID = remap[inst.geometryID];

    log::warning("ClusterLodGltfImporter: dropped %zu geometry(ies) with no LOD "
                 "hierarchy and %zu instance(s) referencing them.",
                 geoCount - kept, droppedInstances);
    return geoCount - kept;
}

// ---------------------------------------------------------------------------
// Load stages
// ---------------------------------------------------------------------------

// Parses and validates the file into `gltf`.  `loadBuffers` faults every buffer
// in up front; the shard path leaves it to BakeOrLoadShards, which loads only
// what the stale geometries need — and nothing at all on a full cache hit.  The
// dedup stage needs only accessor pointers, which parse already gave us.
bool ParseGltfFile(const std::string& filePath, cgltf_options& options,
                   bool loadBuffers, UniqueCgltfPtr& gltf)
{
    cgltf_data*  rawData = nullptr;
    cgltf_result result  = cgltf_parse_file(&options, filePath.c_str(), &rawData);
    gltf.reset(rawData);

    if (result == cgltf_result_legacy_gltf)
    {
        log::error("ClusterLodGltfImporter: '%s' is a legacy glTF 1.0 file; "
                   "only glTF 2.0 is supported.", filePath.c_str());
        return false;
    }
    if (result != cgltf_result_success || !gltf)
    {
        log::error("ClusterLodGltfImporter: cgltf_parse_file failed for '%s' "
                   "(result %d).", filePath.c_str(), (int)result);
        return false;
    }

    result = cgltf_validate(gltf.get());
    if (result != cgltf_result_success)
    {
        log::error("ClusterLodGltfImporter: cgltf_validate failed for '%s' "
                   "(result %d).", filePath.c_str(), (int)result);
        return false;
    }

    if (loadBuffers)
    {
        result = cgltf_load_buffers(&options, gltf.get(), filePath.c_str());
        if (result != cgltf_result_success)
        {
            log::error("ClusterLodGltfImporter: cgltf_load_buffers failed for '%s' "
                       "(result %d). Are the referenced buffer paths valid?",
                       filePath.c_str(), (int)result);
            return false;
        }
    }
    return true;
}

// Meshes sharing the same accessor pointers are the same vertex data (they
// differ only in material or transform), so they share one GeometryStorage.
void DedupMeshesToGeometries(const cgltf_data*    gltf,
                             std::vector<size_t>& meshToGeometry,
                             std::vector<size_t>& geometryToMesh)
{
    meshToGeometry.assign(gltf->meshes_count, size_t(-1));

    // Key: serialised accessor pointer values → geometry index.
    std::unordered_map<std::string, size_t> keyToGeometry;
    keyToGeometry.reserve(gltf->meshes_count);

    for (size_t mi = 0; mi < gltf->meshes_count; ++mi)
    {
        const cgltf_mesh& m = gltf->meshes[mi];
        std::string key;
        key.reserve(64 * m.primitives_count);

        for (size_t pi = 0; pi < m.primitives_count; ++pi)
        {
            const cgltf_primitive& prim = m.primitives[pi];
            if (prim.type != cgltf_primitive_type_triangles)
                continue;

            const void* posPtr  = nullptr;
            const void* normPtr = nullptr;
            const void* uvPtr   = nullptr;
            const void* idxPtr  = prim.indices;

            for (size_t ai = 0; ai < prim.attributes_count; ++ai)
            {
                const cgltf_attribute& attr = prim.attributes[ai];
                if      (strcmp(attr.name, "POSITION")   == 0) posPtr  = attr.data;
                else if (strcmp(attr.name, "NORMAL")     == 0) normPtr = attr.data;
                else if (strcmp(attr.name, "TEXCOORD_0") == 0) uvPtr   = attr.data;
            }

            char buf[128];
            std::snprintf(buf, sizeof(buf), "%p,%p,%p,%p,",
                          posPtr, normPtr, idxPtr, uvPtr);
            key += buf;
        }

        auto [it, inserted] = keyToGeometry.emplace(key, geometryToMesh.size());
        if (inserted)
        {
            meshToGeometry[mi] = geometryToMesh.size();
            geometryToMesh.push_back(mi);
        }
        else
        {
            meshToGeometry[mi] = it->second;
        }
    }
}

// EXT_mesh_gpu_instancing: a warm shard run skips cgltf_load_buffers entirely,
// so map just the buffers the instancing accessors reference.  cgltf skips
// already-loaded buffers, so BakeOrLoadShards' later load composes with this one.
bool PreloadGpuInstancingBuffers(cgltf_data* gltf, const std::string& filePath,
                                 cgltf_options& options, rtxmg::FileMappingList& mappings,
                                 bool useShardCache)
{
    bool hasGpuInstancing = false;
    for (size_t ni = 0; ni < gltf->nodes_count; ++ni)
    {
        const cgltf_node& node = gltf->nodes[ni];
        if (!node.has_mesh_gpu_instancing)
            continue;
        hasGpuInstancing = true;
        for (size_t ai = 0; ai < node.mesh_gpu_instancing.attributes_count; ++ai)
        {
            const cgltf_accessor* acc = node.mesh_gpu_instancing.attributes[ai].data;
            if (!acc || !acc->buffer_view)
                continue;
            // cgltf leaves meshopt-compressed views undecoded, so the TRS
            // reads below would see no data.
            if (acc->buffer_view->has_meshopt_compression)
            {
                log::error("ClusterLodGltfImporter: meshopt-compressed "
                           "EXT_mesh_gpu_instancing accessors are not supported ('%s').",
                           filePath.c_str());
                return false;
            }
            const cgltf_buffer* buf = acc->buffer_view->buffer;
            if (buf && buf->uri && strncmp(buf->uri, "data:", 5) != 0)
                mappings.subsetNames.insert(CgltfBufferCallbackPath(filePath, buf->uri));
        }
    }

    if (hasGpuInstancing && useShardCache)
    {
        // (The monolith path already loaded every buffer above.)
        cgltf_result result = cgltf_load_buffers(&options, gltf, filePath.c_str());
        mappings.subsetNames.clear();
        if (result != cgltf_result_success)
        {
            log::error("ClusterLodGltfImporter: cgltf_load_buffers failed for "
                       "EXT_mesh_gpu_instancing buffers of '%s' (result %d).",
                       filePath.c_str(), (int)result);
            return false;
        }
    }
    else
    {
        mappings.subsetNames.clear();
    }
    return true;
}

// Walks the default scene, or every root node when the file declares no scene.
void CollectSceneInstances(const cgltf_data* gltf, const std::vector<size_t>& meshToGeometry,
                           std::vector<ClusterLodInstance>& instances)
{
    if (gltf->scenes_count > 0)
    {
        const cgltf_scene& scene = gltf->scene ? *gltf->scene : gltf->scenes[0];
        for (size_t ni = 0; ni < scene.nodes_count; ++ni)
            TraverseNodes(meshToGeometry, gltf, scene.nodes[ni], instances);
    }
    else
    {
        for (size_t ni = 0; ni < gltf->nodes_count; ++ni)
        {
            if (!gltf->nodes[ni].parent)
                TraverseNodes(meshToGeometry, gltf, &gltf->nodes[ni], instances);
        }
    }
}

void ParseMaterials(const cgltf_data* gltf, const fs::path& gltfDir,
                    std::vector<ClusterLodMaterialDesc>& materials)
{
    // Returns the absolute path of a texture URI, or empty string for embedded/missing images.
    auto resolveTexUri = [&gltfDir](const cgltf_texture_view& tv) -> std::string
    {
        if (!tv.texture || !tv.texture->image || !tv.texture->image->uri)
            return {};
        // glTF image URIs are percent-encoded, so decode before resolving (cgltf
        // does the same for buffer/mesh URIs) or the lookup keeps the literal %2F.
        std::string uri = tv.texture->image->uri;
        cgltf_decode_uri(uri.data());
        uri = uri.c_str();  // truncate at the decoded (earlier) null terminator
        return (gltfDir / uri).lexically_normal().string();
    };

    materials.reserve(gltf->materials_count);
    for (size_t mi = 0; mi < gltf->materials_count; ++mi)
    {
        const cgltf_material& m = gltf->materials[mi];
        ClusterLodMaterialDesc desc;
        if (m.name) desc.name = m.name;

        if (m.has_pbr_metallic_roughness)
        {
            const auto& pbr = m.pbr_metallic_roughness;
            desc.baseColorFactor = {
                pbr.base_color_factor[0], pbr.base_color_factor[1],
                pbr.base_color_factor[2], pbr.base_color_factor[3] };
            desc.metalness = pbr.metallic_factor;
            desc.roughness = pbr.roughness_factor;
            desc.baseColorTexturePath          = resolveTexUri(pbr.base_color_texture);
            desc.metallicRoughnessTexturePath  = resolveTexUri(pbr.metallic_roughness_texture);
        }

        float3 emissive = { m.emissive_factor[0], m.emissive_factor[1], m.emissive_factor[2] };
        if (m.has_emissive_strength)
            emissive = emissive * m.emissive_strength.emissive_strength;
        float intensity = std::max({ emissive.x, emissive.y, emissive.z });
        if (intensity > 0.f) { desc.emissiveColor = emissive / intensity; desc.emissiveIntensity = intensity; }
        desc.emissiveTexturePath = resolveTexUri(m.emissive_texture);

        desc.normalTexturePath  = resolveTexUri(m.normal_texture);
        desc.normalTextureScale = m.normal_texture.texture ? m.normal_texture.scale : 1.f;

        desc.alphaCutoff  = m.alpha_cutoff;
        desc.doubleSided  = m.double_sided;
        desc.alphaModeGltf = (int)m.alpha_mode;
        // KHR_materials_transmission / KHR_materials_ior (cgltf parses these).
        if (m.has_transmission) desc.transmissionFactor = m.transmission.transmission_factor;
        desc.ior = m.has_ior ? m.ior.ior : 1.5f;
        materials.push_back(desc);
    }
}

// Verbose per-geometry stats, unified for the bake and cache-hit paths.
void LogGeometryStats(const cgltf_data* gltf, const std::vector<size_t>& geometryToMesh,
                      const ClusterLodModel& model)
{
    const size_t loadedCount = model.geometries.size();
    for (size_t gi = 0; gi < loadedCount; ++gi)
    {
        const char* meshName = gltf->meshes[geometryToMesh[gi]].name;
        if (!meshName) meshName = "<unnamed>";

        const GeometryView& geo = model.geometries[gi];
        log::info("  [%zu/%zu] '%s': %u LOD levels, %u total clusters "
                  "(L0: %u clusters, %u triangles, %u vertices)",
                  gi + 1, loadedCount, meshName,
                  geo.lodLevelsCount, geo.totalClustersCount,
                  geo.hiClustersCount, geo.hiTriangleCount, geo.hiVerticesCount);

        for (uint32_t L = 0; L < geo.lodLevelsCount && L < (uint32_t)geo.lodLevels.size(); ++L)
            log::info("    L%u: %u groups, %u clusters",
                      L, geo.lodLevels[L].groupCount, geo.lodLevels[L].clusterCount);
    }
}

// ---------------------------------------------------------------------------
// Shard bake worker
// ---------------------------------------------------------------------------

// Everything one shard-bake worker needs.  The per-geometry outputs are indexed
// by geometry index, matching BakeOrLoadShards' arrays.
struct ShardBakeContext
{
    // One parsed cgltf_data per worker: buffer_view->data is where cgltf hands
    // back a decoded meshopt view, so a private parse is what makes the per-mesh
    // decode race-free without decoding shared views up front.
    const std::vector<UniqueCgltfPtr>& workerGltf;
    const std::vector<size_t>&         geometryToMesh;
    const std::vector<SourceIdentity>& ids;
    const BakerConfig&                 bakerConfig;
    ClusterLodBaker&                   baker;
    const fs::path&                    cacheDir;
    uint64_t                           configHash;
    bool                               log;
    size_t                             missCount;
    std::atomic<size_t>&               doneCount;
    std::mutex&                        logMutex;
    ClusterLodModel&                   model;
    std::vector<uint64_t>&             contentHash;
    std::vector<fs::path>&             shardPath;
    std::vector<uint8_t>&              isEmpty;
    std::vector<uint8_t>&              bakedInStorage;
};

// Extracts, hashes and bakes ONE geometry, freeing both its decompressed input
// and its baked output so the private working set stays bounded by the
// in-flight geometries rather than the whole scene.
void BakeOneGeometry(ShardBakeContext& c, size_t gi, uint32_t worker)
{
    const cgltf_data* gltf = c.workerGltf[worker].get();
    const cgltf_mesh& mesh = gltf->meshes[c.geometryToMesh[gi]];

    rtxmg::GetBakeProgress().SetInFlight(worker, mesh.name ? mesh.name : "<unnamed>");

    thread_local MeshoptDecodeScratch decodeScratch;
    // LoadGeometry fails on its own against an undecoded view; this
    // only names the cause.
    if (!DecodeMeshoptViewsForMesh(mesh, decodeScratch))
        log::warning("ClusterLodGltfImporter: EXT_meshopt_compression decode failed for "
                     "mesh '%s'.", mesh.name ? mesh.name : "<unnamed>");
    const bool extracted = LoadGeometry(mesh, gltf, c.model.storages[gi]);
    decodeScratch.Release();

    if (!extracted)
    {
        c.model.storages[gi] = {};
        c.isEmpty[gi]        = 1;
        rtxmg::GetBakeProgress().CompleteOne(worker);
        return;
    }

    ComputeStateBits(c.model.storages[gi], c.model.materials);
    c.contentHash[gi] = HashGeometryInput(c.model.storages[gi]);
    c.shardPath[gi]   = c.cacheDir / (HashToHex(c.contentHash[gi]) + ".shard");

    const uint32_t tris  = (uint32_t)c.model.storages[gi].triangles.size();
    const uint32_t verts = (uint32_t)c.model.storages[gi].vertexPositions.size();

    // A valid shard already named by this content hash means the bytes
    // are unchanged (only mtime moved, or another geometry is identical),
    // so the expensive bake can be skipped.
    ShardHeader hdr;
    const bool reuse = fs::exists(c.shardPath[gi])
                    && ShardCache::ReadHeader(c.shardPath[gi], hdr)
                    && hdr.configHash == c.configHash;

    if (!reuse)
    {
        c.baker.Build(c.model.storages[gi]);
        GeometryView view = MakeGeometryView(c.model.storages[gi]);
        // Never persist a geometry the baker could not build a hierarchy
        // for: the shard is structurally valid and config-matching, so
        // every later run hits it and underflows lodLevelsCount - 1.
        if (view.lodLevelsCount == 0)
        {
            log::warning("ClusterLodGltfImporter: mesh '%s' baked no LOD levels; "
                         "not caching it.", mesh.name ? mesh.name : "<unnamed>");
            c.model.storages[gi] = {};
            c.isEmpty[gi]        = 1;
        }
        else if (!SaveShardAtomic(c.shardPath[gi], view, c.bakerConfig,
                             c.contentHash[gi], c.ids[gi].sumFileSize, c.ids[gi].maxMtime))
        {
            log::warning("ClusterLodGltfImporter: failed to write shard '%s'; "
                         "keeping baked geometry in RAM.",
                         c.shardPath[gi].string().c_str());
            c.bakedInStorage[gi] = 1;
        }
        else
        {
            // Free the baked output and let step 5 map the shard we just
            // wrote, exactly like a cache hit.  Retaining it would keep a
            // cold bake's whole output in private RAM rather than in the
            // evictable, page-cache-backed mmap a warm load gets.
            c.model.storages[gi] = {};
        }
    }
    else
    {
        // Step 5 maps the existing shard and re-copies the material tables
        // from it, so drop the raw input here rather than letting it
        // accumulate across the whole miss list.
        c.model.storages[gi] = {};
    }

    // "Reused" means the content hash already had a valid shard, so nothing was
    // baked; the counter still advances so the [d/missCount] index stays honest.
    const size_t d = c.doneCount.fetch_add(1, std::memory_order_relaxed) + 1;
    if (c.log && (!reuse || rtxmg::g_verboseLogging))
    {
        const char* meshName = mesh.name ? mesh.name : "<unnamed>";
        std::lock_guard<std::mutex> lock(c.logMutex);
        log::info("  [%zu/%zu] %s '%s': %u triangles, %u vertices",
                  d, c.missCount, reuse ? "Reused" : "Baked", meshName, tris, verts);
    }

    rtxmg::GetBakeProgress().CompleteOne(worker);
}

} // namespace

// ---------------------------------------------------------------------------
// ClusterLodGltfImporter
// ---------------------------------------------------------------------------

ClusterLodGltfImporter::ClusterLodGltfImporter(const BakerConfig& bakerConfig,
                                               bool               log,
                                               const fs::path&    cacheDirOverride,
                                               uint32_t           bakeWorkers)
    : m_bakerConfig(bakerConfig)
    , m_log(log)
    , m_cacheDirOverride(cacheDirOverride)
    , m_bakeWorkers(bakeWorkers)
{}

std::optional<ClusterLodModel> ClusterLodGltfImporter::Load(const fs::path& path) const
{
    const std::string filePath = path.string();

    // Per-mesh import warnings are rate-limited; the guard tallies whatever was
    // suppressed on every return path.
    ResetImportWarnings();
    struct WarnReportGuard { ~WarnReportGuard() { ReportSuppressedImportWarnings(); } } warnReportGuard;

    // Route meshoptimizer's allocations through mimalloc before baking so the
    // parallel bake doesn't serialise on the global heap lock (see helper).
    EnsureMeshoptScalableAllocator();

    // ---- Parse the glTF file -----------------------------------------------
    // Declare `mappings` before `gltf` so it outlives it: cgltf_free releases
    // each buffer through CgltfReleaseMapped, which must still find the list.
    rtxmg::FileMappingList mappings;

    cgltf_options options  = {};
    options.file.read      = rtxmg::CgltfReadMapped;
    options.file.release   = rtxmg::CgltfReleaseMapped;
    options.file.user_data = &mappings;

    UniqueCgltfPtr gltf(nullptr, &cgltf_free);
    if (!ParseGltfFile(filePath, options, /*loadBuffers=*/!m_useShardCache, gltf))
        return std::nullopt;

    // ---- Deduplicate meshes into unique geometries --------------------------
    std::vector<size_t> meshToGeometry;
    std::vector<size_t> geometryToMesh;
    DedupMeshesToGeometries(gltf.get(), meshToGeometry, geometryToMesh);

    // ---- Extract geometry for each unique mesh -----------------------------
    ClusterLodModel model;
    model.storages.resize(geometryToMesh.size());

    // The shard path defers LoadGeometry to BakeOrLoadShards' per-miss loop, so
    // only stale geometries' buffers are ever faulted in.
    if (!m_useShardCache)
    {
        MeshoptDecodeScratch decodeScratch;  // serial path; one reused scratch
        for (size_t gi = 0; gi < geometryToMesh.size(); ++gi)
        {
            const cgltf_mesh& mesh = gltf->meshes[geometryToMesh[gi]];
            // LoadGeometry fails on its own against an undecoded view; this
            // only names the cause.
            if (!DecodeMeshoptViewsForMesh(mesh, decodeScratch))
                log::warning("ClusterLodGltfImporter: EXT_meshopt_compression decode failed for "
                             "mesh '%s'.", mesh.name ? mesh.name : "<unnamed>");
            LoadGeometry(mesh, gltf.get(), model.storages[gi]);
            decodeScratch.Release();
        }
    }

    // ---- EXT_mesh_gpu_instancing: pre-load instance-transform buffers ------
    if (!PreloadGpuInstancingBuffers(gltf.get(), filePath, options, mappings, m_useShardCache))
        return std::nullopt;

    // ---- Collect instances from the scene graph ----------------------------
    CollectSceneInstances(gltf.get(), meshToGeometry, model.instances);

    // ---- Parse materials from the glTF -------------------------------------
    ParseMaterials(gltf.get(), path.parent_path(), model.materials);

    // localMaterialStateBits is derived here rather than in the baker, which
    // keeps the baker Scene-free and makes the bits cache-serialisable with the
    // rest of GeometryStorage.  The shard path does this per cache miss instead.
    if (!m_useShardCache)
        for (size_t gi = 0; gi < model.storages.size(); ++gi)
            ComputeStateBits(model.storages[gi], model.materials);

    // ---- bake-or-load-from-cache pipeline ----------------------------------
    if (m_useShardCache)
    {
        // Per-geometry content-hash shard cache (default).  See BakeOrLoadShards.
        if (!BakeOrLoadShards(gltf.get(), geometryToMesh, path, options, model))
            return std::nullopt;
    }
    else if (!BakeOrLoadMonolith(gltf.get(), geometryToMesh, path, model))
    {
        return std::nullopt;
    }

    DropEmptyGeometries(model, geometryToMesh);

    if (m_log && rtxmg::g_verboseLogging)
        LogGeometryStats(gltf.get(), geometryToMesh, model);

    const char* backing = model.cacheView.IsValid() ? "monolith mmap"
                        : (!model.shardCache.IsEmpty() ? "shard mmap" : "storage");
    // ASCII only: the console codepage renders a UTF-8 em-dash as garbage.
    log::info("ClusterLodGltfImporter: loaded '%s' - %zu geometries, %zu instances (%s).",
              filePath.c_str(), model.geometries.size(), model.instances.size(), backing);

    return model;
}

// ---------------------------------------------------------------------------
// BakeOrLoadMonolith — one .nvsngeo cache file for the whole scene
// ---------------------------------------------------------------------------

bool ClusterLodGltfImporter::BakeOrLoadMonolith(const cgltf_data*          gltf,
                                                const std::vector<size_t>& geometryToMesh,
                                                const fs::path&            gltfPath,
                                                ClusterLodModel&           model) const
{
    const std::string filePath  = gltfPath.string();
    const size_t      geoCount  = model.storages.size();
    const fs::path    cachePath = fs::path(gltfPath).replace_extension(".nvsngeo");

    if (m_log)
        log::info("ClusterLodGltfImporter: '%s' - %zu geometries, checking cache '%s'.",
                  filePath.c_str(), geoCount, cachePath.filename().string().c_str());

    // Try to load a valid, up-to-date cache.
    CacheFileView cacheView;
    bool cacheHit = false;
    if (fs::exists(cachePath))
    {
        if (cacheView.Init(cachePath))
        {
            // Validate both the format version (checked inside init) and the
            // baker parameters so that changing BakerConfig forces a rebake.
            if (cacheView.GetConfigHash() == m_bakerConfig.SemanticHash()
                && cacheView.GetGeometryCount() == geoCount)
            {
                cacheHit = true;
            }
            else
            {
                cacheView.Deinit();
            }
        }
    }

    if (m_log)
        log::info("ClusterLodGltfImporter: %s.", cacheHit ? "cache hit" : "cache miss - baking");

    if (!cacheHit)
    {
        // Capture input mesh stats before baking clears vertexPositions/triangles.
        struct InputStats { uint32_t triangles; uint32_t vertices; };
        std::vector<InputStats> inputStats(geoCount);
        if (m_log)
        {
            for (size_t gi = 0; gi < geoCount; ++gi)
                inputStats[gi] = { (uint32_t)model.storages[gi].triangles.size(),
                                   (uint32_t)model.storages[gi].vertexPositions.size() };
        }

        // One baker is safe to share across workers: it holds only const config,
        // and every Build() keeps its mutable state in a stack-local BakeContext.
        // Largest-first so a few heavy meshes don't strand a worker at the tail.
        ClusterLodBaker baker(m_bakerConfig);

        std::vector<size_t> bakeOrder(geoCount);
        for (size_t i = 0; i < geoCount; ++i)
            bakeOrder[i] = i;
        std::sort(bakeOrder.begin(), bakeOrder.end(), [&](size_t a, size_t b) {
            return model.storages[a].triangles.size() > model.storages[b].triangles.size();
        });

        const uint32_t workerCount = std::max(1u, std::thread::hardware_concurrency());
        donut::engine::ThreadPool bakePool(workerCount);

        std::atomic<size_t> bakedCount{ 0 };
        std::mutex          logMutex;

        rtxmg::ParallelFor(bakePool, workerCount, geoCount, [&](size_t t, uint32_t /*workerIdx*/) {
            const size_t gi = bakeOrder[t];
            if (!model.storages[gi].triangles.empty())
                baker.Build(model.storages[gi]);

            if (m_log)
            {
                const size_t done = bakedCount.fetch_add(1, std::memory_order_relaxed) + 1;
                const char*  meshName = gltf->meshes[geometryToMesh[gi]].name;
                if (!meshName) meshName = "<unnamed>";

                std::lock_guard<std::mutex> lock(logMutex);
                log::info("  [%zu/%zu] Baked '%s': %u triangles, %u vertices",
                          done, geoCount, meshName,
                          inputStats[gi].triangles, inputStats[gi].vertices);
            }
        });

        // Build views from storages for the cache writer.
        std::vector<GeometryView> views;
        views.reserve(geoCount);
        for (const GeometryStorage& storage : model.storages)
            views.push_back(MakeGeometryView(storage));

        // Write cache then memory-map it.
        if (!SaveCache(cachePath, views, m_bakerConfig))
        {
            log::warning("ClusterLodGltfImporter: failed to write cache '%s'.",
                         cachePath.string().c_str());
        }
        else if (!cacheView.Init(cachePath))
        {
            log::warning("ClusterLodGltfImporter: wrote cache but could not map it; "
                         "falling back to storage-backed views.");
        }
    }

    // Populate model.geometries — either from the mmap'd cache or from storages.
    model.geometries.resize(geoCount);
    if (cacheView.IsValid())
    {
        for (size_t gi = 0; gi < geoCount; ++gi)
        {
            if (!cacheView.GetGeometryView(model.geometries[gi], gi))
            {
                log::error("ClusterLodGltfImporter: GetGeometryView failed for index %zu.", gi);
                return false;
            }
        }
        model.cacheView = std::move(cacheView);
    }
    else
    {
        // Fallback: point views into the owned storages.
        for (size_t gi = 0; gi < geoCount; ++gi)
            model.geometries[gi] = MakeGeometryView(model.storages[gi]);
    }

    return true;
}

// ---------------------------------------------------------------------------
// BakeOrLoadShards — per-geometry content-hash shard cache
// ---------------------------------------------------------------------------

bool ClusterLodGltfImporter::BakeOrLoadShards(const cgltf_data*          gltf,
                                              const std::vector<size_t>& geometryToMesh,
                                              const fs::path&            gltfPath,
                                              cgltf_options&             options,
                                              ClusterLodModel&           model) const
{
    const auto   tStart  = std::chrono::steady_clock::now();
    const size_t geoCount = model.storages.size();

    // Cache directory: an explicit override (testing — scratch dir) if given,
    // else next to the gltf.  One location per gltf whatever the bake settings;
    // a settings change re-bakes in place rather than segregating.
    fs::path cacheDir;
    if (!m_cacheDirOverride.empty())
    {
        cacheDir = m_cacheDirOverride;
        if (m_log)
            log::info("ClusterLodGltfImporter: using cache dir override '%s'.",
                      cacheDir.string().c_str());
    }
    else
    {
        // Deliberately not keyed by the gltf filename: sibling gltfs referencing
        // the same meshes then share both the content-hashed shards and the
        // source-keyed manifest, so the second one rebakes nothing.
        cacheDir = fs::path(gltfPath).parent_path() / "_nvsngeocache";
    }
    std::error_code ec;
    fs::create_directories(cacheDir, ec);

    // 1) Cheap per-geometry source identity (size + mtime), from gltf metadata —
    //    no buffer data faulted in.
    std::vector<SourceIdentity> ids(geoCount);
    for (size_t gi = 0; gi < geoCount; ++gi)
        ids[gi] = ComputeSourceIdentity(gltf->meshes[geometryToMesh[gi]], gltf, gltfPath);

    // 2) Load the shared manifest, keyed by source-byte identity rather than this
    //    gltf's geometry index, so sibling gltfs share its entries.
    const fs::path                              manifestPath = cacheDir / "index.bin";
    std::unordered_map<uint64_t, ManifestEntry> manifest;
    ManifestConfigInfo                          prevConfig;
    LoadManifest(manifestPath, m_bakerConfig, manifest, prevConfig);

    // 3) Decide hits vs misses: look each geometry's source-byte key up in the
    //    manifest, gate on size+mtime, and require the named shard to exist.
    std::vector<uint8_t>  isMiss(geoCount, 1);
    std::vector<uint64_t> contentHash(geoCount, 0);
    std::vector<fs::path> shardPath(geoCount);
    size_t                hitCount = 0;
    size_t                staleConfigCount = 0;
    const uint64_t        configHash = m_bakerConfig.SemanticHash();
    for (size_t gi = 0; gi < geoCount; ++gi)
    {
        auto it = manifest.find(ids[gi].sourceKey);
        if (it == manifest.end())
            continue;
        const ManifestEntry& e = it->second;
        if (e.sourceSize != ids[gi].sumFileSize || e.sourceMtime != ids[gi].maxMtime)
            continue;

        fs::path    p = cacheDir / (HashToHex(e.contentHash) + ".shard");
        ShardHeader hdr;
        // The manifest can point at a stale or absent shard, so validate the
        // header too and treat a failure as a miss rather than a fatal map error.
        // fs::exists first, so an absent shard doesn't log a failed-open.
        if (!fs::exists(p) || !ShardCache::ReadHeader(p, hdr))
            continue;
        if (hdr.configHash != configHash)
        {
            ++staleConfigCount;
            continue;
        }
        shardPath[gi]   = std::move(p);
        contentHash[gi] = e.contentHash;
        isMiss[gi]      = 0;
        ++hitCount;
    }
    const size_t missCount = geoCount - hitCount;

    if (prevConfig.valid && !prevConfig.matches)
    {
        const std::string diff = DescribeConfigDifferences(prevConfig.config, m_bakerConfig);
        log::warning("ClusterLodGltfImporter: bake settings changed (%s) - %zu of %zu shards in "
                     "'%s' were baked with the previous settings and will be re-baked in place, "
                     "overwriting those cached copies.",
                     diff.empty() ? "hash differs" : diff.c_str(),
                     staleConfigCount, geoCount, cacheDir.string().c_str());
    }

    if (m_log)
        log::info("ClusterLodGltfImporter: shard cache '%s' - %zu/%zu cached, %zu to (re)bake.",
                  cacheDir.filename().string().c_str(), hitCount, geoCount, missCount);

    // Bake progress for the loading-screen UI; the guard marks it inactive on
    // every return path.
    const uint32_t progressWorkers = std::max(1u, std::thread::hardware_concurrency());
    rtxmg::GetBakeProgress().Begin(uint32_t(missCount), uint32_t(hitCount), progressWorkers);
    struct ProgressEndGuard { ~ProgressEndGuard() { rtxmg::GetBakeProgress().End(); } } progressEndGuard;

    // 4) For misses: map buffers (only miss geometries fault their pages in),
    //    extract geometry, hash, and bake (or reuse an existing content shard).
    std::vector<uint8_t> bakedInStorage(geoCount, 0);
    // Geometries that yielded no LOD hierarchy: neither cached nor mapped, and
    // dropped from the model by the caller.
    std::vector<uint8_t> isEmpty(geoCount, 0);
    if (missCount > 0)
    {
        cgltf_result r = cgltf_load_buffers(&options, const_cast<cgltf_data*>(gltf),
                                            gltfPath.string().c_str());
        if (r != cgltf_result_success)
        {
            log::error("ClusterLodGltfImporter: cgltf_load_buffers failed (result %d).", (int)r);
            return false;
        }

        std::vector<size_t> missList;
        missList.reserve(missCount);
        for (size_t gi = 0; gi < geoCount; ++gi)
            if (isMiss[gi])
                missList.push_back(gi);

        // Largest-first so heavy meshes don't strand a worker at the tail.  The
        // workers do the extraction, so triangle counts aren't known yet; the
        // compressed source size is the available proxy.
        std::sort(missList.begin(), missList.end(), [&](size_t a, size_t b) {
            return ids[a].sumFileSize > ids[b].sumFileSize;
        });

        const uint32_t            workerCount = BakeWorkerCount(m_bakeWorkers);
        ClusterLodBaker           baker(m_bakerConfig);
        donut::engine::ThreadPool pool(workerCount);
        std::atomic<size_t>       done{ 0 };
        std::mutex                logMutex;

        // Two geometries can share a buffer view -- exporters pack several
        // accessors into one -- and cgltf hands a decoded meshopt view back
        // through buffer_view->data, one pointer for the whole file.  Give each
        // worker its own parse instead: decoding is then private, so a shared
        // view is simply decoded once per worker that needs it rather than held
        // for the entire bake.  cgltf builds its arrays in JSON order, so mesh
        // indices match the parse the caller already did.
        rtxmg::GetBakeProgress().Begin(workerCount, 0, workerCount, "Preparing bake workers");
        std::vector<UniqueCgltfPtr> workerGltf;
        workerGltf.reserve(workerCount);
        for (uint32_t w = 0; w < workerCount; ++w)
            workerGltf.emplace_back(nullptr, &cgltf_free);

        std::atomic<bool> parseOk{ true };
        rtxmg::ParallelFor(pool, workerCount, workerCount, [&](size_t i, uint32_t w) {
            cgltf_data* d = nullptr;
            if (cgltf_parse_file(&options, gltfPath.string().c_str(), &d) != cgltf_result_success)
            {
                parseOk.store(false, std::memory_order_relaxed);
                return;
            }
            workerGltf[i].reset(d);
            if (cgltf_load_buffers(&options, d, gltfPath.string().c_str()) != cgltf_result_success)
                parseOk.store(false, std::memory_order_relaxed);
            rtxmg::GetBakeProgress().CompleteOne(w);
        });

        if (!parseOk.load(std::memory_order_relaxed))
        {
            log::error("ClusterLodGltfImporter: per-worker glTF parse failed.");
            return false;
        }

        // Re-arm the bar for the bake itself; progressEndGuard ends it.
        rtxmg::GetBakeProgress().Begin(uint32_t(missList.size()), uint32_t(hitCount), workerCount);

        ShardBakeContext bakeCtx{ workerGltf, geometryToMesh, ids, m_bakerConfig, baker, cacheDir,
                                  configHash, m_log, missList.size(), done, logMutex, model,
                                  contentHash, shardPath, isEmpty, bakedInStorage };

        rtxmg::ParallelFor(pool, workerCount, missList.size(), [&](size_t t, uint32_t w) {
            BakeOneGeometry(bakeCtx, missList[t], w);
        });
    }

    // 5) Populate geometries (+ storages.localMaterialIDs for non-baked ones,
    //    which the scene's material flatten reads).
    model.geometries.resize(geoCount);
    for (size_t gi = 0; gi < geoCount; ++gi)
    {
        if (isEmpty[gi])
            continue;  // left with lodLevelsCount == 0 for DropEmptyGeometries

        if (bakedInStorage[gi])
        {
            // The shard write failed (disk full / permissions), so fall back to a
            // view into the RAM copy the worker kept.
            model.geometries[gi] = MakeGeometryView(model.storages[gi]);
            continue;
        }

        // Cache hit, content-reuse, or freshly baked+saved: zero-copy view
        // into the mmap'd shard.
        GeometryView view;
        if (!model.shardCache.LoadGeometryView(shardPath[gi], m_bakerConfig, view))
        {
            // Warn and drop rather than fail the load: a shard that maps but
            // holds no LOD hierarchy is rejected here, and fataling would make
            // one bad cache entry unloadable until the directory is deleted.
            log::warning("ClusterLodGltfImporter: failed to map shard '%s' for geometry %zu; "
                         "dropping that geometry.",
                         shardPath[gi].string().c_str(), gi);
            isEmpty[gi] = 1;
            continue;
        }
        model.geometries[gi] = view;

        // A cache hit never ran LoadGeometry, so recover the material tables
        // downstream needs from the shard itself.
        GeometryStorage s;
        s.localMaterialIDs.assign(view.localMaterialIDs.begin(), view.localMaterialIDs.end());
        s.localMaterialStateBits.assign(view.localMaterialStateBits.begin(),
                                        view.localMaterialStateBits.end());
        model.storages[gi] = std::move(s);
    }

    // 6) Merge rather than overwrite, so other gltfs' entries in the shared
    //    folder survive.  A concurrent bake of a different gltf can still drop
    //    entries in the race; they simply re-hash on the next load.
    for (size_t gi = 0; gi < geoCount; ++gi)
    {
        if (isEmpty[gi])
            // Drop the entry that pointed at the unusable shard, so the next
            // run misses and re-bakes rather than dropping the geometry again.
            manifest.erase(ids[gi].sourceKey);
        else
            manifest[ids[gi].sourceKey] =
                ManifestEntry{ contentHash[gi], ids[gi].sumFileSize, ids[gi].maxMtime };
    }
    SaveManifest(manifestPath, m_bakerConfig, manifest);

    const double secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - tStart).count();
    log::info("ClusterLodGltfImporter: shard cache resolved in %.1f s "
              "(%zu geometries: %zu cache-hit, %zu (re)baked/decoded).",
              secs, geoCount, hitCount, missCount);

    return true;
}
