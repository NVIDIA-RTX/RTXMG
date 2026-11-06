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

#pragma once

#ifdef __cplusplus
#include <cstdint>
#include <donut/core/math/math.h>
#endif
#include "nvrhi/nvrhiHLSL.h"

#include "rtxmg/cluster_lod/shader_dispatch.h"

// ---------------------------------------------------------------------------
// shaderio -- GPU-facing POD structs for the cluster LOD pipeline, plus the
// sentinels stored inside them.  Each struct's binary layout is pinned by a
// static_assert next to its definition.  The dispatch protocol — thread-group
// sizes and setup-kernel mode IDs — is shader_dispatch.h, included above so
// every consumer of this header still sees both.
// ---------------------------------------------------------------------------

#ifdef __cplusplus
using namespace donut::math;

// Vulkan's scalar block layout adds no tail padding, while C++ and DXIL round
// sizeof up to the widest member's alignment — so an 8-byte-aligned struct that
// does not end on an 8-byte multiple is read 4 bytes off by the SPIR-V.
// Neither sizeof form can see that: `sizeof % alignof == 0` is guaranteed by
// the language, so it never fires, and an exact `sizeof ==` passes with or
// without the pad.  Only the natural end of the members exposes it.
#define SHADERIO_ASSERT_NO_TAIL_PADDING(T, lastMember)                             \
    static_assert(sizeof(T) == offsetof(T, lastMember) + sizeof(T::lastMember),     \
                  #T " has tail padding — Vulkan's scalar layout would place "      \
                  "everything after it 4 bytes low; add an explicit trailing pad")
#endif

#ifndef __cplusplus
// HLSL bit-field extraction helper (global scope so Node_* macros work outside any namespace).
uint BitfieldExtract(uint val, uint offset, uint count) { return (val >> offset) & ((1u << count) - 1u); }
#endif

namespace shaderio {

static const uint32_t kOriginalMeshGroup = 0xffffffffu;
static const uint32_t kMaxLodLevels      = 32;

// ---------------------------------------------------------------------------
// Material + alpha/two-sided state.  These values are written into baked
// .nvsngeo shards, so changing one invalidates every cache.
// ---------------------------------------------------------------------------

// Per-triangle material byte layout: low 6 bits = local material index,
// bit 6 = two-sided, bit 7 = alpha-masked.
static const uint32_t kClusterTriangleTwoSided    = 1u << 6;
static const uint32_t kClusterTriangleAlphaMasked = 1u << 7;

static const uint32_t kMaxLocalMaterials = kClusterTriangleTwoSided - 1u;
static const uint32_t kLocalMaterialMask = kMaxLocalMaterials;
// Cluster::localMaterialID sentinel: per-triangle material bytes are stored in
// the triangle payload after the index bytes.  uint32_t, not uint8_t: HLSL has
// no uint8_t, and 0xFFu already promoted this way at every C++ comparison.
static const uint32_t kPerTriangleMaterials = 0xFFu;

// Sentinel device address (top bit set) for a host-side resident-group record
// whose blob has not been streamed in yet.
static const uint64_t kStreamingInvalidAddressStart = uint64_t(1) << 63;
// GroupAddress::srvIndex sentinel — the group is not resident.
static const uint32_t kStreamingInvalidSrvIndex = 0xffffffffu;

// Cluster::stateBits values.  Fixed underlying type so `~Flag` stays unsigned;
// without it the values promote to int and the complement goes negative.
enum ClusterState : uint32_t
{
    AlphaMasked      = 1u,
    AlphaMaskedMixed = 2u,
    TwoSided         = 4u,
    TwoSidedMixed    = 8u,
};

// RenderInstance::opaqueStatus values.  Scoped because AlphaMasked would
// otherwise collide with ClusterState in this namespace.
enum class OpaqueStatus : uint32_t
{
    Opaque      = 1u,
    AlphaMasked = 2u,
    Mixed       = 3u,
};

// Cluster::attributeBits flag bits.  Also baked into the shard layout.
enum ClusterAttribute : uint32_t
{
    VertexNormal  = 1u,
    VertexTangent = 2u,
    VertexTex0    = 4u,
    VertexTex1    = 8u,
    // Resident copy has NO position array (positions diverted to a transient
    // upload ring for the CLAS build; hit shaders fetch them from the AS).
    // Attribute offset derivation skips the vertexCount*12 term.  Set at UPLOAD
    // time only — the baked / shard layout always carries positions.
    StrippedVertexPos    = 16u,
    CompressedVertexTex0 = 32u,
    CompressedVertexTex1 = 64u,
    CompressedVertexPos  = 128u,
};

// Cluster::encodingBits flag bits.  Set at bake time (BakerConfig::
// quantizeTexCoords): TEXCOORD_n is a 16-byte header (float2 base + float2
// power-of-two step) then one uint per vertex (u16 deltaU | u16 deltaV << 16),
// decoded as base + float(delta) * step.  The po2 step makes that decode
// bit-deterministic across consumers.  Clear = plain float2 array.
enum ClusterEncoding : uint32_t
{
    QuantizedTex0 = 1u,
    QuantizedTex1 = 2u,
};

// Object-space axis-aligned bounding box for a cluster or geometry.
struct BBox
{
    float3 lo;                // 12B
    float3 hi;                // 12B
    float  shortestEdge;      //  4B  (relevant to cluster's triangles)
    float  longestEdge;       //  4B
};
#ifdef __cplusplus
static_assert(sizeof(BBox) == 32, "BBox must be 32 bytes");
#endif

// Traversal metric stored in every Node and Group.
// Scalar fields by design to avoid packing hiccups.
struct TraversalMetric
{
    float boundingSphereX;
    float boundingSphereY;
    float boundingSphereZ;
    float boundingSphereRadius;
    float maxQuadricError;
};
#ifdef __cplusplus
static_assert(sizeof(TraversalMetric) == 20, "TraversalMetric must be 20 bytes");
#endif

// A cluster contains a small number of triangles and vertices.
// It is always part of a group.
//
// Sub-byte fields are expressed as uint32_t bitfields so the layout is
// identical in C++ (MSVC/Clang, little-endian) and HLSL 2021.
struct Cluster
{
    uint32_t triangleCountMinusOne : 8;
    uint32_t vertexCountMinusOne   : 8;
    uint32_t lodLevel              : 8;
    uint32_t groupChildIndex       : 8;

    uint32_t attributeBits   : 8;
    uint32_t localMaterialID : 8;   // kPerTriangleMaterials (0xFF) ⇒ read per-triangle index
    uint32_t stateBits       : 8;   // ClusterState flag bits
    uint32_t encodingBits    : 8;   // ClusterEncoding flag bits (per-cluster vertex-data encoding)

    // Byte offsets relative to the Cluster header.
    uint32_t vertices;   // vec3 positions[vertexCount], then optional per-vertex attributes
    uint32_t triangles;  // uint8 indices[triangleCount * 3], then optional per-triangle material
};
#ifdef __cplusplus
static_assert(sizeof(Cluster) == 16, "Cluster must be 16 bytes");
#endif

// A group contains multiple clusters that are the result of a common mesh
// decimation operation.  Clusters within a group are watertight to each other.
// Groups are always streamed in completely.
struct Group
{
    // Scene-global id of this group's first cluster.  traversal_run_groups
    // emits `clusterResidentID + clusterIndex`, which blas_insert_clusters and
    // the hit shader use to index the scene-global resident CLAS / cluster
    // tables.  Allocator-issued per load in streaming, bump-allocated at Init
    // in preload.
    uint32_t clusterResidentID;
    // Scene-global slot in the resident-group table (traversal clears its age
    // on visit, stream_age_groups increments it).  Same allocation scheme.
    uint32_t groupResidentID;

    uint32_t lodLevel     : 8;
    uint32_t stateBits    : 8;   // aggregate ClusterState over the group's clusters
    uint32_t clusterCount : 16;  // max 128

    TraversalMetric traversalMetric;
};
#ifdef __cplusplus
static_assert(sizeof(Group) == 32, "Group must be 32 bytes");
// The blob's clusters[] starts at `align_up(sizeof(Group), 16)` (GroupView in
// baked_geometry.h) while several streaming/preload paths assume a bare
// sizeof(Group); keeping the size a multiple of 16 makes the two agree.
static_assert(sizeof(Group) % 16 == 0, "Group size must be a multiple of 16 bytes "
                                       "so the blob's clusters[] offset matches "
                                       "sizeof(Group) without an align_up shift");
#endif

// Per-geometry resident/request state for a group.  Resident entries address
// the group blob by bindless ByteAddressBuffer slot + byte offset; non-resident
// entries set srvIndex = kStreamingInvalidSrvIndex and overload byteOffset to
// carry the last frame that requested the group.
struct GroupAddress
{
    uint32_t srvIndex;
    uint32_t byteOffset;
};
#ifdef __cplusplus
static_assert(sizeof(GroupAddress) == 8, "GroupAddress must be 8 bytes");
#endif

#ifdef __cplusplus
// Interior node — children are other nodes.
struct NodeRange
{
    uint32_t isGroup          : 1;
    uint32_t childOffset      : 26;
    uint32_t childCountMinusOne : 5;
};

// Leaf node — children are groups.
struct GroupRange
{
    uint32_t isGroup                  : 1;
    uint32_t groupIndex               : 23;
    uint32_t groupClusterCountMinusOne : 8;
};
#endif // __cplusplus

// LOD traversal tree node.
struct Node
{
#ifdef __cplusplus
    union
    {
        NodeRange  nodeRange;
        GroupRange groupRange;
    };
#else
    // In HLSL use the Node_* macros below to extract bit fields.
    uint32_t packed;
#endif
    TraversalMetric traversalMetric;
};
#ifdef __cplusplus
static_assert(sizeof(Node) == 24, "Node must be 24 bytes");
#endif

#ifndef __cplusplus
// Node bit-field extraction macros (BitfieldExtract is defined in global scope above).
#define Node_isGroup(p)                   BitfieldExtract(p,  0, 1)
#define Node_nodeChildOffset(p)           BitfieldExtract(p,  1, 26)
#define Node_nodeChildCountMinusOne(p)    BitfieldExtract(p, 27, 5)
#define Node_groupIndex(p)                BitfieldExtract(p,  1, 23)
#define Node_groupClusterCountMinusOne(p) BitfieldExtract(p, 24, 8)
#endif

// Per-LOD-level statistics used for traversal.
struct LodLevel
{
    float    minBoundingSphereRadius;
    float    minMaxQuadricError;
    uint32_t groupOffset;
    uint32_t groupCount;
    uint32_t clusterOffset;
    uint32_t clusterCount;
};
#ifdef __cplusplus
static_assert(sizeof(LodLevel) == 24, "LodLevel must be 24 bytes");
#endif

// "No LoD level" sentinel: Geometry::cachedBlasLodLevel when nothing is cached.
// traversal_init also parks it in InstanceBuildInfo::lodLevelMin/Max, on the
// non-sharing path where nothing reads them.
static const uint32_t kTraversalInvalidLodLevel = 0xFFu;

// Per-geometry GPU data accessed by the traversal and BLAS-build shaders.
//
// Fields that HLSL shaders dereference as typed arrays are uint32_t
// ResourceDescriptorHeap indices (SM 6.6 bindless): DX12 HLSL has no
// layout(buffer_reference) equivalent, so a raw GpuVirtualAddress cannot be
// dereferenced in a shader.  Only hardware-consumed addresses (TLAS / BLAS
// inputs) stay GpuVirtualAddress.  The per-cluster CLAS / cluster-address
// tables are scene-global on ClusterLodResourcesBase, indexed by
// `clusterResidentID` rather than reached through this struct.
struct Geometry
{
    uint32_t instancesOffset;
    uint32_t instancesCount;
    uint32_t lodLevelsCount     : 8;
    uint32_t cachedBlasLodLevel : 8;   // kTraversalInvalidLodLevel when unused
    uint32_t lowDetailTriangles : 16;
    uint32_t lowDetailClusterID;
    // 12+4 = 16, so the uint64s below need no explicit padding here.
    nvrhi::GpuVirtualAddress lowDetailBlasAddress;  // 0 until CLAS build
    nvrhi::GpuVirtualAddress cachedBlasAddress;     // cached-BLAS pool allocation; 0 when none

    BBox     bbox;

    // LOD hierarchy SRVs (bindless heap indices).
    uint32_t lodLevelsSRV;                        // StructuredBuffer<LodLevel>
    uint32_t nodesSRV;                            // StructuredBuffer<Node>
    uint32_t nodeBboxesSRV;                       // StructuredBuffer<BBox>
    // Per-geometry group blob address table (see GroupAddress).
    uint32_t streamingGroupAddressesSRV;          // StructuredBuffer<GroupAddress>
    uint32_t streamingGroupAddressesUAV;          // RWStructuredBuffer<GroupAddress>
    // Trailing padding made explicit so C++ and HLSL agree deterministically.
    uint32_t _pad84;

    // Cluster-LoD material resolve: a hit's localID (Cluster::localMaterialID
    // or the per-triangle byte) indexes
    // t_ClusterLodLocalMaterialIDs[localMaterialsOffset + localID], and the result
    // plus materialBaseID is the t_MaterialConstants slot.  materialBaseID is
    // the same scene-wide value for every geometry; it lives here rather than
    // in a push constant so any cluster-LoD shader can call the resolve helper.
    uint32_t materialBaseID;
    uint32_t localMaterialsOffset;
    uint32_t localMaterialsCount;
    // Render-stats: low-detail and cached-BLAS instances skip traversal, so
    // their contribution is tallied from these instead.  cachedBlas* are 0 when
    // nothing is cached; written by stream_update_scene from the geometry patch.
    uint32_t lowDetailClusters;
    uint32_t cachedBlasTriangles;
    uint32_t cachedBlasClusters;
};
#ifdef __cplusplus
static_assert(sizeof(Geometry) == 112, "Geometry must be 112 bytes");
SHADERIO_ASSERT_NO_TAIL_PADDING(Geometry, cachedBlasClusters);
#endif

// ---------------------------------------------------------------------------
// Per-resident-cluster address inside its group-data buffer: bindless heap slot
// of the backing ByteAddressBuffer SRV plus the byte offset of the cluster's
// Cluster header.  The header's vertices/triangles fields are relative to that
// offset.  Preload uses one buffer slot per geometry; streaming's varies with
// the pool block the group was streamed into.
// ---------------------------------------------------------------------------
struct ClusterAddress
{
    uint32_t srvIndex;
    uint32_t byteOffset;
};
#ifdef __cplusplus
static_assert(sizeof(ClusterAddress) == 8, "ClusterAddress must be 8 bytes");
#endif

// ---------------------------------------------------------------------------
// Per-instance data for the LOD traversal shaders.
// 128 bytes (power-of-two for aligned indexed access).
// ---------------------------------------------------------------------------
struct RenderInstance
{
    float3x4 worldMatrix;    // 48B  object-to-world (row-major 3 rows × 4 cols)
    // Affine inverse of worldMatrix (bottom [0,0,0,1] row dropped).  Used by
    // instance_classify_lod to put the camera position in object space.
    float3x4 worldMatrixI;   // 48B  world-to-object
    uint32_t geometryID;     // 4B

    uint32_t opaqueStatus              : 8;   // OpaqueStatus
    uint32_t multiMaterial             : 8;   // 0 = single material; 1 = range starting at materialID
    uint32_t lowDetailClusterStateBits : 8;   // ClusterState of lowest-detail cluster
    uint32_t _stateReserved            : 8;

    // Slot in t_MaterialConstants; with multiMaterial it is the base of a
    // contiguous range of (max local-material-ID + 1) slots.
    uint32_t materialID;     // 4B

    uint32_t _pad2, _pad3, _pad4, _pad5, _pad6;  // 20B → 128-byte stride
};
#ifdef __cplusplus
static_assert(sizeof(RenderInstance) == 128, "RenderInstance must be 128 bytes");
#endif

// ---------------------------------------------------------------------------
// Item descriptor for the LOD traversal producer/consumer queue.
// Fits in two uint32s (treated as uint2 in HLSL).
// ---------------------------------------------------------------------------
struct TraversalInfo
{
    uint32_t instanceID;
    uint32_t packedNode;  // packed NodeRange or GroupRange bitfield
};
#ifdef __cplusplus
static_assert(sizeof(TraversalInfo) == 8, "TraversalInfo must be 8 bytes");
#endif

// ---------------------------------------------------------------------------
// A cluster selected by the traversal for dynamic BLAS building.
// Same 8-byte layout as TraversalInfo.
// ---------------------------------------------------------------------------
struct ClusterInfo
{
    uint32_t instanceID;
    uint32_t clusterID;
};
#ifdef __cplusplus
static_assert(sizeof(ClusterInfo) == 8, "ClusterInfo must be 8 bytes");
static_assert(sizeof(ClusterInfo) == sizeof(TraversalInfo),
              "ClusterInfo and TraversalInfo share the traversal queue slot");
#endif

// ---------------------------------------------------------------------------
// Per-frame constants for the LOD traversal shaders (ConstantBuffer).
// ---------------------------------------------------------------------------
struct SceneBuildingConstants
{
    float4x4 traversalViewMatrix;          // 64B  world-to-view for LOD metric
    uint32_t numRenderInstances;           // 4B
    uint32_t maxTraversalInfos;            // 4B   traversal queue capacity
    uint32_t maxRenderClusters;            // 4B   output cluster array capacity
    float    errorOverDistanceThreshold;   // 4B   LOD selection threshold
    float    nearPlane;                    // 4B
    // BLAS-sharing knobs; zero/ignored on the USE_BLAS_SHARING=0 path.  The
    // host seeds 8 / 7 / 1 when a sharing frame leaves them all zero.
    uint32_t numGeometries;                // 4B   unique geometry count (bounds blas_elect_sharing_provider)
    uint32_t sharingEnabledLevels;         // 4B   #coarse tail levels eligible for sharing
    uint32_t sharingTolerantLevels;        // 4B   #coarse tail levels allowed +1 LoD tolerance
    uint32_t sharingPushCulled;            // 4B   push culled instances out by one LoD level (0/1)
    // Camera world position; instance_classify_lod places the per-LoD
    // min-sphere relative to the eye in object space (worldMatrixI * viewPos).
    float3   viewPos;                      // 12B
    // Runtime gate for the merge behaviour woven into the sharing shaders.
    // The dedicated merge kernel (traversal_blas_merging) ignores this flag.
    uint32_t useBlasMerging;               // @112
    // Frustum + screen-size visibility test against the previous frame's VP
    // (the test itself lives in culling.hlsli).  Soft cull keeps the instance
    // traversed but biases it coarser via culledErrorScale; hard cull skips
    // traversal and falls back to the low-detail BLAS; hard +
    // hardCullForcesInvisible additionally nulls the BLAS reference so the TLAS
    // build drops the instance entirely (gone from reflection/shadow rays too).
    uint32_t useCulling;                   // @116  master toggle (runs the frustum/size test)
    uint32_t useHardCull;                  // @120  0 = soft (coarsen), 1 = hard (skip traversal)
    uint32_t hardCullForcesInvisible;      // @124  hard cull removes the instance instead of low-detail
    float4x4 cullViewProjMatrix;           // @128  world→clip, matching the HiZ pyramid's view (16-aligned row)
    // Render viewport: xy = pixel size, zw = 1/size (screen-size cull).
    float4   viewportf;                    // @192  (16-aligned row)
    float    culledErrorScale;             // @208  soft-cull LoD bias (clamped >= 1 by the host)
    // HiZ occlusion extends the frustum cull: an instance fully behind the
    // max-reduced depth pyramid is soft-coarsened like a
    // frustum-culled one.  The pyramid itself is bound as
    // Texture2D<float>[HIZ_MAX_LODS] in register space 1; these two fields give
    // the shader the screen-px -> tile -> uv mapping.  hizNumLODs == 0 disables
    // just the HiZ test — e.g. when the Z pre-pass didn't run this frame.
    uint32_t hizNumLODs;                   // @212  HiZ pyramid levels (0 = HiZ test off)
    float2   hizInvSize;                   // @216  1 / HiZ-pyramid-base size (xy)
};
#ifdef __cplusplus
// C++ packs tight here (everything is align-4), so the only constraint is that
// no vector straddles an HLSL 16-byte cbuffer row: the four uints @112-124 fill
// a row so cullViewProjMatrix lands 16-aligned, and the tail row @208-224
// fills exactly.
static_assert(sizeof(SceneBuildingConstants) == 224, "SceneBuildingConstants must be 224 bytes");
#endif

// ---------------------------------------------------------------------------
// Per-frame mutable counters for the LOD traversal shaders (RWStructuredBuffer
// of one element, cleared to zero at the start of each frame).  See
// ClusterLodPass for the multipass dispatch sequence that drives them.
// ---------------------------------------------------------------------------
struct SceneBuildingCounters
{
    // Producer cursors (monotonically grow within a frame).
    uint32_t traversalNodeWriteCounter;  // 4B  next free slot in u_TraversalNodeQ
    uint32_t traversalGroupWriteCounter; // 4B  next free slot in u_TraversalGroupQ
    uint32_t renderClusterCounter;       // 4B  raw output cluster count (append cursor)
    uint32_t numRenderedClusters;        // 4B  clamped count (written by traversal_setup)

    // Multipass slicing: traversal_run / traversal_run_groups read [start, end).
    uint32_t traversalPass;              // 4B  current pass index (0..maxDepth-1)
    uint32_t traversalNodeStart;         // 4B
    uint32_t traversalNodeEnd;           // 4B
    uint32_t traversalGroupStart;        // 4B
    uint32_t traversalGroupEnd;          // 4B

    // Indirect dispatch arg triples (X,Y,Z packed flat so offsetof(...X) works).
    uint32_t indirectDispatchNodesX;     // 4B  traversal_run dispatch size
    uint32_t indirectDispatchNodesY;
    uint32_t indirectDispatchNodesZ;
    uint32_t indirectDispatchGroupsX;    // 4B  traversal_run_groups dispatch size
    uint32_t indirectDispatchGroupsY;
    uint32_t indirectDispatchGroupsZ;
    uint32_t indirectDispatchBlasInsertionX; // 4B blas_insert_clusters dispatch size
    uint32_t indirectDispatchBlasInsertionY;
    uint32_t indirectDispatchBlasInsertionZ;

    // BLAS build counters (zeroed with the rest of the struct each frame).
    uint32_t blasClasCounter;            // 4B  atomic allocator into blasClasAddresses pool
    uint32_t blasBuildCounter;           // 4B  number of per-instance BLASes built this frame

    // BLAS sharing diagnostics — written only on the USE_BLAS_SHARING path,
    // read back under --debug-clusterlod.
    uint32_t numSharingProviders;        // 4B  geometries that elected a sharing provider
    uint32_t numSharingConsumers;        // 4B  instances that reused a provider's BLAS (ShareBit)

    // Atomic counter for the cached-BLAS MOVE_OBJECTS copy ops staged by
    // blas_cache_stage_move (one per cached BLAS patched this frame).
    uint32_t cachedBlasCopyCounter;      // 4B

    // Raw (pre-clamp) render-cluster count, preserved by traversal_setup before it
    // clamps renderClusterCounter.  Exceeding effectiveMaxRenderClusters means
    // clusters were dropped this frame (-> flicker); the UI warns on it.
    uint32_t desiredRenderClusters;      // 4B
    // The cap the clamp actually used: the render-cluster budget minus the
    // cached-cluster reservation when BLAS caching is on.
    uint32_t effectiveMaxRenderClusters; // 4B
    // Raw (pre-clamp) traversal-queue demand, the peak over the frame's passes.
    // Above maxTraversalInfos means nodes/groups were dropped by the queue cap.
    uint32_t desiredTraversalNodes;      // 4B
    uint32_t desiredTraversalGroups;     // 4B
    // Distinct resident clusters (CLAS) referenced this frame — the unique
    // counterpart to numRenderedClusters, which is the per-build count.  Also
    // keeps the uint64 tallies below 8B-aligned.
    uint32_t uniqueClusters;             // 4B

    // Per-frame TLAS triangle tallies (TRACK_RENDER_STATS only; 0 otherwise).
    // 64-bit because the instanced total overflows uint32 on instance-heavy
    // scenes.  unique* dedups at the resident-CLAS level (= BLAS-memory
    // footprint); total* is the per-instance sum the ray tracer traverses.
    // Both include low-detail BLAS triangles.  Accumulated with SM6.6 64-bit
    // atomics -- see [[rwstructuredbuffer-64bit-atomics]].
    uint64_t uniqueTriangles;            // 8B
    uint64_t totalTriangles;             // 8B
    uint32_t totalClusters;              // 4B

    // The slice of the tallies above served by cached BLASes, whose instances
    // reference the cached level instead of traversing.  Accumulated in
    // instance_assign_blas.
    uint32_t cachedClusters;             // 4B
    uint64_t cachedTriangles;            // 8B
    uint64_t cachedUniqueTriangles;      // 8B
    uint32_t cachedUniqueClusters;       // 4B
    // Geometries with a live merged BLAS this frame; 0 when merging is off.
    uint32_t numMergedBlas;              // 4B
};
#ifdef __cplusplus
static_assert(sizeof(SceneBuildingCounters) == 160, "SceneBuildingCounters must be 160 bytes");
SHADERIO_ASSERT_NO_TAIL_PADDING(SceneBuildingCounters, numMergedBlas);
#endif

// InstanceBuildInfo.blasBuildIndex.  LowDetail means the instance falls back to
// the geometry's pre-built lowDetailBlasAddress (set by traversal_init,
// overwritten by blas_reserve_clusters for slow-path instances).  The high bits
// select an indirection: ShareBit → the low bits encode the provider instanceID
// whose BLAS this instance reuses; CacheBit → they encode the geometryID whose
// cached BLAS to use.  LowDetail has both bits set, so test it first.
enum BlasBuildIndex : uint32_t
{
    LowDetail = 0xFFFFFFFFu,
    ShareBit  = 0x80000000u,
    CacheBit  = 0x40000000u,
    // Either indirection; LowDetail matches it too, hence the ordering rule.
    IndirectMask = ShareBit | CacheBit,
};

// Per-instance visibility flags (u_InstanceVisibility).  UsesMerged marks
// instances whose clusters live in the geometry's merged BLAS, so traversal_run
// skips their cluster iteration and the merge proxy hosts the build.
enum InstanceVisibility : uint32_t
{
    Visible    = 1u,
    UsesMerged = 2u,
};

// ---------------------------------------------------------------------------
// Per-instance data written by traversal_run and consumed by blas_setup /
// blas_insert_clusters to build one BLAS per instance.
// ---------------------------------------------------------------------------
struct InstanceBuildInfo
{
    uint32_t clusterReferencesCount; // 4B  cluster count for this instance (atomically incremented by traversal_run)
    uint32_t blasBuildIndex;         // 4B  index into BlasBuildArgs (written by blas_reserve_clusters)
    uint32_t blasClasOffset;         // 4B  base index into blasClasAddresses pool (written by blas_reserve_clusters)
    // Conservative LoD range from instance_classify_lod, read only on the
    // BLAS-sharing path.  The host clears the whole buffer each frame, so the
    // pre-classification value is 0, not kTraversalInvalidLodLevel.
    uint32_t lodLevelMin  : 8;       // highest potential detail (finest, near end)
    uint32_t lodLevelMax  : 8;       // lowest potential detail (coarsest, far end)
    uint32_t _lodReserved : 16;
};
#ifdef __cplusplus
static_assert(sizeof(InstanceBuildInfo) == 16, "InstanceBuildInfo must be 16 bytes");
#endif

// ---------------------------------------------------------------------------
// Per-geometry histograms accumulated by instance_classify_lod (one entry per
// unique geometry, cleared to zero each frame).  blas_elect_sharing_provider reads
// these to elect a canonical sharing-provider instance per lodLevelMax bucket.
// ---------------------------------------------------------------------------
struct GeometryBuildHistogram
{
    uint32_t lodLevelMinHistogram[kMaxLodLevels];     // #instances whose lodLevelMin == L
    uint32_t lodLevelMaxHistogram[kMaxLodLevels];     // #instances whose lodLevelMax == L
    uint32_t lodLevelMaxPackedInstance[kMaxLodLevels];// atomic-max (lodLevelMin<<27)|instanceID per L
};
#ifdef __cplusplus
static_assert(sizeof(GeometryBuildHistogram) == 3 * kMaxLodLevels * 4,
              "GeometryBuildHistogram must be 3*kMaxLodLevels u32");
#endif

// ---------------------------------------------------------------------------
// Per-geometry BLAS-reduction decisions written by blas_elect_sharing_provider (one
// entry per unique geometry).  The caching / merging fields stay zero while
// those features are off, so the buffer size is independent of them.
// ---------------------------------------------------------------------------
struct GeometryBuildInfo
{
    uint32_t shareInstanceID;        // 4B   instance whose BLAS the consumers reuse (BLAS sharing)
    uint32_t shareLevelMin   : 8;    // provider's highest potential detail (BLAS sharing)
    uint32_t shareLevelMax   : 8;    // provider's lowest potential detail (BLAS sharing)
    uint32_t cachedLevel     : 16;   // finest in-use level >= cachedBlasLodLevel (BLAS caching)
    uint32_t cachedBuildIndex;       // 4B   build slot of the cached BLAS (BLAS caching)
    uint32_t mergedInstanceID;       // 4B   proxy instance hosting the merged BLAS build (BLAS merging)
};
#ifdef __cplusplus
static_assert(sizeof(GeometryBuildInfo) == 16, "GeometryBuildInfo must be 16 bytes");
#endif

// ===========================================================================
// Persistent CLAS streaming allocator shaderio.
//
// Conventions the structs below are cited against:
//   (a) Per-element arrays are explicitly bound (RW)StructuredBuffer<T>s, not
//       fields of the scalar header.
//   (b) Pool addresses are (uint32 bindless SRV slot, uint32 byteOffset), since
//       HLSL cannot dereference a raw VA.
//   (c) Addresses handed to the cluster/BLAS build APIs are uint64 VAs.
//   (d) StreamingAllocator's sub-arrays are uint32 byte offsets into a single
//       RWByteAddressBuffer.
// ===========================================================================

// Bit-array word size: must be 32 so usedBits / freeGaps processing is u32-
// aligned and a single allocation straddles at most two u32 words.
static const uint32_t kStreamingAllocatorMinSize = 32u;

// ---------------------------------------------------------------------------
// Single (count, offset) range used by the persistent allocator's size-class
// binning.
// ---------------------------------------------------------------------------
struct AllocatorRange
{
    int32_t  count;
    uint32_t offset;
};
#ifdef __cplusplus
static_assert(sizeof(AllocatorRange) == 8, "AllocatorRange must be 8 bytes");
#endif

// ---------------------------------------------------------------------------
// Persistent-allocator stats (one per allocator, at statsByteOffset inside the
// management buffer).
// ---------------------------------------------------------------------------
struct AllocatorStats
{
    int64_t allocatedSize;   // current live allocation total, in units
    int64_t wastedSize;      // worst-case slack the allocator reserves
};
#ifdef __cplusplus
static_assert(sizeof(AllocatorStats) == 16, "AllocatorStats must be 16 bytes");
SHADERIO_ASSERT_NO_TAIL_PADDING(AllocatorStats, wastedSize);
#endif

// ---------------------------------------------------------------------------
// Indirect-dispatch arguments triple (mirrors nvrhi::DispatchIndirectArguments).
// ---------------------------------------------------------------------------
struct StreamingDispatchIndirect
{
    uint32_t groupsX;
    uint32_t groupsY;
    uint32_t groupsZ;
};
#ifdef __cplusplus
static_assert(sizeof(StreamingDispatchIndirect) == 12, "StreamingDispatchIndirect must be 12 bytes");
#endif

// ---------------------------------------------------------------------------
// Persistent CLAS allocator management header.  Per convention (d) the seven
// sub-arrays (freeGapsPos, freeGapsSize, freeGapsPosBinned, freeSizeRanges,
// usedBits, usedSectorBits, stats) live contiguously inside one
// RWByteAddressBuffer and this struct holds their byte offsets.  Bind as
// RWStructuredBuffer<StreamingAllocator> (single element) so atomic-add on
// freeGapsCounter works.
// ---------------------------------------------------------------------------
struct StreamingAllocator
{
    uint32_t freeGapsCounter;          // atomic-incremented by setup_insertion
    uint32_t granularityByteShift;     // unit size = 1 << shift bytes
    uint32_t maxAllocationSize;        // in units, not bytes (multiple of 32)
    uint32_t sectorCount;
    uint32_t sectorMaxAllocationSized;
    uint32_t sectorSizeShift;
    uint32_t baseWastedSize;
    uint32_t usedBitsCount;

    // build_freegaps writes (groupsX = ceil(freeGapsCounter / threads)) for the
    // freegaps_insert dispatch; the host seeds (1,1,1) at Init.
    StreamingDispatchIndirect dispatchFreeGapsInsert;

    // Byte offsets into the host-bound RWByteAddressBuffer that backs all 7
    // sub-arrays.  Populated by StreamingAllocator::Init.
    uint32_t freeGapsPosByteOffset;        // uint32  per slot
    uint32_t freeGapsSizeByteOffset;       // uint16  per slot (2-byte stride)
    uint32_t freeGapsPosBinnedByteOffset;  // uint32  per slot
    uint32_t freeSizeRangesByteOffset;     // AllocatorRange (8 B) per slot
    uint32_t usedBitsByteOffset;           // uint32 bitmap
    uint32_t usedSectorBitsByteOffset;     // uint32 bitmap
    uint32_t statsByteOffset;              // single AllocatorStats (16 B)
};
#ifdef __cplusplus
static_assert(sizeof(StreamingAllocator) == 72, "StreamingAllocator must be 72 bytes");
#endif

// ---------------------------------------------------------------------------
// Per-frame allocator request header.  The load/unload geometry-group lists are
// explicit bindings (convention (a)).
// ---------------------------------------------------------------------------
struct StreamingFrameRequest
{
    uint32_t maxLoads;
    uint32_t maxUnloads;
    uint32_t loadCounter;                           // atomic-incremented by allocator
    uint32_t unloadCounter;
    uint64_t frameIndex;                            // current request frame

    uint64_t clasCompactionUsedSize;                // compaction-allocator scratch
    uint32_t clasCompactionCount;
    uint32_t clasAllocatedMaxSizedLeft;             // worst-case-budget tracker
    uint64_t clasAllocatedUsedSize;
    uint64_t clasAllocatedWastedSize;

    uint32_t taskIndex;
    int32_t  errorUpdate;
    int32_t  errorAgeFilter;
    int32_t  errorClasNotFound;
    int32_t  errorClasList;
    int32_t  errorClasAlloc;
    int32_t  errorClasDealloc;
    int32_t  errorClasUsedVsAlloc;

    // Distinct error code for "group's allocSize exceeds maxAllocationSize"
    // (set by load_groups' early-detect path).  Separate from errorClasNotFound
    // so the host can tell "config under-sized" apart from "pool full / bug".
    // Value = 1u + threadID of the first reporting thread.
    int32_t  errorClasOversized;

    // Per-task ring stride for the load/unload geometry-group arrays, in uint2
    // elements.  The load-emit / unload-emit shaders derive their write base as
    // `taskIndex * taskSlotStride`, so one full-ring binding serves every task.
    uint32_t taskSlotStride;
    // Within-slot offset (in uint2 elements) to the unload region; the load
    // region always starts at element 0 of the slot.
    uint32_t unloadGroupsOffsetElems;
    // EXPLICIT tail padding — do not remove.  Vulkan's scalar block layout adds
    // no tail padding, so without this the SPIR-V places the next SceneStreaming
    // member (clasAllocator) at 100 rather than C++'s 104 and every allocator
    // field the VK shaders touch lands 4 bytes low.
    uint32_t _padTail;
};
#ifdef __cplusplus
static_assert(sizeof(StreamingFrameRequest) == 104, "StreamingFrameRequest must be 104 bytes");
SHADERIO_ASSERT_NO_TAIL_PADDING(StreamingFrameRequest, _padTail);
#endif

// ---------------------------------------------------------------------------
// Per-load/unload work item, processed by allocator + update shaders.
//
// On LOADS the resident IDs, cluster count, and LOD are read back out of the
// freshly streamed group blob.  On UNLOADS the patch carries the resident IDs
// itself so the CLAS free never reads the dying blob through a bindless
// descriptor.
// ---------------------------------------------------------------------------
struct StreamingPatch
{
    uint32_t geometryID;
    uint32_t groupID;

    // Pool address (convention b); srvIndex == kStreamingInvalidSrvIndex
    // encodes the unload / not-loaded sentinel.
    GroupAddress groupAddress;

    // Loads: first cluster's offset within the per-frame CLAS-build arrays.
    // Unloads: the group's clusterResidentID, from host resident bookkeeping.
    uint32_t clasBuildOffset;
    // Unloads: the group's groupResidentID.  Zero on loads.
    uint32_t unloadGroupResidentID;

    // Absolute GVA of the group blob's first byte, from which
    // stream_update_scene derives the per-cluster CLAS-build argument VAs the
    // build hardware requires.  Load-only (zero on unloads).
    uint64_t groupGpuAddress;

    // Load-only: absolute GVA of this group's position array in the transient
    // upload ring, when the resident blob was uploaded position-stripped
    // (ClusterAttribute::StrippedVertexPos).  The ring range lives until the
    // task's fence, which covers the build.  Zero otherwise.
    uint64_t positionsGpuAddress;
};
#ifdef __cplusplus
static_assert(sizeof(StreamingPatch) == 40, "StreamingPatch must be 40 bytes");
SHADERIO_ASSERT_NO_TAIL_PADDING(StreamingPatch, positionsGpuAddress);
#endif

// Upper bound on the cluster count of a geometry's cached-BLAS LoD level.
// cachedBlasClustersCount is a uint16_t, so a level wider than this can't be
// cached (host's HandleBlasCaching skips it).
static const uint32_t kStreamingCachedBlasMaxClusters = 0xFFFFu;

// ---------------------------------------------------------------------------
// Cached-BLAS patch.  One entry per geometry whose resident LoD set changed
// this frame and whose cached BLAS needs to be (re)built or invalidated.
// Written host-side by HandleBlasCaching, consumed by stream_update_scene
// (which applies cachedBlasLodLevel/cachedBlasAddress to the Geometry) and
// blas_cache_gather_clusters/copy.
// ---------------------------------------------------------------------------
struct StreamingGeometryPatch
{
    uint32_t geometryID;
    uint16_t cachedBlasLodLevel;
    uint16_t cachedBlasClustersCount;
    // Triangle count of the cached level (render-stats); 0 on invalidate.
    uint32_t cachedBlasTriangles;
    uint32_t _pad;                  // 8-byte align for cachedBlasAddress
    uint64_t cachedBlasAddress;
};
#ifdef __cplusplus
static_assert(sizeof(StreamingGeometryPatch) == 24, "StreamingGeometryPatch must be 24 bytes");
SHADERIO_ASSERT_NO_TAIL_PADDING(StreamingGeometryPatch, cachedBlasAddress);
#endif

// ---------------------------------------------------------------------------
// One entry per resident group.  The group's own header is reached through
// groupAddress (convention b), i.e.
// ResourceDescriptorHeap[srvIndex].Load<Group>(byteOffset).
// ---------------------------------------------------------------------------
struct StreamingGroup
{
    uint32_t geometryID;

    uint16_t lodLevel;
    uint16_t age;

    GroupAddress groupAddress;
};
#ifdef __cplusplus
static_assert(sizeof(StreamingGroup) == 16, "StreamingGroup must be 16 bytes");
#endif

// ---------------------------------------------------------------------------
// StreamingUpdate — per-frame change-set scalar header.  Per convention (a)
// the accompanying arrays (patches[], newClasBuilds[], newClasAddresses[],
// geometryPatches[], moveClasSrc/Dst[]) are bound as separate resources; only
// the scalars live here.
// ---------------------------------------------------------------------------
struct StreamingUpdate
{
    uint32_t patchUnloadGroupsCount;
    uint32_t patchGroupsCount;
    uint32_t patchCachedBlasCount;
    uint32_t patchCachedClustersCount;
    uint32_t loadActiveGroupsOffset;
    uint32_t loadActiveClustersOffset;
    uint32_t taskIndex;
    uint32_t frameIndex;
    uint32_t newClasCount;
    uint32_t moveClasCounter;
    uint64_t moveClasSize;                  // 8-aligned at offset 40

    // Mixed-cluster geometry-index task queue: stream_update_scene appends one
    // task per mixed alpha/two-sided cluster, stream_dispatch_setup turns the counter
    // into the indirect dispatch grid, and
    // stream_fill_clas_geometry_indices fills each task's slot with the
    // per-triangle CLAS geometry-index/flag entries.  The buffer is also bound
    // as a UAV; the raw VA is here as well because the CLAS-build hardware
    // consumes geometryIndexAndFlagsBuffer as a VA that the shader assembles.
    uint32_t newClasGeometryIndicesTaskCounter;  // offset 48
    uint32_t dispatchClasGeometryIndicesX;       // offset 52 — indirect dispatch arg .x
    uint32_t dispatchClasGeometryIndicesY;       // offset 56 — indirect dispatch arg .y
    uint32_t dispatchClasGeometryIndicesZ;       // offset 60 — indirect dispatch arg .z
    nvrhi::GpuVirtualAddress newClasGeometryIndicesBufferVA;  // offset 64

    // Scene-wide max triangles per cluster, and hence the per-task slot stride
    // into the geometry-indices buffer.  Passed at runtime so the shader stays
    // decoupled from the compile-time 256-triangle spec ceiling and the host
    // can size that buffer to the scene's actual maximum.
    uint32_t sceneMaxClusterTriangles;           // offset 72
    uint32_t _padForAlignment;                   // offset 76 — keep struct 8-aligned
};
#ifdef __cplusplus
static_assert(sizeof(StreamingUpdate) == 80, "StreamingUpdate must be 80 bytes");
SHADERIO_ASSERT_NO_TAIL_PADDING(StreamingUpdate, _padForAlignment);
#endif

// ---------------------------------------------------------------------------
// StreamingResident — resident-set scalar header.  Per convention (a) the
// accompanying arrays (activeGroups[], groups[], groupIDs[], clusters[],
// clasAddresses[], clasSizes[], groupClasSizes[], …) are separate bindings.
// ---------------------------------------------------------------------------
struct StreamingResident
{
    uint32_t activeGroupsCount;
    uint32_t activeClustersCount;
    uint32_t taskIndex;
    uint32_t frameIndex;
    // Element index where the dynamic region of the activeGroups buffer starts
    // (== the persistent low-detail prefix size).  The buffer is bound WHOLE
    // because Vulkan requires 16B-aligned storage-buffer descriptor offsets and
    // the prefix is an arbitrary group count; shaders add this base themselves.
    uint32_t persistentGroupsCount;
    uint32_t _pad0;                   // keep the uint64s 8-byte aligned
    uint64_t clasBaseAddress;         // BLAS-build seam (convention c)
    uint64_t clasMaxSize;
};
#ifdef __cplusplus
static_assert(sizeof(StreamingResident) == 40, "StreamingResident must be 40 bytes");
SHADERIO_ASSERT_NO_TAIL_PADDING(StreamingResident, clasMaxSize);
// _pad0 is interior, so the tail check above cannot see it.
static_assert(offsetof(StreamingResident, clasBaseAddress) == 24,
              "StreamingResident::clasBaseAddress must sit at offset 24 (keep _pad0)");
#endif

// ---------------------------------------------------------------------------
// StreamingResidentPersistent — CLAS-allocator scalars the GPU owns frame to
// frame.  They live in their own buffer, zero-inited once and never re-uploaded,
// because the host writes the whole SceneStreaming aggregate every
// StageResidencyUpdate and would otherwise clobber them.  Touched only by
// stream_dispatch_setup.hlsl; the host observes them through the
// StreamingFrameRequest mirror fields.
//   * clasCompactionUsedSize — prior frame's compaction byte cursor;
//     CompactionOldNoUnloads seeds moveClasSize from it so new CLAS append
//     after the resident set without re-defragging.
//   * clasAllocatedMaxSizedLeft — the persistent allocator's worst-case budget,
//     republished on no-update frames instead of a host round-trip.
// ---------------------------------------------------------------------------
struct StreamingResidentPersistent
{
    uint64_t clasCompactionUsedSize;     // compaction cursor
    uint32_t clasAllocatedMaxSizedLeft;  // persistent-allocator budget
    uint32_t _pad;                        // 8-byte align (16 bytes total)
};
#ifdef __cplusplus
static_assert(sizeof(StreamingResidentPersistent) == 16, "StreamingResidentPersistent must be 16 bytes");
SHADERIO_ASSERT_NO_TAIL_PADDING(StreamingResidentPersistent, _pad);
#endif

// ---------------------------------------------------------------------------
// CLAS build descriptor — one entry per cluster being CLAS-built this frame.
// uint64 fields are at the BLAS-build hardware seam (convention c).
//
// Bit layout of `packed`:
//   [ 8: 0]  triangleCount         (9 bits)
//   [17: 9]  vertexCount           (9 bits)
//   [23:18]  positionTruncateBits  (6 bits)
//   [27:24]  indexType             (4 bits)
//   [31:28]  opacityMicromapIndexType (4 bits)
// ---------------------------------------------------------------------------
struct ClasBuildInfo
{
    uint32_t clusterID;
    uint32_t clusterFlags;
    uint32_t packed;
    uint32_t baseGeometryIndexAndFlags;

    // [15:0] indexBufferStride, [31:16] vertexBufferStride
    uint32_t indexAndVertexBufferStrides;
    // [15:0] geometryIndexAndFlagsBufferStride, [31:16] opacityMicromapIndexBufferStride
    uint32_t geomFlagsAndOmStrides;

    // BLAS-build hardware seam — uint64 VAs (convention c).
    uint64_t indexBuffer;
    uint64_t vertexBuffer;
    uint64_t geometryIndexAndFlagsBuffer;
    uint64_t opacityMicromapArray;
    uint64_t opacityMicromapIndexBuffer;
};
#ifdef __cplusplus
static_assert(sizeof(ClasBuildInfo) == 64, "ClasBuildInfo must be 64 bytes");
SHADERIO_ASSERT_NO_TAIL_PADDING(ClasBuildInfo, opacityMicromapIndexBuffer);
#endif

// ---------------------------------------------------------------------------
// SceneStreaming — the per-frame scalar header every streaming compute shader
// binds.  ClusterLodStreaming keeps a host-side copy and writeBuffer's it
// wholesale each StageResidencyUpdate; sub-manager readbacks source from that buffer
// at the appropriate offsetof(), so this is the single source of truth.
// ---------------------------------------------------------------------------
struct SceneStreaming
{
    int32_t  ageThreshold;
    uint32_t frameIndex;
    uint32_t useBlasCaching;
    uint32_t clasPositionTruncateBits;

    // Debug override (StreamingConfig::enableMaterials).  0 makes every
    // streaming dispatch treat clusters as opaque/single-sided/material 0,
    // bypassing the alpha-mask geometryIndexAndFlagsBuffer chain.
    uint32_t materialsEnabled;

    StreamingResident  resident;
    StreamingUpdate    update;
    StreamingFrameRequest request;
    StreamingAllocator clasAllocator;
};
#ifdef __cplusplus
// No exact-size assert: nothing indexes this aggregate by stride and every host
// access into it goes through offsetof, so only the tail is layout-critical.
SHADERIO_ASSERT_NO_TAIL_PADDING(SceneStreaming, clasAllocator);
#endif

// Byte offset of update.moveClasSize within SceneStreaming.  The compaction
// shaders bump this 64-bit cursor through a raw view of the aggregate
// (RTXMG_BYTEBUFFER_ATOMIC_ADD_I64): a free-function 64-bit InterlockedAdd on a
// nested field of a RWStructuredBuffer does not lower to a correct atomic.
//
// Must be a hand-written literal — HLSL has no `offsetof` — so the static_assert
// below is what keeps it honest against layout changes.
static const uint32_t kStreamingMoveClasSizeByteOffset = 104u;
#ifdef __cplusplus
static_assert(offsetof(SceneStreaming, update) + offsetof(StreamingUpdate, moveClasSize)
              == kStreamingMoveClasSizeByteOffset,
              "kStreamingMoveClasSizeByteOffset must track the SceneStreaming layout");
#endif

} // namespace shaderio
