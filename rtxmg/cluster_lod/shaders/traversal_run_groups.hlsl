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

/*

  Shader Description
  ==================

  This compute shader implements the traversal of cluster groups
  in the scene. Cluster groups iterate over their children
  and test the traversal metric of their generating groups
  in the opposite direction. Depending on the result
  it will then enqueue the clusters for rendering.

  `traversal_run.hlsl` is run before and outputs
    - u_TraversalGroupQ          all traversed cluster groups that fulfill the metric.
    - traversalGroupWriteCounter number of the groups (may exceed recorded maximum).
    - indirectDispatchGroupsX    the dimensions of this kernel's dispatch based on above.

  The cluster groups fill the list of to be rendered clusters.
    - u_RenderClusters            stores all clusters that are to be rendered as a linear array.
    - renderClusterCounter        is used to append the clusters.

  One thread represents one cluster group.

  D3D12 bindings:
  ---------------

  * The per-geometry GroupAddress table and each group blob are reached through
    bindless SRVs: Geometry.streamingGroupAddressesSRV yields
    GroupAddress{srvIndex, byteOffset}, and the generating-group lookup uses
    the same table.
  * Age reset writes to the scene-global `u_ResidentGroups` at
    `group.groupResidentID`.
  * Multipass: bounds come from traversalGroupStart/End in
    SceneBuildingCounters, and the host drives one dispatch via
    `indirectDispatchGroups` on the last pass (see ClusterLodPass::Execute).
  * TraversalInfo packs as uint2 (matches u_TraversalGroupQ / u_RenderClusters
    element type).
*/

#pragma pack_matrix(row_major)

#include "feature_gates.hlsli"
#include "traversal_common.hlsli"

// Group-level HiZ occlusion: the depth-pyramid texture array + sampler in
// register space 1.  culling.hlsli must come AFTER traversal_common for
// ToFloat4x4.  Fully-occluded groups are HARD-skipped (no clusters emitted),
// which is why the skip additionally requires useHardCull at runtime.
#if CLUSTER_LOD_HIZ_OCCLUSION
#include "culling.hlsli"
#include "rtxmg/utils/wave_size.hlsli"
Texture2D<float> t_ClusterLodHiZ[HIZ_MAX_LODS] : register(t0, space1);
SamplerState     s_ClusterLodHiZSampler        : register(s0, space1);
#endif

////////////////////////////////////////////

void MainBody(uint3 gid)
{
    uint threadReadIndex = gid.x + u_Counters[0].traversalGroupStart;
    if (threadReadIndex >= u_Counters[0].traversalGroupEnd) return;

    // load group and test its clusters

    // pull required inputs (u_TraversalGroupQ stores uint2 == TraversalInfo)
    shaderio::TraversalInfo traversalInfo = UnpackTraversalInfo(u_TraversalGroupQ[threadReadIndex]);
    uint instanceID        = traversalInfo.instanceID;
    uint groupIndex        = Node_groupIndex(traversalInfo.packedNode);
    uint groupClusterCount = Node_groupClusterCountMinusOne(traversalInfo.packedNode) + 1u;

    uint geometryID            = t_Instances[instanceID].geometryID;
    shaderio::Geometry geometry = t_Geometries[geometryID];

    // retrieve traversal & culling related information from the child node or cluster
    shaderio::TraversalMetric traversalMetric;

    float3x4 worldMatrix     = t_Instances[instanceID].worldMatrix;
    float    uniformScale    = ComputeUniformScale(worldMatrix);
    float    errorScale      = 1.0f;
    // A soft-culled instance attenuates its cluster-selection LoD via
    // culledErrorScale.  u_InstanceVisibility was published this frame by
    // instance_classify_lod / traversal_init.
    if (g_Constants.useCulling && !g_Constants.useHardCull
        && (u_InstanceVisibility[instanceID] & shaderio::InstanceVisibility::Visible) == 0u)
        errorScale = g_Constants.culledErrorScale;
    float4x4 traversalMatrix = mul(g_Constants.traversalViewMatrix, ToFloat4x4(worldMatrix));

    // traversal_run ensured we never get here without ensuring residency.
    StructuredBuffer<shaderio::GroupAddress> groupAddresses =
        ResourceDescriptorHeap[NonUniformResourceIndex(geometry.streamingGroupAddressesSRV)];
    shaderio::GroupAddress groupAddress = groupAddresses[groupIndex];
    ByteAddressBuffer groupBlob =
        ResourceDescriptorHeap[NonUniformResourceIndex(groupAddress.srvIndex)];
    shaderio::Group group = groupBlob.Load<shaderio::Group>(groupAddress.byteOffset);

    // Age-reset the scene-global resident-group slot on visit; both preload and
    // streaming patch a valid groupResidentID into the blob before publishing
    // the address-table entry.  When merging, traversal_run owns this reset for
    // every resident group it visits, so skip it here to keep one owner.
    if (!g_Constants.useBlasMerging)
        u_ResidentGroups[group.groupResidentID].age = uint16_t(0u);
    uint clusterGeneratingGroupByteOffset =
        groupAddress.byteOffset +
        uint(sizeof(shaderio::Group)) +
        uint(sizeof(shaderio::Cluster)) * uint(group.clusterCount);

#if CLUSTER_LOD_HIZ_OCCLUSION
    // Group-level HiZ occlusion: an occluded group emits none of its clusters.
    // That drops them from the BLAS entirely (unlike the node-level soft coarsen
    // in traversal_run), which is RT-unsafe for secondary rays — hence the
    // useHardCull gate.  The group stays resident so disocclusion doesn't churn
    // streaming, and at worst pops in one frame late.
    if (g_Constants.useCulling && g_Constants.useHardCull && g_Constants.hizNumLODs > 0u)
    {
        float  r = group.traversalMetric.boundingSphereRadius;
        float3 c = float3(group.traversalMetric.boundingSphereX,
                          group.traversalMetric.boundingSphereY,
                          group.traversalMetric.boundingSphereZ);
        float4 clipMin, clipMax;
        bool   clipValid;
        bool   inFrustum = IntersectFrustum(g_Constants.cullViewProjMatrix,
                                            c - r, c + r, worldMatrix,
                                            clipMin, clipMax, clipValid);
        bool   groupVisible = inFrustum && (!clipValid ||
            (IntersectSize(clipMin, clipMax, 1.0f, g_Constants.viewportf.xy)
             && IntersectHiz(clipMin, clipMax, t_ClusterLodHiZ, s_ClusterLodHiZSampler,
                             g_Constants.hizNumLODs, g_Constants.hizInvSize,
                             g_Constants.viewportf.xy)));
        if (!groupVisible)
            return;
    }
#endif

    for (uint clusterIndex = 0u; clusterIndex < groupClusterCount; clusterIndex++)
    {
        bool forceCluster = false;
        bool isValid      = true;

        {
            // The continuous lod algorithm optimizes to get the lowest detail we can get away with.
            //
            // We render a cluster if its own group was traversed because it had an error
            // greater than the threshold (it is "coarse enough"). This is fulfilled when reach
            // the code here.
            //
            // However, multiple cluster groups of previous lod levels (higher detail) may cover
            // this same region. Therefore we must ensure that it's really this cluster to be drawn
            // (it is "fine enough").
            //
            // This is achieved by looking at the cluster's generating group. The generating group
            // contained the geometry that this cluster was simplified from and is from the previous,
            // lower, lod level with a lower error.
            //
            // If that group wasn't traversed then we know we must be drawn, because we have the
            // highest detail required. We use the negated result of TestForTraversal for clusters.
            //
            // If this cluster is from the highest detail level, then there is no generating group,
            // encoded by `shaderio::kOriginalMeshGroup`.  Under streaming, the generating group
            // may also not be resident; that also means this cluster is the highest detail
            // available — sentinel-checked via the generating group's GroupAddress.

            uint clusterGeneratingGroup =
                groupBlob.Load<uint>(clusterGeneratingGroupByteOffset + 4u * clusterIndex);

            bool useGeneratingMetric = false;
            if (clusterGeneratingGroup != shaderio::kOriginalMeshGroup)
            {
                shaderio::GroupAddress genGroupAddress = groupAddresses[clusterGeneratingGroup];
                if (genGroupAddress.srvIndex != shaderio::kStreamingInvalidSrvIndex)
                {
                    ByteAddressBuffer genGroupBlob =
                        ResourceDescriptorHeap[NonUniformResourceIndex(genGroupAddress.srvIndex)];
                    shaderio::Group genGroup =
                        genGroupBlob.Load<shaderio::Group>(genGroupAddress.byteOffset);
                    // generating group is resident; use its metric so the cluster only fires
                    // when this lod is "fine enough"
                    traversalMetric    = genGroup.traversalMetric;
                    useGeneratingMetric = true;
                }
            }

            if (!useGeneratingMetric)
            {
                // The generating group doesn't exist (highest detail) or isn't resident.
                // Draw this group's cluster directly.  This should always evaluate true.
                traversalMetric = group.traversalMetric;
                forceCluster    = true;
            }

            // prepare to append this cluster for rendering, if metric evaluates properly.
            // TraversalInfo aliases with ClusterInfo, packedNode == clusterID.
            traversalInfo.packedNode = group.clusterResidentID + clusterIndex;
        }

        // perform traversal & culling logic
        bool traverse         = TestForTraversal(traversalMatrix, uniformScale, traversalMetric, errorScale);
        bool renderClusterAny = isValid && (!traverse || forceCluster);  // clusters use negated test or are forced

        // The render list is filled in an unsorted manner with clusters from
        // different instances; the blas_insert_clusters kernel later builds the
        // per-BLAS list.
        uint offsetClusters = ReserveRenderClusters(renderClusterAny ? 1u : 0u);

        if (AppendRenderCluster(renderClusterAny, offsetClusters, instanceID, traversalInfo.packedNode))
        {
            // For ray tracing count how many clusters we later add to each instance / BLAS.
            // This helps determine the list length for each BLAS — blas_reserve_clusters then
            // sub-allocates space for the lists based on this counter.
            uint unused;
            InterlockedAdd(u_InstanceBuildInfos[instanceID].clusterReferencesCount, 1u, unused);

#if TRACK_RENDER_STATS
            // Render-stats: u_PerInstanceTriangles feeds the instanced total
            // (resolved per instance in instance_assign_blas); the per-clusterID
            // seen flag dedups the unique count across the instances / builds
            // that reference the same resident CLAS.
            {
                shaderio::Cluster clHdr = groupBlob.Load<shaderio::Cluster>(
                    groupAddress.byteOffset + uint(sizeof(shaderio::Group)) +
                    uint(sizeof(shaderio::Cluster)) * clusterIndex);
                uint clusterTris = uint(clHdr.triangleCountMinusOne) + 1u;

                uint prevInst;
                InterlockedAdd(u_PerInstanceTriangles[instanceID], clusterTris, prevInst);

                uint clusterID = group.clusterResidentID + clusterIndex;
                uint prevSeen;
                InterlockedExchange(u_UniqueSeenClusters[clusterID], 1u, prevSeen);
                if (prevSeen == 0u)
                {
                    uint64_t prevUnique;
                    InterlockedAdd(u_Counters[0].uniqueTriangles, (uint64_t)clusterTris, prevUnique);
                    uint prevUC;
                    InterlockedAdd(u_Counters[0].uniqueClusters, 1u, prevUC);
                }
            }
#endif
        }
    }
}

RTXMG_WAVE_SIZE_32
[numthreads(shaderio::kTraversalGroupsThreads, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    MainBody(gid);
}
