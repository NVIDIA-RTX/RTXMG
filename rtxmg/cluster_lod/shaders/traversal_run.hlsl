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

  This compute shader handles the scene's lod hierarchy traversal for all
  instances.

  Two kernels are used for the traversal.  The hierarchical node traversal is
  handled within this kernel, but the leaves (cluster groups and their
  clusters) are processed in `traversal_run_groups.hlsl`.  This reduces
  divergence and can speed things up overall.

  D3D12 bindings:
  ---------------

    * Multipass: ClusterLodPass dispatches traversal_setup + traversal_run pairs
      maxNodeTreeDepth times, advancing the window via indirectDispatchNodes /
      traversalNodeStart/End in SceneBuildingCounters.
    * geometry.nodes / nodeBboxes / streamingGroupAddresses are bindless
      (RW)StructuredBuffers at ResourceDescriptorHeap[geometry.*SRV / *UAV].
      streamingGroupAddresses carries combined residency + request-dedupe
      state: resident entries are {srvIndex, byteOffset}, non-resident entries
      are {shaderio::kStreamingInvalidSrvIndex, lastRequestedFrame}.
    * SceneStreaming is a single RWStructuredBuffer<SceneStreaming>
      (`streamingRW`); all field access goes via `streamingRW[0].X`.
    * Load requests are emitted into u_LoadGeometryGroups as
      uint2(geometryID, groupIndex).
    * Frustum culling and BLAS merging are runtime switches
      (g_Constants.useCulling / useBlasMerging); HiZ node occlusion is the
      CLUSTER_LOD_HIZ_OCCLUSION compile permutation.
*/

#pragma pack_matrix(row_major)

#include "feature_gates.hlsli"
#include "traversal_common.hlsli"

// Node-level HiZ occlusion: the depth-pyramid texture array + sampler in
// register space 1.  culling.hlsli must come AFTER traversal_common for
// ToFloat4x4.  Occluded nodes coarsen via errorScale (soft, RT-safe).
#if CLUSTER_LOD_HIZ_OCCLUSION
#include "culling.hlsli"
#include "rtxmg/utils/wave_size.hlsli"
Texture2D<float> t_ClusterLodHiZ[HIZ_MAX_LODS] : register(t0, space1);
SamplerState     s_ClusterLodHiZSampler        : register(s0, space1);
#endif

////////////////////////////////////////////
// Computes the number of children for an incoming node traversal task.
// These children are then processed within `ProcessSubTask`.
////////////////////////////////////////////

uint SetupTask(inout shaderio::TraversalInfo traversalInfo, uint readIndex, uint pass)
{
    uint subCount = Node_nodeChildCountMinusOne(traversalInfo.packedNode);
    return subCount + 1u;
}

////////////////////////////////////////////
// Wave helpers (HLSL doesn't expose ExclusiveBitCount as a single intrinsic).
////////////////////////////////////////////

uint BallotBitCount(uint4 ballot)
{
    // kWaveSize == 32 → only the .x lane carries votes.  Keep the .y/.z/.w
    // adds so this remains correct if a future SM bumps wave size to 64+.
    return countbits(ballot.x) + countbits(ballot.y) + countbits(ballot.z) + countbits(ballot.w);
}

uint BallotExclusiveBitCount(uint4 ballot, uint laneIndex)
{
    uint mask = laneIndex < 32u ? ((1u << laneIndex) - 1u) : 0xFFFFFFFFu;
    return countbits(ballot.x & mask);
}

////////////////////////////////////////////
// ProcessSubTask — primary traversal work for a single child of an input
// node.  Each thread is a child (`taskSubID`) of an incoming task
// (`taskID`); all tasks are stored in registers across the wave within
// `waveTasks`, accessed via WaveReadLaneAt.
////////////////////////////////////////////

void ProcessSubTask(const shaderio::TraversalInfo waveTasks,
                    uint taskID, uint taskSubID, bool isValid,
                    uint threadReadIndex, uint pass)
{
    // Pull required input item from the wave-register-resident task array.
    shaderio::TraversalInfo traversalInfo;
    traversalInfo.instanceID = WaveReadLaneAt(waveTasks.instanceID, taskID);
    traversalInfo.packedNode = WaveReadLaneAt(waveTasks.packedNode, taskID);

    uint instanceID  = traversalInfo.instanceID;
    uint geometryID  = t_Instances[instanceID].geometryID;
    shaderio::Geometry geometry = t_Geometries[geometryID];

    // Retrieve traversal information from the child node.
    shaderio::TraversalMetric traversalMetric;
#if CLUSTER_LOD_HIZ_OCCLUSION
    shaderio::BBox bbox;
#endif
    {
        uint childIndex     = taskSubID;
        uint childNodeIndex = Node_nodeChildOffset(traversalInfo.packedNode) + childIndex;

        StructuredBuffer<shaderio::Node> nodes =
            ResourceDescriptorHeap[NonUniformResourceIndex(geometry.nodesSRV)];
        shaderio::Node childNode = nodes[childNodeIndex];
        traversalMetric          = childNode.traversalMetric;
#if CLUSTER_LOD_HIZ_OCCLUSION
        StructuredBuffer<shaderio::BBox> nodeBboxes =
            ResourceDescriptorHeap[NonUniformResourceIndex(geometry.nodeBboxesSRV)];
        bbox = nodeBboxes[childNodeIndex];
#endif

        // Prepare to enqueue this child node later, if the metric evaluates
        // properly.
        traversalInfo.packedNode = childNode.packed;
    }

    // Perform the LOD-metric evaluation.
    float3x4 worldMatrix = t_Instances[instanceID].worldMatrix;
    float    uniformScale = ComputeUniformScale(worldMatrix);
    float4x4 instanceToEye = mul(g_Constants.traversalViewMatrix, ToFloat4x4(worldMatrix));

    float errorScale = 1.0f;
    // u_InstanceVisibility is always bound (host-cleared each frame); reading it
    // unconditionally keeps the merge skip below a runtime switch instead of a
    // compile permutation.  Packs shaderio::InstanceVisibility bits.
    uint visibilityState = u_InstanceVisibility[instanceID];
    // A soft-culled instance (visible bit clear) uses coarser LoD;
    // forced-invisible instances never reach traversal.
    bool instanceCulled = g_Constants.useCulling && !g_Constants.useHardCull
                          && (visibilityState & shaderio::InstanceVisibility::Visible) == 0u;
    bool coarsen = instanceCulled;
#if CLUSTER_LOD_HIZ_OCCLUSION
    // Node-level HiZ occlusion (soft, RT-safe): an occluded node's subtree is
    // coarsened rather than dropped, so its coarse representation stays in the
    // BLAS and secondary rays see no holes.  Skipped when the instance is
    // already culled, since errorScale is biased either way.
    if (g_Constants.useCulling && !g_Constants.useHardCull
        && !instanceCulled && g_Constants.hizNumLODs > 0u)
    {
        float4 clipMin, clipMax;
        bool   clipValid;
        bool   inFrustum = IntersectFrustum(g_Constants.cullViewProjMatrix,
                                            bbox.lo, bbox.hi, worldMatrix,
                                            clipMin, clipMax, clipValid);
        bool   nodeVisible = inFrustum && (!clipValid ||
            (IntersectSize(clipMin, clipMax, 1.0f, g_Constants.viewportf.xy)
             && IntersectHiz(clipMin, clipMax, t_ClusterLodHiZ, s_ClusterLodHiZSampler,
                             g_Constants.hizNumLODs, g_Constants.hizInvSize,
                             g_Constants.viewportf.xy)));
        coarsen = coarsen || !nodeVisible;
    }
#endif
    if (coarsen)
        errorScale = g_Constants.culledErrorScale;

    bool traverse     = TestForTraversal(instanceToEye, uniformScale, traversalMetric, errorScale);
    bool traverseNode = isValid && traverse;                    // nodes test if we can descend

    bool isGroup = Node_isGroup(traversalInfo.packedNode) != 0u;

    // Streaming residency gate — if this child is a leaf group, look up its
    // resident-pool address; bail and trigger a load request if not resident.
    if (traverseNode)
    {
        uint groupIndex = Node_groupIndex(traversalInfo.packedNode);

        // BLAS merging: a merged instance's clusters live in the geometry's
        // merged BLAS, so don't iterate this group's clusters.  It still falls
        // through to the residency gate below, which drives streaming requests
        // and the keep-alive age reset.
        if (g_Constants.useBlasMerging && isGroup
            && (visibilityState & shaderio::InstanceVisibility::UsesMerged) != 0u)
        {
            traverseNode = false;
        }

        if (isGroup)
        {
            RWStructuredBuffer<shaderio::GroupAddress> groupAddresses =
                ResourceDescriptorHeap[NonUniformResourceIndex(geometry.streamingGroupAddressesUAV)];
            shaderio::GroupAddress groupAddress = groupAddresses[groupIndex];

            if (groupAddress.srvIndex == shaderio::kStreamingInvalidSrvIndex)
            {
                // Not streamed in yet — cannot process this group.
                traverseNode = false;

                // Uncached InterlockedMax so we can test (lastFrame != thisFrame)
                // to decide whether we already requested this group this frame.
                uint lastRequestFrameIndex;
                InterlockedMax(groupAddresses[groupIndex].byteOffset,
                               streamingRW[0].frameIndex,
                               lastRequestFrameIndex);

                bool triggerRequest = lastRequestFrameIndex != streamingRW[0].frameIndex;

                uint4 voteRequested   = WaveActiveBallot(triggerRequest);
                uint  countRequested  = BallotBitCount(voteRequested);
                uint  offsetRequested = 0u;
                if (WaveIsFirstLane())
                {
                    InterlockedAdd(streamingRW[0].request.loadCounter, countRequested, offsetRequested);
                }
                offsetRequested  = WaveReadLaneFirst(offsetRequested);
                offsetRequested += BallotExclusiveBitCount(voteRequested, WaveGetLaneIndex());

                if (triggerRequest && offsetRequested < streamingRW[0].request.maxLoads)
                {
                    // Streaming work-list keyed by (geometryID, groupIndex);
                    // the host's UpdateGPU consumes it and schedules the blob
                    // upload + CLAS build.  u_LoadGeometryGroups is bound as the
                    // whole slot ring, so index off this task's slot base.
                    const uint slotBase = streamingRW[0].request.taskIndex
                                        * streamingRW[0].request.taskSlotStride;
                    u_LoadGeometryGroups[slotBase + offsetRequested] = uint2(geometryID, groupIndex);
                }
            }
            else if (g_Constants.useBlasMerging)
            {
                // BLAS-merging keep-alive.  When merging, traversal_run owns the
                // age reset for every resident group it visits and
                // traversal_run_groups skips its own — merged-instance groups
                // are never enqueued for cluster iteration, so this is their
                // only reset.
                ByteAddressBuffer groupBlob =
                    ResourceDescriptorHeap[NonUniformResourceIndex(groupAddress.srvIndex)];
                shaderio::Group group = groupBlob.Load<shaderio::Group>(groupAddress.byteOffset);
                u_ResidentGroups[group.groupResidentID].age = uint16_t(0u);
            }
        }
    }

    bool traverseGroup = isValid && traverseNode && isGroup;
    if (traverseGroup)
    {
        traverseNode = false;
    }

    // Nodes will enqueue their children again (producer); groups will be
    // enqueued for traversal_run_groups to resolve into render-clusters.
    uint4 voteNodes  = WaveActiveBallot(traverseNode);
    uint  countNodes = BallotBitCount(voteNodes);

    uint4 voteGroups  = WaveActiveBallot(traverseGroup);
    uint  countGroups = BallotBitCount(voteGroups);

    uint offsetNodes  = 0u;
    uint offsetGroups = 0u;

    if (WaveIsFirstLane())
    {
        InterlockedAdd(u_Counters[0].traversalNodeWriteCounter,  countNodes,  offsetNodes);
        InterlockedAdd(u_Counters[0].traversalGroupWriteCounter, countGroups, offsetGroups);
    }

    offsetNodes   = WaveReadLaneFirst(offsetNodes);
    offsetNodes  += BallotExclusiveBitCount(voteNodes,  WaveGetLaneIndex());
    offsetGroups  = WaveReadLaneFirst(offsetGroups);
    offsetGroups += BallotExclusiveBitCount(voteGroups, WaveGetLaneIndex());

    // Verify we actually have output space left.
    traverseNode  = traverseNode  && offsetNodes  < g_Constants.maxTraversalInfos;
    traverseGroup = traverseGroup && offsetGroups < g_Constants.maxTraversalInfos;

    // By design a thread cannot be a node and a group at the same time.
    bool doStore = traverseNode || traverseGroup;
    if (doStore)
    {
        // TraversalInfo and ClusterInfo were chosen to alias in memory as a
        // single uint2 — adjust the output address.
        uint writeIndex = traverseNode ? offsetNodes : offsetGroups;
        uint2 packed    = PackTraversalInfo(traversalInfo);

        if (traverseNode)
            u_TraversalNodeQ[writeIndex] = packed;
        else
            u_TraversalGroupQ[writeIndex] = packed;
    }
}

////////////////////////////////////////////
// Warp-packing work distribution.
//
// imagine three threads with 4,2,1 children
// looping individually means we may get poor SIMT utilization.
//
//   T0 T1 T2
//  ----------
//   A0 B0 C0
//   A1 B1
//   A2
//   A3
//
//   packing across warp
//
//   T0 T1 T2 T3 T4 T5 T6
//  ---------------------
//   A0 A1 A2 A3 B0 B1 C0
////////////////////////////////////////////

struct TaskInfo {
    uint taskID;
};

groupshared TaskInfo s_tasks[shaderio::kTraversalRunThreads];

void ProcessAllSubTasks(inout shaderio::TraversalInfo traversalInfo,
                        bool threadRunnable, int threadSubCount,
                        uint threadReadIndex, uint pass,
                        uint waveOffset)
{
    // Distribute new work across the wave.  Each task may have a variable
    // number of threads to be run; we pack them tightly over a minimum amount
    // of warp iterations.
    //
    // `waveOffset` is the wave-base index into the thread-group-scoped scratch
    // array s_tasks[].  It has to be passed in: HLSL has no portable "wave index
    // within thread group" intrinsic to recompute it here.

    int endOffset    = WavePrefixSum(threadSubCount) + threadSubCount;  // inclusive scan
    int startOffset  = endOffset - threadSubCount;
    int totalThreads = WaveReadLaneAt(endOffset, shaderio::kWaveSize - 1);
    int totalRuns    = (totalThreads + shaderio::kWaveSize - 1) / shaderio::kWaveSize;

    const uint laneIndex = WaveGetLaneIndex();

    bool  hasTask    = threadSubCount > 0;
    uint4 taskVote   = WaveActiveBallot(hasTask);
    uint  taskCount  = BallotBitCount(taskVote);
    uint  taskOffset = BallotExclusiveBitCount(taskVote, laneIndex);

    if (hasTask)
    {
        s_tasks[waveOffset + taskOffset].taskID = laneIndex;
    }

    GroupMemoryBarrierWithGroupSync();

    int taskBase = -1;
    for (int r = 0; r < totalRuns; r++)
    {
        int tFirst = r * shaderio::kWaveSize;
        int t      = tFirst + int(laneIndex);

        int relStart = startOffset - tFirst;

        // Set bit where the task starts if within the current run.
        uint startBits = WaveActiveBitOr(
            (threadRunnable && relStart >= 0 && relStart < shaderio::kWaveSize)
                ? (1u << relStart) : 0u);

        // gl_SubgroupLeMask = lanes < laneID + this lane (inclusive).
        uint leMask = (laneIndex < 31u)
                          ? ((1u << (laneIndex + 1u)) - 1u)
                          : 0xFFFFFFFFu;
        int task = int(countbits(startBits & leMask)) + taskBase;

        uint taskID       = s_tasks[waveOffset + task].taskID;
        uint taskSubID    = uint(int(t) - WaveReadLaneAt(startOffset, taskID));
        uint taskSubCount = uint(WaveReadLaneAt(threadSubCount, taskID));

        uint taskReadIndex = 0u;

        taskBase = WaveReadLaneAt(task, shaderio::kWaveSize - 1);

        bool taskValid = taskSubID < taskSubCount;
        ProcessSubTask(traversalInfo, taskID, min(taskSubID, taskSubCount - 1u),
                       taskValid, taskReadIndex, pass);
    }
}

////////////////////////////////////////////
// Multipass entry: each pass consumes [traversalNodeStart, traversalNodeEnd)
// and appends new node tasks back to the queue.  The host (ClusterLodPass::
// Execute) interleaves traversal_setup dispatches to bump the window between
// passes and reads the final group queue on the last pass.
////////////////////////////////////////////

RTXMG_WAVE_SIZE_32
[numthreads(shaderio::kTraversalRunThreads, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID,
          uint  gtid : SV_GroupIndex)
{
    // Per-wave base for the thread-group-scoped s_tasks[] partition.
    uint waveOffset = (gtid / shaderio::kWaveSize) * shaderio::kWaveSize;

    uint threadReadIndex = dtid.x + u_Counters[0].traversalNodeStart;
    bool threadRunnable  = threadReadIndex < u_Counters[0].traversalNodeEnd;
    uint pass            = u_Counters[0].traversalPass;

    shaderio::TraversalInfo nodeTraversalInfo;
    nodeTraversalInfo.instanceID = ~0u;
    nodeTraversalInfo.packedNode = ~0u;

    if (threadRunnable)
    {
        uint2 rawValue    = u_TraversalNodeQ[threadReadIndex];
        nodeTraversalInfo = UnpackTraversalInfo(rawValue);

        // The multipass window advancement fences producer against consumer, not
        // per-slot sentinels; re-check the cleared sentinel for safety.
        threadRunnable = nodeTraversalInfo.instanceID != ~0u
                      && nodeTraversalInfo.packedNode != ~0u;
    }

    // No wave-level early-out around this: ProcessAllSubTasks' group barrier has
    // to stay in thread-group-uniform control flow, and there are 4 waves per
    // thread group.  With no runnable lane the call is a no-op (totalRuns == 0).
    int threadSubCount = 0;
    if (threadRunnable)
    {
        threadSubCount = int(SetupTask(nodeTraversalInfo, threadReadIndex, pass));
    }

    ProcessAllSubTasks(nodeTraversalInfo, threadRunnable, threadSubCount,
                       threadReadIndex, pass, waveOffset);
}
