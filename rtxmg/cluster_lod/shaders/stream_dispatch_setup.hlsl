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

  This compute shader does a few simple operations that require only a single thread.

  shaderio::StreamSetup enumerates the various operations.  The host selects
  the branch via the `setup` push constant.

  D3D12 notes (no GL_EXT_buffer_reference):
  -----------------------------------------

  * `streaming` (UBO) + `streamingRW` (SSBO) aliases collapse into a single
    RWStructuredBuffer<SceneStreaming> (`streamingRW`).  All field access goes
    via `streamingRW[0].X`.
  * `streaming.clasAllocator.freeSizeRanges.d[i].count` becomes a Load on
    `u_AllocatorMem` at `freeSizeRangesByteOffset + i*sizeof(AllocatorRange)`.
  * `streaming.clasAllocator.stats.d.{allocatedSize,wastedSize}` becomes Loads
    on `u_AllocatorMem` at `statsByteOffset + {0,8}`.
  * The push constant carrying the branch ID lives at `register(b0)` (root
    constant in D3D12), set via `commandList->setPushConstants()` before
    dispatch.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "feature_gates.hlsli"
#include <donut/shaders/binding_helpers.hlsli>
#include <donut/shaders/cluster_args.hlsli>

////////////////////////////////////////////

// DECLARE_PUSH_CONSTANTS emits [[vk::push_constant]] on Vulkan (a real push-
// constant range, which nvrhi's PushConstants layout item maps to) and a plain
// register(b0) cbuffer on D3D12.  The static alias keeps the bare `setup` use
// sites working.
struct SetupPushData
{
    uint setup;
};
DECLARE_PUSH_CONSTANTS(SetupPushData, g_Push, 0, 0);
static const shaderio::StreamSetup setup = (shaderio::StreamSetup)g_Push.setup;

RWByteAddressBuffer                                   u_AllocatorMem           : register(u1);
RWStructuredBuffer<uint64_t>                          u_ResidentClasAddresses  : register(u2);
RWStructuredBuffer<uint>                              u_ResidentClasSizes      : register(u3);
RWStructuredBuffer<uint2>                             u_ResidentGroupClasSizes : register(u4);
RWStructuredBuffer<uint64_t>                          u_MoveClasSrcAddresses   : register(u5);
RWStructuredBuffer<uint64_t>                          u_MoveClasDstAddresses   : register(u6);
RWStructuredBuffer<shaderio::SceneStreaming>          streamingRW              : register(u7);
StructuredBuffer<shaderio::StreamingPatch>            t_Patches                : register(t0);
StructuredBuffer<donut::cluster::OperationSize>    t_NewClasSizes           : register(t1);
StructuredBuffer<uint64_t>                            t_NewClasAddresses       : register(t2);
// GPU-persistent allocator scalars (clasCompactionUsedSize /
// clasAllocatedMaxSizedLeft).  Kept out of the host-uploaded SceneStreaming so
// the GPU's values survive frame-to-frame; this shader is the only writer.
RWStructuredBuffer<shaderio::StreamingResidentPersistent> u_ResidentPersistent : register(u0, space2);

////////////////////////////////////////////

[numthreads(1, 1, 1)]
void main()
{
    if (setup == shaderio::StreamSetup::CompactionOldNoUnloads)
    {
        // we will not do compaction of old when there are no unloads.
        // However appending new still depends on the `moveClasSize` to be
        // configured correctly, so that we will append after it.

        // first streaming frame has special rule
        // (note we start at frame 1 not 0)
        if (streamingRW[0].frameIndex == 1u)
        {
            // reset the persistent stored value to zero
            u_ResidentPersistent[0].clasCompactionUsedSize = 0ull;
            streamingRW[0].update.moveClasSize             = 0ull;
        }
        else
        {
            streamingRW[0].update.moveClasSize = u_ResidentPersistent[0].clasCompactionUsedSize;
        }
    }
    else if (setup == shaderio::StreamSetup::CompactionStatus)
    {
        // move compaction for clas memory management
        if (streamingRW[0].update.patchGroupsCount > 0u)
        {
            // persistently store the total compacted clas size (GPU-owned buffer)
            u_ResidentPersistent[0].clasCompactionUsedSize = streamingRW[0].update.moveClasSize;
            // for readback
            streamingRW[0].request.clasCompactionUsedSize  = streamingRW[0].update.moveClasSize;
            streamingRW[0].request.clasCompactionCount     = streamingRW[0].update.moveClasCounter;
        }
        else
        {
            // no update, pull value from persistent storage
            streamingRW[0].request.clasCompactionUsedSize = u_ResidentPersistent[0].clasCompactionUsedSize;
            streamingRW[0].request.clasCompactionCount    = 0u;
        }
    }
    else if (setup == shaderio::StreamSetup::AllocatorFreeInsert)
    {
        uint freeGaps    = streamingRW[0].clasAllocator.freeGapsCounter;
        uint maxFreeGaps = (streamingRW[0].clasAllocator.sectorCount
                            << streamingRW[0].clasAllocator.sectorSizeShift);

        // reset to zero for `stream_allocator_bin_offsets.hlsl`
        streamingRW[0].clasAllocator.freeGapsCounter = 0u;

        // and setup actual dispatch that inserts the freegaps into the lists
        // within `stream_allocator_bin_gaps.hlsl`
        uint threadGroupCount = (min(freeGaps, maxFreeGaps) + shaderio::kStreamAllocatorFreegapsInsertThreads - 1u)
                              / shaderio::kStreamAllocatorFreegapsInsertThreads;
        streamingRW[0].clasAllocator.dispatchFreeGapsInsert.groupsX = threadGroupCount;
        streamingRW[0].clasAllocator.dispatchFreeGapsInsert.groupsY = 1u;
        streamingRW[0].clasAllocator.dispatchFreeGapsInsert.groupsZ = 1u;

    }
    else if (setup == shaderio::StreamSetup::AllocatorStatus)
    {
        if (streamingRW[0].frameIndex == 1u)
        {
            // seed all available for first frame
            uint clasAllocatedMaxSizedLeft = streamingRW[0].clasAllocator.sectorMaxAllocationSized
                                             * streamingRW[0].clasAllocator.sectorCount;
            streamingRW[0].request.clasAllocatedMaxSizedLeft = clasAllocatedMaxSizedLeft;
            u_ResidentPersistent[0].clasAllocatedMaxSizedLeft = clasAllocatedMaxSizedLeft;
        #if TRACK_RENDER_STATS
            // Reset stats baseline.  AllocatorStats = { int64 allocatedSize @+0, int64 wastedSize @+8 }.
            uint statsBase = streamingRW[0].clasAllocator.statsByteOffset;
            u_AllocatorMem.Store<int64_t>(statsBase + 0u, 0);
            int64_t wastedSize = int64_t(streamingRW[0].clasAllocator.baseWastedSize)
                                 << streamingRW[0].clasAllocator.granularityByteShift;
            u_AllocatorMem.Store<int64_t>(statsBase + 8u, wastedSize);
        #endif
        }
        else
        {
            // persistent allocator for clas memory management
            if (streamingRW[0].update.patchGroupsCount > 0u)
            {
                // count can be negative
                uint rangeSlotByteOffset = streamingRW[0].clasAllocator.freeSizeRangesByteOffset
                                         + (streamingRW[0].clasAllocator.maxAllocationSize - 1u)
                                           * uint(sizeof(shaderio::AllocatorRange));
                shaderio::AllocatorRange range = u_AllocatorMem.Load<shaderio::AllocatorRange>(rangeSlotByteOffset);
                uint clasAllocatedMaxSizedLeft = uint(max(0, range.count));
                streamingRW[0].request.clasAllocatedMaxSizedLeft  = clasAllocatedMaxSizedLeft;
                // persist the freshly recomputed budget (GPU-owned buffer)
                u_ResidentPersistent[0].clasAllocatedMaxSizedLeft = clasAllocatedMaxSizedLeft;
            }
            else
            {
                // No update this frame: republish from the GPU-persistent value.
                // Kept in u_ResidentPersistent (a buffer the host never
                // re-uploads) instead of a host round-trip.
                streamingRW[0].request.clasAllocatedMaxSizedLeft =
                    u_ResidentPersistent[0].clasAllocatedMaxSizedLeft;
            }
        }
    #if TRACK_RENDER_STATS
        uint statsBase = streamingRW[0].clasAllocator.statsByteOffset;
        streamingRW[0].request.clasAllocatedUsedSize   = uint64_t(u_AllocatorMem.Load<int64_t>(statsBase + 0u));
        streamingRW[0].request.clasAllocatedWastedSize = uint64_t(u_AllocatorMem.Load<int64_t>(statsBase + 8u));
    #endif
    }
    else if (setup == shaderio::StreamSetup::UpdateGeometryIndices)
    {
        // Indirect dispatch grid for stream_fill_clas_geometry_indices.hlsl:
        // each group handles kGeometryIndicesTasksPerGroup tasks (one
        // wave per task).
        const uint taskCount        = streamingRW[0].update.newClasGeometryIndicesTaskCounter;
        const uint threadGroupCount =
            (taskCount + shaderio::kGeometryIndicesTasksPerGroup - 1u)
                / shaderio::kGeometryIndicesTasksPerGroup;
        streamingRW[0].update.dispatchClasGeometryIndicesX = threadGroupCount;
        streamingRW[0].update.dispatchClasGeometryIndicesY = 1u;
        streamingRW[0].update.dispatchClasGeometryIndicesZ = 1u;
    }
}
