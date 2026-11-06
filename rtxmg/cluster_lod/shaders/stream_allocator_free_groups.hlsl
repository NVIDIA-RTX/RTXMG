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


    Note: The sample showcases two ways to manage CLAS memory on the device.
    One using a persistent allocator system (`stream_allocator...` files),
    and one using a simple compaction scheme (`stream_compaction...` files).
    This file is part of the allocator system.

  This compute shader handles de-allocation of clas memory space
  of unloaded groups.

  It marks the appropriate bits of the memory regions as empty again.
  `streaming.clasAllocator.usedBits` is modified accordingly.

  One thread represents an unloaded group

  Notes:
  ------

  * The allocator usedBits live as byte-offset sub-regions inside a single
    RWByteAddressBuffer (`u_AllocatorMem`); read/write with `Load<uint>` /
    `Store<uint>` and `InterlockedAnd` on the byte-addressed slot at
    `usedBitsByteOffset + i*4`.
  * The old group's blob address AND resident IDs are carried in the patch
    (host-authoritative; see the unload scheduling in streaming.cpp), which
    keeps the free fully self-contained — see the note at the read site.
    STREAMING_DEBUG_UNLOAD_PATCH_IDS adds a blob read as a cross-check only.
  * resident.clasAddresses / groupClasSizes are discrete RWStructuredBuffers
    bound at u2 / u4 in the unified streaming layout (groupClasSizes is uint2).
  * streamingRW is a single RWStructuredBuffer<SceneStreaming>; scalars are read
    as streamingRW[0].X.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "feature_gates.hlsli"

////////////////////////////////////////////

RWByteAddressBuffer                           u_AllocatorMem            : register(u1);
RTXMG_BYTEBUFFER_U64_ALIAS(u_AllocatorMem, register(u1))
RWStructuredBuffer<uint64_t>                  u_ResidentClasAddresses   : register(u2);
RWStructuredBuffer<uint>                      u_ResidentClasSizes       : register(u3);
RWStructuredBuffer<uint2>                     u_ResidentGroupClasSizes  : register(u4);
RWStructuredBuffer<shaderio::SceneStreaming>  streamingRW               : register(u7);
StructuredBuffer<shaderio::StreamingPatch>    t_Patches                 : register(t0);
// Unused here; declared so the slot map matches the unified streaming layout,
// where geometries is a UAV (stream_update_scene writes cachedBlas* into it).
RWStructuredBuffer<shaderio::Geometry>        u_Geometries              : register(u15);

////////////////////////////////////////////

[numthreads(shaderio::kStreamAllocatorUnloadGroupsThreads, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint threadID = gid.x;

    // The grid is rounded up to a kStreamAllocatorUnloadGroupsThreads
    // multiple; padding lanes have no unload patch.  Bail before reading
    // t_Patches, bound as a tight sub-range of exactly patchUnloadGroupsCount
    // entries (unloads come first in the patch list).
    if (threadID >= streamingRW[0].update.patchUnloadGroupsCount)
        return;

    shaderio::StreamingPatch spatch = t_Patches[threadID];
    // Address AND resident IDs come from the patch.  Do NOT re-read either from
    // the GPU side: stream_update_scene sentinel-writes the per-geometry
    // groupAddresses[] table in the SAME batch through bindless descriptors
    // that nvrhi cannot barrier, so any such read races it.
    shaderio::GroupAddress oldGroupAddress = spatch.groupAddress;
    uint clusterResidentID = spatch.clasBuildOffset;   // unloads: clusterResidentID
    uint groupResidentID   = spatch.unloadGroupResidentID;
    if (oldGroupAddress.srvIndex == shaderio::kStreamingInvalidSrvIndex)
    {
        // Defensive only — the host never schedules an unload without a valid
        // address.
        return;
    }

#if STREAMING_DEBUG_UNLOAD_PATCH_IDS
    // Debug cross-check: a mismatch between the blob's resident IDs and the
    // patch-carried ones means the free below would hit the WRONG CLAS region.
    // The negative errorClasDealloc code marks it as GPU-side (the host's own
    // codes are positive; any nonzero value is fatal there).
    {
        ByteAddressBuffer groupBlob =
            ResourceDescriptorHeap[NonUniformResourceIndex(oldGroupAddress.srvIndex)];
        shaderio::Group group = groupBlob.Load<shaderio::Group>(oldGroupAddress.byteOffset);
        if (group.clusterResidentID != clusterResidentID ||
            group.groupResidentID   != groupResidentID)
        {
            streamingRW[0].request.errorClasDealloc = -int(1u + threadID);
        }
    }
#endif

    // get the first clas address of the group, as all clas of a
    // group are allocated together
    uint64_t firstClasAddress = u_ResidentClasAddresses[clusterResidentID];
    // then convert this into a relative address compared to the clas base address
    uint64_t firstClasOffset  = firstClasAddress - streamingRW[0].resident.clasBaseAddress;

    // recreate the allocation properties of the group
    // get allocation position in units
    uint allocPos   = uint(firstClasOffset >> streamingRW[0].clasAllocator.granularityByteShift);
    // retrieve the size of allocation as well as the associated memory waste.
    // One uint2 per group: (allocSize units, wastedByteSize bytes).
    uint2 groupSize = u_ResidentGroupClasSizes[groupResidentID];
    // allocation size was stored in units, which is what we need here, but wasted size in bytes
    uint allocSize      = groupSize.x;
    uint wastedByteSize = groupSize.y;

#if TRACK_RENDER_STATS
    // int64 atomic on AllocatorStats { allocatedSize @ +0, wastedSize @ +8 };
    // must be the 64-bit form or the byte counter wraps at 4 GB (see
    // RTXMG_BYTEBUFFER_ATOMIC_ADD_I64 in feature_gates.hlsli).
    int64_t allocBytesDelta  = -int64_t(uint64_t(allocSize) << streamingRW[0].clasAllocator.granularityByteShift);
    int64_t wastedBytesDelta = -int64_t(wastedByteSize);
    int64_t oldStat;
    RTXMG_BYTEBUFFER_ATOMIC_ADD_I64(u_AllocatorMem, streamingRW[0].clasAllocator.statsByteOffset + 0u, allocBytesDelta,  oldStat);
    RTXMG_BYTEBUFFER_ATOMIC_ADD_I64(u_AllocatorMem, streamingRW[0].clasAllocator.statsByteOffset + 8u, wastedBytesDelta, oldStat);
#endif

    // for allocation management, tag bits as unusued
    //
    // allocPos and allocSize are in minimum granularity,
    // which is what we use to tag the appropriate bits.

    uint startPos = allocPos;
    uint endPos   = allocPos + allocSize - 1u;

    uint startBit = (startPos) & 31u;
    uint endBit   = (endPos) & 31u;

    uint start32 = startPos / 32u;
    uint end32   = endPos / 32u;

    uint startMask = ~0u;
    uint endMask   = ~0u;

    if (startBit != 0u)
    {
        startMask = ~((1u << (startBit)) - 1u);
    }
    if (endBit != 31u)
    {
        endMask = (1u << (endBit + 1u)) - 1u;
    }

    bool single32 = start32 == end32;
    if (single32)
    {
        startMask = endMask | startMask;
    }

    // start and end of an allocated region may end up in the same u32,
    // hence we need atomics for start and end

    uint usedBitsBase = streamingRW[0].clasAllocator.usedBitsByteOffset;

    uint oldMask;
    u_AllocatorMem.InterlockedAnd(usedBitsBase + start32 * 4u, ~startMask, oldMask);

    if (!single32)
    {
        // process the region that is exclusively covered by this allocation
        for (uint i = start32 + 1u; i < end32; i++)
        {
            u_AllocatorMem.Store<uint>(usedBitsBase + i * 4u, 0u);
        }

        u_AllocatorMem.InterlockedAnd(usedBitsBase + end32 * 4u, ~endMask, oldMask);
    }
}
