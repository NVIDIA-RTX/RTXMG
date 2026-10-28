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


  This compute shader handles allocation of CLAS space for newly built
  groups.  It also builds the appropriate move operations of newly built
  CLAS to their new persistent locations.

  One thread represents one newly loaded group.

  First we try to find a gap for each group based on its requested size
  (and search a bit more).  If the individual group doesn't find space,
  we will make a request over all groups that didn't find space in
  batches up to maxAllocationSize.

  Notes:
  ------

  * The 7 allocator sub-arrays are backed by one RWByteAddressBuffer
    (`u_AllocatorMem`); per-region base byte offsets live in
    `streamingRW[0].clasAllocator.XByteOffset` (see
    shaderio::StreamingAllocator).
        - freeSizeRanges:    AllocatorRange (8 B/slot, int32 count @+0,
                             uint32 offset @+4).
        - freeGapsPosBinned: uint32/slot.
        - usedBits:          uint32 bitmap.
        - usedSectorBits:    uint32 bitmap.
  * newClas{Sizes,Addresses}, moveClas{Src,Dst}Addresses and
    resident.{clasSizes,clasAddresses,groupClasSizes} are discrete
    (RW)StructuredBuffers at fixed register slots matching
    m_streamingBindingLayout in streaming.cpp.
  * Resident IDs and cluster count come from the streamed Group header via
    `spatch.groupAddress`, the same source data that traversal reads.
  * streamingRW is a single RWStructuredBuffer<SceneStreaming>; scalars are
    read as streamingRW[0].X.
  * Wave size is 32, so WaveActiveBallot payload stays in `.x` and control
    flow is driven off `.x != 0`.
  * The batched-allocation loop needs voteInLimit != 0, or firstbithigh
    returns "no bit set" and feeds WaveReadLaneAt a lane of -1.  It is
    non-empty by construction: the firstInBatch lane is itself in voteNotFound
    and, after the FindAllocation re-fit, in-limit.  That holds only while the
    host sizes `clasMaxAllocationByteSize = perClusterWorst ×
    maxClustersPerGroup`; there is no runtime guard.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "feature_gates.hlsli"
#include <donut/shaders/cluster_args.hlsli>
#include "rtxmg/utils/wave_size.hlsli"

////////////////////////////////////////////

RWByteAddressBuffer                           u_AllocatorMem            : register(u1);
RWStructuredBuffer<uint64_t>                  u_ResidentClasAddresses   : register(u2);
RWStructuredBuffer<uint>                      u_ResidentClasSizes       : register(u3);
RWStructuredBuffer<uint2>                     u_ResidentGroupClasSizes  : register(u4);
RWStructuredBuffer<uint64_t>                  u_MoveClasSrcAddresses    : register(u5);
RWStructuredBuffer<uint64_t>                  u_MoveClasDstAddresses    : register(u6);
RWStructuredBuffer<shaderio::SceneStreaming>  streamingRW               : register(u7);
StructuredBuffer<shaderio::StreamingPatch>    t_Patches                 : register(t0);
StructuredBuffer<donut::cluster::OperationSize> t_NewClasSizes            : register(t1);
StructuredBuffer<uint64_t>                    t_NewClasAddresses        : register(t2);

////////////////////////////////////////////

// groupClasSizes is one uint2 per slot: .x = allocSize (units), .y =
// wastedByteSize (bytes).

////////////////////////////////////////////

bool FindAllocation(inout uint allocSize, inout uint allocPos, uint attempts, uint requestGrowth)
{
    const uint maxAllocationSize = streamingRW[0].clasAllocator.maxAllocationSize;

    // We start out looking for a free gap with exactly the size we asked for.
    // If not succesfull we try a few more times with bigger requests.

    uint requestSize = allocSize;
    bool found = false;
    while (!found && attempts > 0 && requestSize <= maxAllocationSize)
    {
        uint rangeSlotByteOffset = streamingRW[0].clasAllocator.freeSizeRangesByteOffset
                                 + (requestSize - 1u) * uint(sizeof(shaderio::AllocatorRange));

        int idx;
        u_AllocatorMem.InterlockedAdd(rangeSlotByteOffset, uint(-1), idx);
        if (idx >= 1)
        {
            // there was a gap left, let's use it
            uint rangeOffset = u_AllocatorMem.Load<uint>(rangeSlotByteOffset + 4u);
            allocPos = u_AllocatorMem.Load<uint>(streamingRW[0].clasAllocator.freeGapsPosBinnedByteOffset
                                                 + (uint(idx - 1) + rangeOffset) * 4u);

            found = true;
        }
        else
        {
            // no gap left, try a larger one
            requestSize += requestGrowth;
            attempts--;
        }
    }

    if (found)
    {
        // we don't want to leave small unused space after the allocation behind,
        // so we associate the space with this allocation, despite some waste
        // (tracked in statistics)
        allocSize = requestSize;
    }

    return found;
}

RTXMG_WAVE_SIZE_32
[numthreads(shaderio::kStreamAllocatorLoadGroupsThreads, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    // in allocation system units, not in bytes!
    const uint maxAllocationSize    = streamingRW[0].clasAllocator.maxAllocationSize;
    // to convert between bytes and units
    const uint granularityByteShift = streamingRW[0].clasAllocator.granularityByteShift;
    const uint granularityByteMask  = (1u << granularityByteShift) - 1u;

    // loads are stored after unloads within the patch array
    const uint patchLoadGroupsCount = streamingRW[0].update.patchGroupsCount
                                    - streamingRW[0].update.patchUnloadGroupsCount;

    const uint threadID = gid.x;
    const bool valid    = threadID < patchLoadGroupsCount;

    // treat invalid as found, this avoids threads contributing to our
    // batched group allocation scheme.
    bool found            = !valid;
    uint newGroupByteSize = 0;
    uint allocSize        = 0;
    uint allocPos         = 0;
    uint newBuildOffset   = 0;
    uint groupResidentID  = 0;
    uint clusterCount     = 0;
    uint groupClusterResidentID = 0;

    if (valid)
    {
        // get details of the newly loaded groups
        shaderio::StreamingPatch spatch = t_Patches[threadID + streamingRW[0].update.patchUnloadGroupsCount];
        ByteAddressBuffer groupBlob =
            ResourceDescriptorHeap[NonUniformResourceIndex(spatch.groupAddress.srvIndex)];
        shaderio::Group group = groupBlob.Load<shaderio::Group>(spatch.groupAddress.byteOffset);
        groupResidentID        = group.groupResidentID;
        groupClusterResidentID = group.clusterResidentID;
        clusterCount           = uint(group.clusterCount);

        // First compute space of all newly built clas within a group.
        //
        // All clas are built in a canonical order that was pre-determined on
        // the CPU and offsets are encoded in the group itself.
        newBuildOffset = spatch.clasBuildOffset;
        newGroupByteSize = 0;

        for (uint c = 0; c < clusterCount; c++)
        {
            uint clasSize = uint(t_NewClasSizes[newBuildOffset + c]);
            newGroupByteSize += clasSize;
        }

        // The allocation system works in a certain byte granularity, convert
        // the request in the units of the system.
        allocSize = (newGroupByteSize + granularityByteMask) >> granularityByteShift;
        allocPos  = 0;

        // We use bit scanning to find free gaps; if allocations were < 32 bits
        // then we could have multiple tiny gaps encoded in a single u32 and
        // have to account for that, which complicates our scan logic
        // (stream_allocator_scan_gaps).  Hence easier to just waste a bit.
        // In reality there will hardly ever be waste due to this, cause a
        // group contains multiple clusters and so the sum of the allocation
        // size is almost always greater than this.
        allocSize = max(allocSize, shaderio::kStreamingAllocatorMinSize);

        // Let's look for a free space for this group.  We search a few times
        // with minimum growth, hoping most groups end up with similar sizes.
        found = FindAllocation(allocSize, allocPos, 16, 1);
    }

    // If we couldn't make an allocation individually, combine multiple groups
    // up to maxAllocationSize.
    uint4 voteNotFound = WaveActiveBallot(!found);
    if (voteNotFound.x != 0u)
    {
        uint inclusiveSum = WavePrefixSum(!found ? allocSize : 0u) + (!found ? allocSize : 0u);
        uint exclusiveSum = inclusiveSum - (!found ? allocSize : 0u);

        // we iteratively find batches that fit in the maxAllocationSize
        // example for maxAllocationSize == 8
        //
        //       invocation:  0  1  2  3  4  5  6  7  8  9 ...
        //     voteNotFound:  -  -  x  x  x  -  x  x  -  x
        //
        //             size:  0  0  1  2  4  0  3  2  0  3
        //            i.sum:  0  0  1  3  7  7 10 12 12 15
        //            e.sum:  0  0  0  1  3  7  7 10 12 12
        //
        // first batch iteration:
        //     voteNotFound:  -  -  x  x  x  -  x  x  -  x
        //            first:  2 (uniform)
        //          rebased:  -  -  1  3  7  7 10 12 12 15
        //         in limit:  -  -  x  x  x  -  -  -  -  -
        //             last:  4 (uniform)
        //     request size:  7 (last rebased)
        //   delta to first:  -  -  0  1  3  -  -  -  -  -
        //
        // second batch iteration:
        //     voteNotFound:  -  -  -  -  -  -  x  x  -  x
        //            first:  6 (uniform)
        //          rebased:  -  -  -  -  -  -  3  5  5  8
        //         in limit:  -  -  -  -  -  -  x  x  -  x
        //             last:  9 (uniform)
        //     request size:  8 (last rebased)
        //   delta to first:  -  -  -  -  -  -  0  3  5  5

        while (voteNotFound.x != 0u)
        {
            // find where the batch starts and ends
            uint firstInBatch = firstbitlow(voteNotFound.x);
            uint firstBase    = WaveReadLaneAt(exclusiveSum, firstInBatch);

            uint rebasedInclusiveSum = inclusiveSum - firstBase;

            uint4 voteInLimit = WaveActiveBallot(rebasedInclusiveSum <= maxAllocationSize && !found);
            uint  lastInBatch = firstbithigh(voteInLimit.x);

            uint requestSize  = WaveReadLaneAt(rebasedInclusiveSum, lastInBatch);
            uint requestWaste = 0;
            bool batchFound   = false;

            // now that we have a bunch of groups we allocate in this batch,
            // try to make the allocation
            if (WaveGetLaneIndex() == firstInBatch)
            {
                uint requestSizeOrig = requestSize;
                // search a bit again, this time less, and with larger growth
                batchFound = FindAllocation(requestSize, allocPos, 8, 32);
                if (!batchFound)
                {
                    // Again no luck, now we need to fall back to max
                    // allocation size.
                    //
                    // By design our allocation system guarantees that each
                    // group that is loaded can use a full maxAllocationSize
                    // gap, so this here should never fail.

                    uint rangeSlotByteOffset = streamingRW[0].clasAllocator.freeSizeRangesByteOffset
                                             + (maxAllocationSize - 1u) * uint(sizeof(shaderio::AllocatorRange));
                    int idx;
                    u_AllocatorMem.InterlockedAdd(rangeSlotByteOffset, uint(-1), idx);
                    if (idx >= 1)
                    {
                        uint rangeOffset = u_AllocatorMem.Load<shaderio::AllocatorRange>(rangeSlotByteOffset).offset;
                        allocPos = u_AllocatorMem.Load<uint>(streamingRW[0].clasAllocator.freeGapsPosBinnedByteOffset
                                                             + (uint(idx - 1) + rangeOffset) * 4u);

                        // While `FindAllocation` does associate the full gap
                        // size to the allocation request, for worst-case slot
                        // we do want to leave gaps behind, as long as they
                        // are not small.
                        if (requestSize + shaderio::kStreamingAllocatorMinSize >= maxAllocationSize)
                        {
                            requestSize = maxAllocationSize;
                        }

                        batchFound = true;
                    }
                }
                // we still might allocate a bit more than needed
                requestWaste = requestSize - requestSizeOrig;
            }

            batchFound = WaveActiveAnyTrue(batchFound);

            if (!batchFound)
            {
                // should never happen by design
                break;
            }

            // Update allocPos and allocSize for all threads within this
            // batch.  !found condition is required, as the batch can span
            // threads that already have an allocation.
            if (!found && firstInBatch <= WaveGetLaneIndex() && WaveGetLaneIndex() <= lastInBatch)
            {
                // the first thread was doing the actual allocation, get
                // details from it
                allocPos        = WaveReadLaneAt(allocPos,      firstInBatch);
                uint firstExSum = WaveReadLaneAt(exclusiveSum,  firstInBatch);
                requestWaste    = WaveReadLaneAt(requestWaste,  firstInBatch);

                // compute our relative position to the allocation position,
                // given more than one group might share this allocation.
                uint deltaToFirst = exclusiveSum - firstExSum;
                // apply the delta
                allocPos += deltaToFirst;

                // the last group in the batch will get the wasted space tail
                // of the batch allocation.
                if (WaveGetLaneIndex() == lastInBatch)
                {
                    allocSize += requestWaste;
                }

                // this group has been served
                found = true;
            }

            // for next iteration remove the groups (bits) of the current
            // batch
            voteNotFound &= ~voteInLimit;
        }
    }

    if (valid)
    {
        if (!found)
        {
            // should never happen by design, we only load new groups if there
            // is guaranteed clas allocation space left
            uint maxRangeByteOffset = streamingRW[0].clasAllocator.freeSizeRangesByteOffset
                                     + (maxAllocationSize - 1u) * uint(sizeof(shaderio::AllocatorRange));
            shaderio::AllocatorRange maxRange = u_AllocatorMem.Load<shaderio::AllocatorRange>(maxRangeByteOffset);
            streamingRW[0].request.errorClasNotFound    = int(1u + threadID);
            streamingRW[0].request.errorClasList        = maxRange.count;
            streamingRW[0].request.errorClasAlloc       = int(maxRange.offset);
            streamingRW[0].request.errorClasDealloc     = int(streamingRW[0].clasAllocator.freeGapsCounter);
            streamingRW[0].request.errorClasUsedVsAlloc = int(allocSize);
            streamingRW[0].update.moveClasCounter = 0u;
            for (uint c = 0; c < clusterCount; c++)
            {
                u_MoveClasSrcAddresses[newBuildOffset + c] = 0;
                u_MoveClasDstAddresses[newBuildOffset + c] = 0;
            }
            return;
        }

        // convert the allocation position from units back to bytes
        uint64_t groupBaseAddress = streamingRW[0].resident.clasBaseAddress
                                  + (uint64_t(allocPos) << granularityByteShift);

        // we keep some allocation information in the resident object table,
        // so we can speed up the unloading process, where we give back the
        // memory range we used.

        uint allocByteSize  = allocSize << granularityByteShift;
        uint wastedByteSize = allocByteSize - newGroupByteSize;

        // store group allocation size in units, but waste in bytes.
        // One uint2 per group: (allocSize, wastedByteSize).
        u_ResidentGroupClasSizes[groupResidentID] = uint2(allocSize, wastedByteSize);

        // then assign new clas address and fill the move operations
        for (uint c = 0; c < clusterCount; c++)
        {
            uint clusterResidentID = groupClusterResidentID + c;

            uint     clasSize    = uint(t_NewClasSizes[newBuildOffset + c]);
            uint64_t clasAddress = t_NewClasAddresses[newBuildOffset + c];

            uint64_t clasNewAddress = groupBaseAddress;

            // update persistent information in resident table
            u_ResidentClasSizes[clusterResidentID]     = clasSize;
            u_ResidentClasAddresses[clusterResidentID] = clasNewAddress;

            // setup the move of the newly built clas from the scratch
            u_MoveClasSrcAddresses[newBuildOffset + c] = clasAddress;
            // to the allocated persistent address
            u_MoveClasDstAddresses[newBuildOffset + c] = clasNewAddress;

            groupBaseAddress += uint64_t(clasSize);
        }

#if TRACK_RENDER_STATS
        // int64 atomic on AllocatorStats { allocatedSize @ +0, wastedSize @ +8 };
        // must be the 64-bit form or the byte counter wraps at 4 GB (see
        // RTXMG_BYTEBUFFER_ATOMIC_ADD_I64 in feature_gates.hlsli).
        const uint statsBase = streamingRW[0].clasAllocator.statsByteOffset;
        int64_t oldStat;
        RTXMG_BYTEBUFFER_ATOMIC_ADD_I64(u_AllocatorMem, statsBase + 0u, int64_t(allocByteSize),  oldStat);
        RTXMG_BYTEBUFFER_ATOMIC_ADD_I64(u_AllocatorMem, statsBase + 8u, int64_t(wastedByteSize), oldStat);
#endif

        // for allocation management, tag bits as used
        //
        // allocPos and allocSize are in minimum granularity, which is what we
        // use to tag the appropriate bits.

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
        u_AllocatorMem.InterlockedOr(usedBitsBase + start32 * 4u, startMask, oldMask);

        if (!single32)
        {
            // process the region that is exclusively covered by this
            // allocation
            for (uint i = start32 + 1u; i < end32; i++)
            {
                u_AllocatorMem.Store<uint>(usedBitsBase + i * 4u, ~0u);
            }

            u_AllocatorMem.InterlockedOr(usedBitsBase + end32 * 4u, endMask, oldMask);
        }

        // Tag sector is in use.  An allocation can only be within one sector.
        uint sectorID = start32 >> streamingRW[0].clasAllocator.sectorSizeShift;
        uint sectorWordOffset = streamingRW[0].clasAllocator.usedSectorBitsByteOffset
                              + (sectorID / 32u) * 4u;
        uint sectorOldMask;
        u_AllocatorMem.InterlockedOr(sectorWordOffset, 1u << (sectorID & 31u), sectorOldMask);
    }
}
