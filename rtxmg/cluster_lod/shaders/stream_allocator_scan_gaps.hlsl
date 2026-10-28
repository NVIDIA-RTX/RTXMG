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

  This compute shader analyzes the used clas memory and builds a list
  of free memory gaps that the alloction phase then can make use of.
  To find the free gaps the memory usage is represented in a giant
  bit array where one bit represents a memory region of a certain number of bytes
  (the granularity at which the allocatoer operates on).

  Allocation is done in `stream_allocator_alloc_groups.hlsl` and
  tags bits as used, while freeing is performed when unloading groups in
  `stream_allocator_free_groups.hlsl` and marks bits
  as unused. The unloading must be performed prior this kernel.

  This compute shader's waves look at a range of
  memory usage bits and scan them linearly to find and merge
  free gaps up to `maxAllocationSize`.

  One thread operates on one 32-bit `usedBits` value.

  We update how many gaps of a certain size exist by incrementing
  `streaming.clasAllocator.freeSizeRanges.d[freeGapSize-1].count`, which
  is later used to build the size-binned lists of gaps that the
  allocation process depends on.

  The starting positions and the sizes of free gaps are written to
  `streaming.clasAllocator.freeGapsPos` and `streaming.clasAllocator.freeGapsSize`

  We later bin the gap positions based on their sizes in the
  `stream_allocator_bin_gaps.hlsl` kernel writing out
  `streaming.clasAllocator.freeGapsPosBinned`.

  If allocations are required then the the follow-up operations to this kernel are
  the StreamSetup::AllocatorFreeInsert step within `stream_dispatch_setup.hlsl`
  and then `stream_allocator_bin_offsets.hlsl`.

  TODO potential improvement: build the un-binned freegaps in per sector lists
  and only if there was a change to the sector (triggered by load or unload).
  Then do the free gaps insertion into the global binned list on per-sector basis.
  This would avoid looking at bits and building lists of unchanged sectors.

  rtxmg port notes (D3D12, no GL_EXT_buffer_reference):
  -----------------------------------------------------

  * The allocator's sub-arrays are byte-offset regions of one
    RWByteAddressBuffer (u_AllocatorMem @ u1); the per-region base offsets live
    in streamingRW[0].clasAllocator.*ByteOffset (see StreamingAllocator in
    shaderio.h).  `streaming` (UBO) + `streamingRW` (SSBO) collapse into a
    single RWStructuredBuffer<SceneStreaming>.
  * freeGapsSize is uint16 per slot; SM6.2+ -enable-16bit-types lets us
    Store<uint16_t> on a 2-byte aligned offset.
  * HLSL has no subgroupExclusiveMax intrinsic — see WavePrefixMaxExclusive.
  * gl_SubgroupID is thread-group-LOCAL, so the wave index must be derived
    from SV_GroupThreadID, not SV_DispatchThreadID.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "rtxmg/utils/wave_size.hlsli"

////////////////////////////////////////////

RWByteAddressBuffer                          u_AllocatorMem : register(u1);
RWStructuredBuffer<shaderio::SceneStreaming> streamingRW    : register(u7);

////////////////////////////////////////////

// Debug switch; never set by the host, so default it rather than relying on the
// preprocessor's implicit "undefined == 0".
#ifndef STREAMING_DEBUG_ALWAYS_BUILD_FREEGAPS
#define STREAMING_DEBUG_ALWAYS_BUILD_FREEGAPS 0
#endif

////////////////////////////////////////////

// freeSizeRanges is an array of shaderio::AllocatorRange entries indexed by
// (size-1).  The .count field is at byte 0; stride is sizeof(AllocatorRange).

uint LoadUsedBits(uint idx)
{
    return u_AllocatorMem.Load<uint>(streamingRW[0].clasAllocator.usedBitsByteOffset + idx * 4u);
}

uint LoadUsedSectorBits(uint idx)
{
    return u_AllocatorMem.Load<uint>(streamingRW[0].clasAllocator.usedSectorBitsByteOffset + idx * 4u);
}

void StoreFreeGapPos(uint slot, uint pos)
{
    u_AllocatorMem.Store<uint>(streamingRW[0].clasAllocator.freeGapsPosByteOffset + slot * 4u, pos);
}

void StoreFreeGapSize(uint slot, uint size)
{
#if defined(TARGET_VULKAN)
    // On SPIR-V a ByteAddressBuffer is a 32-bit uint runtime array, so DXC
    // lowers Store<uint16_t> to a NON-atomic 32-bit read-modify-write.  Adjacent
    // slots (2k, 2k+1) share a word and are written concurrently, so the RMWs
    // race; disjoint-half atomics commute instead.  DXIL emits a native 2-byte
    // store, so D3D12 keeps the plain path.
    uint byteOff = streamingRW[0].clasAllocator.freeGapsSizeByteOffset + slot * 2u;
    uint wordOff = byteOff & ~3u;
    uint shift   = (byteOff & 3u) * 8u;             // 0 or 16
    uint old;
    u_AllocatorMem.InterlockedAnd(wordOff, ~(0xFFFFu << shift), old);
    u_AllocatorMem.InterlockedOr(wordOff, (size & 0xFFFFu) << shift, old);
#else
    u_AllocatorMem.Store<uint16_t>(streamingRW[0].clasAllocator.freeGapsSizeByteOffset + slot * 2u,
                                   uint16_t(size));
#endif
}

void AtomicAddFreeSizeRangeCount(uint sizeMinusOne, uint count)
{
    uint old;
    u_AllocatorMem.InterlockedAdd(
        streamingRW[0].clasAllocator.freeSizeRangesByteOffset
            + sizeMinusOne * uint(sizeof(shaderio::AllocatorRange)),
        count,
        old);
}

uint AtomicAddFreeGapsCounter(uint count)
{
    uint old;
    InterlockedAdd(streamingRW[0].clasAllocator.freeGapsCounter, count, old);
    return old;
}

void AtomicClearUsedSectorBit(uint sectorID)
{
    uint old;
    u_AllocatorMem.InterlockedAnd(
        streamingRW[0].clasAllocator.usedSectorBitsByteOffset + (sectorID / 32u) * 4u,
        ~(1u << (sectorID & 31u)),
        old);
}

////////////////////////////////////////////

// Exclusive prefix-max across the active wave.  HLSL has WavePrefixSum but
// no WavePrefixMax intrinsic — implement via Hillis-Steele scan with
// WaveReadLaneAt at clamped lane indices.  Lane 0 returns 0 (identity for
// max over non-negative uints, matching subgroupExclusiveMax of all-zero input).
uint WavePrefixMaxExclusive(uint v)
{
    uint laneCount = WaveGetLaneCount();
    uint lane      = WaveGetLaneIndex();
    // Inclusive scan first, then shift right by one lane to get exclusive.
    uint inclusive = v;
    [unroll]
    for (uint offset = 1; offset < 64; offset <<= 1)
    {
        if (offset >= laneCount) break;
        uint srcLane  = (lane >= offset) ? (lane - offset) : lane;
        uint neighbor = WaveReadLaneAt(inclusive, srcLane);
        if (lane >= offset)
        {
            inclusive = max(inclusive, neighbor);
        }
    }
    // Shift right by one lane to make the scan exclusive; lane 0 → 0.
    //
    // The shuffle MUST stay hoisted out of the ternary.  Folded in as
    // `(lane==0) ? 0 : WaveReadLaneAt(inclusive, lane-1)` it sits on the
    // `lane!=0` branch, so lane 0 is inactive when the other lanes read it and
    // lane 1 gets an undefined value.
    uint src       = (lane == 0) ? 0u : lane - 1u;
    uint shuffled  = WaveReadLaneAt(inclusive, src);
    uint exclusive = (lane == 0) ? 0u : shuffled;
    return exclusive;
}

////////////////////////////////////////////

RTXMG_WAVE_SIZE_32
[numthreads(shaderio::kStreamAllocatorBuildFreegapsThreads, 1, 1)]
void main(uint3 gid    : SV_DispatchThreadID,
          uint3 groupID: SV_GroupID,
          uint3 gtid   : SV_GroupThreadID)
{
    // Each wave operates on STREAMING_ALLOCATOR_SECTOR_SIZE many u32s
    // looping over them while linearly scanning and merging free gaps into regions up to
    // maxAllocationSize.
    //
    // waveID must be thread-group-local, hence SV_GroupThreadID: a global
    // wave index would double-count and skip half the sectors.
    // kWaveSize, not WaveGetLaneCount(): streaming.cpp sizes the dispatch
    // from the same constant, and the demo gates on the device matching it.
    const uint laneCount     = shaderio::kWaveSize;
    const uint waveCount     = shaderio::kStreamAllocatorBuildFreegapsThreads / laneCount;
    const uint waveID        = gtid.x / laneCount;
    const uint laneID        = WaveGetLaneIndex();

    const uint threadGroupID   = groupID.x;

    const uint sectorID      = threadGroupID * waveCount + waveID;
    // each sector operates on this many 32-bit values
    const uint sectorSize32  = 1u << streamingRW[0].clasAllocator.sectorSizeShift;
    // where the sector starts in the global `usedBits` array that represents the entire memory
    const uint sectorStart32 = sectorID << streamingRW[0].clasAllocator.sectorSizeShift;

    // in units and not bytes (overall within this shader we only operate in units)
    const uint maxAllocationSize = streamingRW[0].clasAllocator.maxAllocationSize;

    // when no loads are performed in this frame, then we only need the statistics
    // of the state of the free space, and not the actual gap positions
    const bool updateHasNoLoads = streamingRW[0].update.patchGroupsCount == streamingRW[0].update.patchUnloadGroupsCount
                                  && STREAMING_DEBUG_ALWAYS_BUILD_FREEGAPS == 0;

    if (sectorID >= streamingRW[0].clasAllocator.sectorCount) return;

    // Take shortcut to a simpler logic if we know the entire sector is empty.
    // INVARIANT: sectorID must stay wave-uniform, so the whole wave takes this
    // branch together and lane 0 is guaranteed active for the
    // WaveReadLaneFirst(storageStart) broadcast below.
    if ((LoadUsedSectorBits(sectorID / 32u) & (1u << (sectorID & 31u))) == 0u)
    {
        // The pre-computed number tells us how many max-sized allocations fit within a sector.
        uint maxGapsCount = streamingRW[0].clasAllocator.sectorMaxAllocationSized;
        // their might be a tail depending on the max allocation size and the sector size
        uint sizeLeft     = (32u << streamingRW[0].clasAllocator.sectorSizeShift) - (maxGapsCount * maxAllocationSize);

        // do not record gaps < shaderio::kStreamingAllocatorMinSize

        // first thread in wave reports the sizes to the atomic counters
        uint storageStart = 0;
        if (laneID == 0)
        {
            AtomicAddFreeSizeRangeCount(maxAllocationSize - 1u, maxGapsCount);
            if (sizeLeft >= shaderio::kStreamingAllocatorMinSize)
            {
                AtomicAddFreeSizeRangeCount(sizeLeft - 1u, 1u);
            }
            storageStart = AtomicAddFreeGapsCounter(maxGapsCount + (sizeLeft >= shaderio::kStreamingAllocatorMinSize ? 1u : 0u));
        }

        if (updateHasNoLoads)
        {
            // Without loads happening this frame, we don't actually need to output the detailed
            // positions of the gaps, we are just interested in the histogram.
            // Zero things here and the subsquent fill operations won't do any work.
            maxGapsCount = 0;
            sizeLeft     = 0;
        }

        // distribute filling the max gaps over the entire wave
        storageStart = WaveReadLaneFirst(storageStart);
        for (uint gap = laneID; gap < maxGapsCount; gap += laneCount)
        {
            uint freeGapPos = gap * maxAllocationSize + sectorStart32 * 32u;
            StoreFreeGapPos(storageStart + gap, freeGapPos);
            StoreFreeGapSize(storageStart + gap, maxAllocationSize);
        }

        // the tail is handled by the first thread alone
        if (laneID == 0 && sizeLeft >= shaderio::kStreamingAllocatorMinSize)
        {
            uint freeGapPos = maxGapsCount * maxAllocationSize + sectorStart32 * 32u;
            StoreFreeGapPos(storageStart + maxGapsCount, freeGapPos);
            StoreFreeGapSize(storageStart + maxGapsCount, sizeLeft);
        }

        return;
    }

    // Without the shortcut we actually have to look at all bits.

    // We distribute this loop of scanning all bits by iterating in wave-wide operations,
    // however we need some persistent state to be brought from one iteration to the next.
    uint previousIterationLastBit              = 1;
    uint previousIterationLastGlobalRangeStart = 0;

    // want to find out if the sector is acually fully empty
    // and for debugging also how many bits were set
    uint sumUsedBitsCount = 0;

    // iterate over all bits, each thread is looking at one 32 bit value and we loop over sectorSize32
    // may values with the wave in lock-step.
    for (uint idx32 = laneID; idx32 < sectorSize32; idx32 += laneCount)
    {
        bool isLastIdx = idx32 == (sectorSize32 - 1u);

        uint usedBits  = LoadUsedBits(idx32 + sectorStart32);
        uint freeBits  = ~usedBits;

        sumUsedBitsCount += countbits(usedBits);

        // some simple cases, all 32-bit are used or unused
        bool allUsed  = freeBits == 0u;
        bool allFree  = usedBits == 0u;

        // find the region of free bits within
        int freeBeginBit  = freeBits != 0u ? int(firstbitlow(freeBits)) : -1;
        int freeEndBit    = (freeBits != 0u && !allFree) ? int(firstbitlow(~(freeBits >> freeBeginBit))) - 1 + freeBeginBit : 31;
        // is our last bit free
        uint lastBit      = usedBits >> 31;

        //  We are looking for "free" regions
        //
        //   fb freeBeginBit
        //   fe freeEndBit
        //   -  marks free bit
        //   x  marks used bit
        //  | | defines boundaries of u32 we operate on, we may access
        //      the previous u32's last bit through shuffle
        //
        //  There are four states the u32 of the thread can have:
        //
        // all bits free
        //     | - - - - . . . - - - - |
        //    fb 0
        //                        fe 31
        //
        // all bits used
        //     | x x x x . . . x x x x |
        // fb -1
        //                        fe 31
        // begin partial free
        //     | x x x - . . . - - - - |
        //          fb 3
        //                        fe 31
        // end partial free
        //     | - - - x . . . x x x x |
        //    fb 0
        //        fe 2
        //
        // other scenarios, like multiple small gaps, are eliminated by design
        // and can be ignored
        //

        uint previousLastBit = WaveReadLaneAt(lastBit, (laneID == 0) ? laneID : laneID - 1u);
        if (laneID == 0) previousLastBit = previousIterationLastBit;


        // To detect the start of longer ranges that may span multiple u32s,
        // we use an exclusive max to the last begin.
        // If the u32 has a start that doesn't end within, we will pass this value to the
        // wave max.
        //
        // we start a new free region in this u32 if the previous ended used and we have a begin
        //    x | - - - - . . . - - - - |
        // or if we have a new begin within, independent of previous
        //    - | x x - - . . . - - - - |
        //    x | x x - - . . . - - - - |
        //
        // in both cases the free region must contain the last bit, to allow continuation

        uint globalRangeStart     = (((previousLastBit == 1u && freeBeginBit == 0) || freeBeginBit > 0) && freeEndBit == 31)
                                      ? (idx32 * 32u + uint(freeBeginBit)) : previousIterationLastGlobalRangeStart;
        uint lastGlobalRangeStart = WavePrefixMaxExclusive(globalRangeStart);
        if (laneID == 0) lastGlobalRangeStart = previousIterationLastGlobalRangeStart;

        // for next wave loop iteration
        previousIterationLastBit              = WaveReadLaneAt(lastBit, laneCount - 1u);
        previousIterationLastGlobalRangeStart = WaveReadLaneAt(max(lastGlobalRangeStart, globalRangeStart), laneCount - 1u);

        // Actual free range insertion is delayed until there is a transition
        // from free to used, therefore the previous last bit matters.

        // all used and previous also used, do nothing
        //
        //   x | x x x x . . . x x x x |
        //
        if (allUsed && previousLastBit == 1u) {}

        // all free, leave to next, unless isLastIdx
        //
        //   ? | - - - - . . . - - - - |
        //
        // only start region, leave to next, unless isLastIdx
        //
        //   x | x x - - . . . - - - - |
        //
        else if ((allFree || (freeBeginBit > 0 && freeEndBit == 31 && previousLastBit == 1u)) && !isLastIdx) {}

        // create a new region
        //
        //  finish previous
        //   - | x x x x . . . x x x x |
        //   - | - - - x . . . x x x x |
        //  if isLastIdx, finish continued allFree
        //   - | - - - - . . . - - - - |
        //  if isLastIdx, start & finish independent allFree
        //   x | - - - - . . . - - - - |
        //
        // Note: due to allocation size minimum of 32 it cannot happen that the very last u32
        // in a sector would require two range starts (end previous, and start & within).
        //   - | x - - - . . . - - - - |
        // we also cannot start and end within
        //   x | - - - - . . . - x x x |
        // nor have to end previous and start a new range
        //   - | x x x - . . . - - - - |
        else
        {
            uint rangeStart;
            // the previous u32 ended with a free bit, so
            // get the information where the free range started from the global range
            if (previousLastBit == 0u)
            {
                // start is from previous
                rangeStart = lastGlobalRangeStart;
            }
            else
            {
                // we start fresh within
                // strictly speaking this can only happen on isLastIdx and with freeBeginBit == 0
                rangeStart = idx32 * 32u + uint(freeBeginBit);
            }


            // allUsed means we end with previous bit (-1)
            // otherwise we end with first region within us
            uint rangeEnd  = idx32 * 32u + uint(allUsed ? -1 : freeEndBit);
            uint rangeSize = rangeEnd + 1u - rangeStart;

            uint maxGapsCount = rangeSize / maxAllocationSize;
            uint sizeLeft     = rangeSize - (maxGapsCount * maxAllocationSize);

            rangeStart += sectorStart32 * 32u;
            rangeEnd   += sectorStart32 * 32u;

            // do not record gaps < shaderio::kStreamingAllocatorMinSize

            uint gapsCount    = maxGapsCount + (sizeLeft >= shaderio::kStreamingAllocatorMinSize ? 1u : 0u);
            uint storageStart = AtomicAddFreeGapsCounter(gapsCount);

            if (maxGapsCount > 0u)
            {
                AtomicAddFreeSizeRangeCount(maxAllocationSize - 1u, maxGapsCount);
            }
            if (sizeLeft >= shaderio::kStreamingAllocatorMinSize)
            {
                AtomicAddFreeSizeRangeCount(sizeLeft - 1u, 1u);
            }

            if (updateHasNoLoads)
            {
                // Without loads happening this frame, we don't actually need to output the detailed
                // positions of the gaps, we are just interested in the histogram.
                // Zero things here and the subsquent fill operations won't do any work.
                maxGapsCount = 0;
                sizeLeft     = 0;
            }
            for (uint gap = 0u; gap < maxGapsCount; gap++)
            {
                uint freeGapPos = rangeStart + gap * maxAllocationSize;
                StoreFreeGapPos(storageStart + gap, freeGapPos);
                StoreFreeGapSize(storageStart + gap, maxAllocationSize);
            }
            if (sizeLeft >= shaderio::kStreamingAllocatorMinSize)
            {
                uint freeGapPos = rangeStart + maxGapsCount * maxAllocationSize;
                StoreFreeGapPos(storageStart + maxGapsCount, freeGapPos);
                StoreFreeGapSize(storageStart + maxGapsCount, sizeLeft);
            }
        }
    }

    if (WaveActiveAllTrue(sumUsedBitsCount == 0u))
    {
        // entire sector was empty
        AtomicClearUsedSectorBit(sectorID);
    }
}
