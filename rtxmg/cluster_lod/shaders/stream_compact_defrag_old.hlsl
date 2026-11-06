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
    This file is part of the compaction scheme.

  This compute shader compacts / defrags cluster CLAS storage
  of all previously active resident groups.

  A thread represents one resident group.

  Binding notes:
  --------------

  * All streaming state lives in one RWStructuredBuffer<SceneStreaming>
    (`streamingRW`); the remaining buffers are discretely bound at the unified
    streaming layout's fixed register slots (see streaming.cpp
    InitShadersAndPipelines).
  * Reaching a Group blob takes two steps: read the resident StreamingGroup at
    u0, then bindless-load the Group header from its GroupAddress{srvIndex,
    byteOffset}.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "feature_gates.hlsli"

////////////////////////////////////////////

// Unified streaming binding layout — see streaming.cpp's InitShadersAndPipelines
// comment for the full slot map.  This shader binds the subset it touches.
RWStructuredBuffer<shaderio::StreamingGroup>  u_ResidentGroups        : register(u0);
RWStructuredBuffer<uint64_t>                  u_ResidentClasAddresses : register(u2);
RWStructuredBuffer<uint>                      u_ResidentClasSizes     : register(u3);
RWStructuredBuffer<uint64_t>                  u_MoveClasSrcAddresses  : register(u5);
RWStructuredBuffer<uint64_t>                  u_MoveClasDstAddresses  : register(u6);
RWStructuredBuffer<shaderio::SceneStreaming>  streamingRW             : register(u7);
RWStructuredBuffer<uint>                      u_ActiveGroups          : register(u8);
// Raw view of the same SceneStreaming buffer for the 64-bit moveClasSize cursor
// atomic: a free-function 64-bit InterlockedAdd on the nested struct field does
// not lower to a real atomic (see shaderio::kStreamingMoveClasSizeByteOffset in
// shaderio.h and RTXMG_BYTEBUFFER_ATOMIC_ADD_I64 in feature_gates.hlsli).
RWByteAddressBuffer                           u_StreamingRaw          : register(u0, space1);
RTXMG_BYTEBUFFER_U64_ALIAS(u_StreamingRaw, register(u0, space1))

////////////////////////////////////////////

[numthreads(shaderio::kStreamCompactionOldClasThreads, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    // Pre-emptive load before the count guard: u_ActiveGroups is over-allocated
    // by one thread group past m_maxGroups (see StreamingResident::Init), so the
    // read stays in-bounds for the padding lanes.

    uint threadID        = dtid.x;
    // u_ActiveGroups is bound WHOLE (Vulkan storage-buffer descriptor offsets
    // must be 16B-aligned), so the persistent low-detail prefix is skipped
    // in-shader rather than by a sub-range binding offset.
    uint groupResidentID = u_ActiveGroups[streamingRW[0].resident.persistentGroupsCount + threadID];

    // old resident groups come first, then after this offset are the newly loaded,
    // which we can ignore here.
    bool valid           = threadID < streamingRW[0].update.loadActiveGroupsOffset;

    if (valid)
    {
        // Walk over all old resident groups' clusters and compact their clas
        // objects storage so that the newly built clas can be appended to the
        // end.

        // This will result in a lot of movement of clas and is not recommended,
        // but avoids a more sophisticated clas allocation scheme.

        shaderio::StreamingGroup residentGroup = u_ResidentGroups[groupResidentID];
        ByteAddressBuffer groupBlob =
            ResourceDescriptorHeap[NonUniformResourceIndex(residentGroup.groupAddress.srvIndex)];
        shaderio::Group group = groupBlob.Load<shaderio::Group>(residentGroup.groupAddress.byteOffset);

        // TODO improve divergence
        for (uint c = 0; c < group.clusterCount; c++)
        {
            uint clusterResidentID = group.clusterResidentID + c;

            uint clasSize        = u_ResidentClasSizes[clusterResidentID];
            uint64_t clasAddress = u_ResidentClasAddresses[clusterResidentID];

            int64_t clasMoveOffset;
            RTXMG_BYTEBUFFER_ATOMIC_ADD_I64(u_StreamingRaw, shaderio::kStreamingMoveClasSizeByteOffset, int64_t(clasSize), clasMoveOffset);
            uint64_t clasNewAddress = uint64_t(clasMoveOffset) + streamingRW[0].resident.clasBaseAddress;

            // don't move identical addresses (in reality this will hardly happen due to
            // non-deterministic nature of atomicAdd)
            bool move       = clasNewAddress != clasAddress;
            uint moveOffset;
            InterlockedAdd(streamingRW[0].update.moveClasCounter, move ? 1u : 0u, moveOffset);

            if (move) {
                // set up move to new destination
                u_MoveClasSrcAddresses[moveOffset] = clasAddress;
                u_MoveClasDstAddresses[moveOffset] = clasNewAddress;
                // update internal state of destination
                u_ResidentClasAddresses[clusterResidentID] = clasNewAddress;
                u_ResidentClasSizes[clusterResidentID]     = clasSize;
            }
        }
    }
}
