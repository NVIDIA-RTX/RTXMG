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

  This compute shader compacts cluster CLAS storage
  of all newly built clusters. They are appended after the
  compaction of old clusters CLAS.

  The compaction is done in `stream_compact_defrag_old.hlsl`

  A thread represents one newly built CLAS.

  Binding notes:
  --------------

  * Same binding conventions as stream_compact_defrag_old.hlsl: one
    RWStructuredBuffer<SceneStreaming> for all streaming state, everything else
    discretely bound at the unified streaming layout's fixed slots.
  * `u_NewClasResidentIDs` maps each newly built CLAS (newID = per-frame
    CLAS-build argIdx) to its scene-global clusterResidentID; stream_update_scene
    populates it alongside the IndirectTriangleClasArgs write.
  * `moveOffset = newID` (no atomic): a MOVE_OBJECTS op consumes the old-CLAS
    move pairs written by stream_compact_defrag_old before this shader runs, so
    the move arrays are reused from index 0.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "feature_gates.hlsli"
#include <donut/shaders/cluster_args.hlsli>

////////////////////////////////////////////

// Unified streaming binding layout — see streaming.cpp's InitShadersAndPipelines
// comment for the full slot map.  This shader binds the subset it touches.
RWStructuredBuffer<uint64_t>                  u_ResidentClasAddresses : register(u2);
RWStructuredBuffer<uint>                      u_ResidentClasSizes     : register(u3);
RWStructuredBuffer<uint64_t>                  u_MoveClasSrcAddresses  : register(u5);
RWStructuredBuffer<uint64_t>                  u_MoveClasDstAddresses  : register(u6);
RWStructuredBuffer<shaderio::SceneStreaming>  streamingRW             : register(u7);
RWStructuredBuffer<uint>                      u_NewClasResidentIDs    : register(u14);
StructuredBuffer<donut::cluster::OperationSize> t_NewClasSizes          : register(t1);
StructuredBuffer<uint64_t>                    t_NewClasAddresses      : register(t2);
// Raw view of the same SceneStreaming buffer for the 64-bit moveClasSize cursor
// atomic (see shaderio::kStreamingMoveClasSizeByteOffset in shaderio.h and
// RTXMG_BYTEBUFFER_ATOMIC_ADD_I64 in feature_gates.hlsli).
RWByteAddressBuffer                           u_StreamingRaw          : register(u0, space1);

////////////////////////////////////////////

[numthreads(shaderio::kStreamCompactionNewClasThreads, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    // can load pre-emptively given the array is guaranteed to be sized as multiple of kStreamCompactionNewClasThreads

    uint newID             = dtid.x;
    uint clusterResidentID = u_NewClasResidentIDs[newID];
    bool valid             = newID < streamingRW[0].update.newClasCount;

    uint     clasSize    = 0;
    uint64_t clasAddress = 0;

    if (valid)
    {
        clasSize    = uint(t_NewClasSizes[newID]);
        clasAddress = t_NewClasAddresses[newID];
    }

    int64_t clasMoveOffset;
    RTXMG_BYTEBUFFER_ATOMIC_ADD_I64(u_StreamingRaw, shaderio::kStreamingMoveClasSizeByteOffset, int64_t(clasSize), clasMoveOffset);
    uint64_t clasNewAddress = uint64_t(clasMoveOffset) + streamingRW[0].resident.clasBaseAddress;

    uint moveOffset = newID;

    if (valid) {
        // set up move to new destination
        u_MoveClasSrcAddresses[moveOffset] = clasAddress;
        u_MoveClasDstAddresses[moveOffset] = clasNewAddress;
        // update internal state of destination
        u_ResidentClasAddresses[clusterResidentID] = clasNewAddress;
        u_ResidentClasSizes[clusterResidentID]     = clasSize;
    }
}
