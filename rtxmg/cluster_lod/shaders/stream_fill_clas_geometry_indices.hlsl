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

// stream_fill_clas_geometry_indices.hlsl
//
// Fills the per-triangle geometryIndexAndFlagsBuffer for every
// mixed-alpha/two-sided cluster that stream_update_scene queued this frame.
// Each task slot (sceneMaxClusterTriangles uint32s) arrives holding the 2-uint32
// cluster reference (a shaderio::ClusterAddress) in slot[0..1]; this shader
// reads it into per-thread locals and then overwrites the full slot with one
// uint32 per triangle.  Safe because every thread reads slot [0..1] before any
// thread writes its own per-triangle entry.
//
// Layout: one wave per task, kGeometryIndicesTasksPerGroup tasks per
// thread group, and each wave strides its task's triangles by kWaveSize so
// sceneMaxClusterTriangles > kWaveSize still works.

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "rtxmg/cluster_lod/shaders/cluster_lod_material_resolve.hlsli"
#include <nvrhi/nvrhiHLSL.h>
#include "rtxmg/utils/wave_size.hlsli"

// Same streaming binding set as stream_update_scene.hlsl — see that file
// for the full slot map. Only the fields this shader touches are declared
// here.
RWStructuredBuffer<shaderio::SceneStreaming>                  streamingRW              : register(u7);
// Packed uint32 (low 24 = geometryIndex, high 3 = ClusterGeometryFlags), the
// same byte layout as GeometryIndexAndFlags but written as plain uint: DXC's
// bit-field-struct stores left raw material bytes in the buffer.
RWStructuredBuffer<uint>                                      u_NewClasGeometryIndices : register(u13);

RTXMG_WAVE_SIZE_32
[numthreads(shaderio::kStreamUpdateClasGeometryIndicesThreads, 1, 1)]
void main(uint3 gid : SV_GroupID, uint gti : SV_GroupIndex)
{
    const uint waveIndexInGroup = gti / shaderio::kWaveSize;
    const uint taskIdx          = gid.x * shaderio::kGeometryIndicesTasksPerGroup
                                + waveIndexInGroup;

    if (taskIdx >= streamingRW[0].update.newClasGeometryIndicesTaskCounter)
        return;

    const uint slotUint32Base =
        taskIdx * streamingRW[0].update.sceneMaxClusterTriangles;

    // Every lane in this wave loads the cluster reference identically, and
    // all those reads precede any write to [base+t] below — the lanes that later
    // iterate t={0,1} overwrite their own read locations.
    const uint srvIndex          = u_NewClasGeometryIndices[slotUint32Base + 0u];
    const uint clusterByteOffset = u_NewClasGeometryIndices[slotUint32Base + 1u];

    ByteAddressBuffer groupData =
        ResourceDescriptorHeap[NonUniformResourceIndex(srvIndex)];
    const shaderio::Cluster cluster =
        groupData.Load<shaderio::Cluster>(clusterByteOffset);

    const uint triangleCount = cluster.triangleCountMinusOne + 1u;

    // For mixed clusters the baker guarantees localMaterialID ==
    // shaderio::kPerTriangleMaterials, so the material bytes are present.
    const uint triangleMaterialsByteBase =
        ClusterGetTriangleMaterialsByteOffset(cluster, clusterByteOffset);

    for (uint t = WaveGetLaneIndex(); t < triangleCount; t += shaderio::kWaveSize)
    {
        const uint matByte =
            ClusterLoadGroupDataByte(groupData, triangleMaterialsByteBase + t);
        u_NewClasGeometryIndices[slotUint32Base + t] =
            GeometryIndexAndFlagsToUint(
                ClasEncodePerTriangleGeometryIndexAndFlags(matByte));
    }
}
