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

#include <span>
#include <donut/core/math/math.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/DescriptorTableManager.h>
#include <nvrhi/utils.h>
#include <opensubdiv/tmr/surfaceTable.h>

using namespace donut::math;

struct Shape;
class TopologyCache;
struct TopologyMap;

class SubdivisionSurface
{
public:
    // hashes the mesh topology into a TopologyCache and
    // initializes the device-side data structures corresponding to the
    // Tmr::SurfaceTables for 'vertex' and 'face-vayring' data (position &
    // texcoords).
    SubdivisionSurface(TopologyCache& topologyCache, std::unique_ptr<Shape> shape,
        const std::vector<std::unique_ptr<Shape>>& keyFrames,
        std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTableManager,
        nvrhi::ICommandList* commandList);

    bool HasAnimation() const;
    uint32_t NumKeyframes() const;

    // Posed positions of an animated mesh at one point in its keyframe track,
    // shared by every instance playing at the same frame offset.  Its positions
    // are the `range` slice of the mesh's m_posePositions / m_posePositionsPrev.
    struct Pose
    {
        uint32_t                        meshID      = 0;
        float                           frameOffset = 0.f;
        nvrhi::BufferRange              range;        // bytes
        nvrhi::BufferHandle             patchPoints;  // tessellator scratch derived from positions
        donut::engine::DescriptorHandle positionsDescriptor;
        donut::engine::DescriptorHandle positionsPrevDescriptor;
        uint32_t                        f0 = 0, f1 = 0;
        float                           blend = 0.f;
        box3                            aabb;  // object space

        // `animTime` in seconds; the track loops past the last keyframe.
        void Animate(const SubdivisionSurface& mesh, float animTime, float frameRate);
    };
    // Allocates the pose buffers, one slice per frame offset.
    std::vector<Pose> CreatePoses(uint32_t meshID, std::span<const float> frameOffsets,
                                  donut::engine::DescriptorTableManager& descriptorTable,
                                  nvrhi::ICommandList* commandList);

    uint32_t NumVertices() const;
    uint32_t SurfaceCount() const;

    // Device bytes of the mesh's own buffers; the poses' patch points are the scene's.
    uint64_t GetVramBytes() const;

    // AABBs are in object-space !
    std::vector<box3> m_aabbKeyframes;
    box3 m_aabb;

public:
    struct SurfaceTableDeviceData
    {
        nvrhi::BufferHandle surfaceDescriptors;
        nvrhi::BufferHandle controlPointIndices;

        nvrhi::BufferHandle patchPoints;
        nvrhi::BufferHandle patchPointsOffsets;
    };

    //
    // 'vertex' limit interpolation surface-table ; see :
    // https://graphics.pixar.com/opensubdiv/docs/subdivision_surfaces.html#vertex-and-varying-data
    //
    SurfaceTableDeviceData m_vertexDeviceData;

    // NumKeyframes() frames of NumVertices() positions, back to back.
    nvrhi::BufferHandle m_positionKeyframes;
    nvrhi::BufferHandle m_positionsBuffer;  // rest pose; animated instances read their Pose

    nvrhi::BufferHandle m_posePositions;
    nvrhi::BufferHandle m_posePositionsPrev;
    uint32_t            m_poseStride = 0;  // elements between pose slices

    //
    // 'face-varying' (texcoords) limit interpolation surface-table ; see :
    // https://graphics.pixar.com/opensubdiv/docs/subdivision_surfaces.html#face-varying-data-and-topology
    //
    SurfaceTableDeviceData m_texcoordDeviceData;
    nvrhi::BufferHandle m_texcoordsBuffer;
    nvrhi::BufferHandle m_surfaceToMaterialIndexBuffer; // scene-global indices, not per-mesh local ones
    nvrhi::BufferHandle m_topologyQualityBuffer;

    // Local subshape index per surface; kept past InitDeviceData() because the
    // material indices it maps through only exist once the whole scene is loaded.
    std::vector<uint16_t> m_surfaceToGeometryIndexCpu;

    donut::engine::DescriptorHandle m_vertexSurfaceDescriptorDescriptor;
    donut::engine::DescriptorHandle m_vertexControlPointIndicesDescriptor;
    donut::engine::DescriptorHandle m_positionsDescriptor;
    donut::engine::DescriptorHandle m_surfaceToMaterialIndexDescriptor;

    // Surface Types
    enum class SurfaceType : uint32_t
    {
        PureBSpline,
        RegularBSpline,
        Limit,
        NoLimit,
        Count
    };
    std::array<uint32_t, size_t(SurfaceType::Count)> m_surfaceOffsets;
    uint32_t m_surfaceCount = 0;
    
    bool m_hasDisplacementMaterial = false;
    donut::engine::DescriptorHandle m_topologyQualityDescriptor;

public:
    Shape const* GetShape() const { return m_shape.get(); }
    TopologyMap const* GetTopologyMap() const { return m_topology_map; }

    // subshapeMaterialIndices[i] is the scene-level material index for local subshape i.
    void BuildSurfaceToMaterialIndex(
        std::span<const uint32_t> subshapeMaterialIndices,
        nvrhi::ICommandList* commandList,
        std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTable);

protected:
    TopologyMap const* m_topology_map = nullptr;

    void InitDeviceData(nvrhi::ICommandList* commandList);

    std::unique_ptr<Shape> m_shape;
    uint32_t m_numKeyframes = 0;

    std::unique_ptr<const OpenSubdiv::Tmr::SurfaceTable> m_surface_table;
    std::unique_ptr<const OpenSubdiv::Tmr::LinearSurfaceTable>
        m_texcoord_surface_table;
};