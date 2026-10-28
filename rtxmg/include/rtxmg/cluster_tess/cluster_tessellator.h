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
//

#pragma once

// clang-format off
#include "rtxmg/cluster_tess/cluster_tess.h"
#include "rtxmg/cluster_tess/cluster_tess_accels.h"
#include "rtxmg/cluster_tess/tessellator_config.h"
#include "rtxmg/cluster_tess/tessellation_counters.h"
#include "rtxmg/cluster_tess/copy_cluster_offset_params.h"
#include "rtxmg/cluster_tess/fill_blas_from_clas_args_params.h"

#include "rtxmg/scene/model.h"
#include "rtxmg/utils/buffer.h"
#include "rtxmg/utils/shader_debug.h"

#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/ShaderFactory.h>
#include <nvrhi/utils.h>
#include <nvrhi/nvrhi.h>

#include <memory>
#include <vector>
#include <span>
// clang-format on

class RTXMGScene;
class SubdivisionSurface;
struct TopologyMap;

struct TemplateGridDesc
{
    uint32_t xEdges = 0;
    uint32_t yEdges = 0;
    uint32_t indexOffset = 0;
    uint32_t vertexOffset = 0;

    uint32_t getXVerts() const { return xEdges + 1; }
    uint32_t getYVerts() const { return yEdges + 1; }
    uint32_t getNumTriangles() const { return xEdges * yEdges * 2; }
    uint32_t getNumVerts() const { return getXVerts() * getYVerts(); }
};

struct TemplateGrids
{
    typedef uint8_t IndexType;
    static_assert(sizeof(IndexType) == 1 || sizeof(IndexType) == 2 || sizeof(IndexType) == 4);
    static constexpr nvrhi::rt::cluster::OperationIndexFormat kIndexFormat =
        sizeof(IndexType) == 1 ? nvrhi::rt::cluster::OperationIndexFormat::IndexFormat8bit :
        sizeof(IndexType) == 2 ? nvrhi::rt::cluster::OperationIndexFormat::IndexFormat16bit :
                                 nvrhi::rt::cluster::OperationIndexFormat::IndexFormat32bit;

    std::vector<TemplateGridDesc> descs;
    std::vector<IndexType> indices;
    std::vector<float> vertices;

    uint32_t maxVertices = 0;
    uint32_t maxTriangles = 0;
    uint32_t totalVertices = 0;
    uint32_t totalTriangles = 0;
};

enum class ShaderPermutationSurfaceType : uint32_t
{
    PureBSpline,
    RegularBSpline,
    Limit,
    All,
    Count
};

// Permutation definitions
class ComputeClusterTilingPermutation
{
public:
    static constexpr uint32_t kTessModeBitCount = 2;
    static constexpr uint32_t kVisibilityBitCount = 1;
    static constexpr uint32_t kSurfaceTypeBitCount = 2;
    static_assert(uint32_t(ShaderPermutationSurfaceType::Count) <= (1u << kSurfaceTypeBitCount));

    static_assert(uint32_t(TessellatorConfig::AdaptiveTessellationMode::COUNT) <= (1u << kTessModeBitCount));
    static_assert(uint32_t(TessellatorConfig::VisibilityMode::COUNT) <= (1u << kVisibilityBitCount));

    enum BitIndices : uint32_t
    {
        DisplacementMaps,
        FrustumVisibility,
        TessMode,
        VisibilityMode = TessMode + kTessModeBitCount,
        SurfaceTypeStartBit = VisibilityMode + kVisibilityBitCount,
        Count = SurfaceTypeStartBit + kSurfaceTypeBitCount
    };

    static constexpr size_t kCount = 1u << BitIndices::Count;

    ComputeClusterTilingPermutation(bool enableDisplacement,
        bool enableFrustumVisibility,
        TessellatorConfig::AdaptiveTessellationMode tessMode,
        TessellatorConfig::VisibilityMode visMode,
        ShaderPermutationSurfaceType surfaceType)
        : m_bits
        ((enableDisplacement ? (1u << BitIndices::DisplacementMaps) : 0u) 
         | (enableFrustumVisibility ? (1u << BitIndices::FrustumVisibility) : 0u)
         | (uint32_t(tessMode) << BitIndices::TessMode)
         | (uint32_t(visMode) << BitIndices::VisibilityMode)
         | (uint32_t(surfaceType) << BitIndices::SurfaceTypeStartBit))
    {}

    bool isDisplacementEnabled() const { return m_bits & (1u << BitIndices::DisplacementMaps); }
    bool isFrustumVisibilityEnabled() const { return m_bits & (1u << BitIndices::FrustumVisibility); }

    TessellatorConfig::AdaptiveTessellationMode tessellationMode() const
    {
        constexpr uint32_t kBitMask = (1 << kTessModeBitCount) - 1;
        return TessellatorConfig::AdaptiveTessellationMode((m_bits >> BitIndices::TessMode) & kBitMask);
    }

    TessellatorConfig::VisibilityMode visibilityMode() const
    {
        constexpr uint32_t kBitMask = (1 << kVisibilityBitCount) - 1;
        return TessellatorConfig::VisibilityMode((m_bits >> BitIndices::VisibilityMode) & kBitMask);
    }

    ShaderPermutationSurfaceType surfaceType() const
    {
        constexpr uint32_t kBitMask = (1 << kSurfaceTypeBitCount) - 1;
        return ShaderPermutationSurfaceType((m_bits >> BitIndices::SurfaceTypeStartBit) & kBitMask);
    }
    void setSurfaceType(ShaderPermutationSurfaceType surfaceType)
    {
        constexpr uint32_t kBitMask = (1 << kSurfaceTypeBitCount) - 1;
        m_bits &= ~(kBitMask << BitIndices::SurfaceTypeStartBit);
        m_bits |= (uint32_t(surfaceType) << BitIndices::SurfaceTypeStartBit);
    }

    uint32_t index() const { return m_bits; }

private:
    uint32_t m_bits = 0;
};

class FillClustersPermutation
{
public:
    static constexpr uint32_t kSurfaceTypeBitCount = 2;
    static_assert(uint32_t(ShaderPermutationSurfaceType::Count) <= (1u << kSurfaceTypeBitCount));

    enum BitIndices : uint32_t
    {
        DisplacementMaps = 0,
        VertexNormals,
        SurfaceTypeStartBit,
        Count = SurfaceTypeStartBit + kSurfaceTypeBitCount
    };
    static constexpr size_t kCount = 1u << BitIndices::Count;
    uint32_t index() const { return m_bits; }

    FillClustersPermutation(bool enableDisplacement,
        bool enableVertexNormals,
        ShaderPermutationSurfaceType surfaceType)
        : m_bits((enableDisplacement ? (1u << BitIndices::DisplacementMaps) : 0u)
         | (enableVertexNormals ? (1u << BitIndices::VertexNormals) : 0u)
         | (uint32_t(surfaceType) << BitIndices::SurfaceTypeStartBit))
    {}

    bool isDisplacementEnabled() const { return m_bits & (1u << BitIndices::DisplacementMaps); }
    bool isVertexNormalsEnabled() const { return m_bits & (1u << BitIndices::VertexNormals); }
    ShaderPermutationSurfaceType surfaceType() const
    {
        constexpr uint32_t kBitMask = (1 << kSurfaceTypeBitCount) - 1;
        return ShaderPermutationSurfaceType((m_bits >> BitIndices::SurfaceTypeStartBit) & kBitMask);
    }

private:
    uint32_t m_bits = 0;
};


class ClusterTessellator
{
public:
    ClusterTessellator(donut::engine::ShaderFactory& shaderFactory, 
        std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
        nvrhi::DescriptorTableHandle descriptorTable,
        nvrhi::DeviceHandle device);

    // Resize the AS storage for this frame's config.  Returns true if anything was
    // released, in which case the previous frame's TLAS now dangles.  Call before
    // the HiZ prepass is recorded, and only when BuildAccel will follow.
    bool ResizeAccelStorage(const RTXMGScene& scene, const TessellatorConfig& config,
        ClusterTessAccels& accels);

    void BuildAccel(const RTXMGScene& scene, const TessellatorConfig& config,
        ClusterTessAccels& accels, ClusterTessStatistics& stats, uint32_t frameIndex, nvrhi::ICommandList* commandList);

    // Device bytes outside the ClusterTessStatistics buffers: per-instance
    // indirect args, per-patch grid samplers, cluster templates and counters.
    uint64_t GetWorkingBufferBytes(const ClusterTessAccels& accels) const;

#if ENABLE_SHADER_DEBUG
    RTXMGBuffer<ShaderDebugElement>& GetDebugBuffer() { return m_debugBuffer; }
#endif

protected:
    bool UpdateMemoryAllocations(ClusterTessAccels& accels, uint32_t numInstances, uint32_t sceneSubdPatches);

    nvrhi::BufferHandle GenerateStructuredClusterTemplateArgs(const TemplateGrids& grids, nvrhi::ICommandList* commandList);
    void InitStructuredClusterTemplates(nvrhi::ICommandList* commandList);
    void BuildStructuredCLASes(ClusterTessAccels& accels, const nvrhi::BufferRange& tessCounterRange, nvrhi::ICommandList* commandList);
    void BuildBlasFromClas(ClusterTessAccels& accels, std::span<Instance const> instances, nvrhi::ICommandList* commandList);

    void FillInstantiateTemplateArgs(nvrhi::IBuffer* outArgs, nvrhi::IBuffer* templateAddresses, uint32_t numTemplates, nvrhi::ICommandList* commandList);
    void FillInstanceClusters(const RTXMGScene& scene, ClusterTessAccels& accels, nvrhi::ICommandList* commandList);
    void FillBlasFromClasArgs(nvrhi::IBuffer* outArgs, nvrhi::IBuffer* clusterOffsets, 
        nvrhi::GpuVirtualAddress clasPtrsBaseAddress, uint32_t numInstances, nvrhi::ICommandList* commandList);

    // Calculates the cluster layout based off of various visibility metrics
    // A cluster tiling is the number of clusters and cluster sizes that are used to cover a surface.
    // Outputs cluster headers, shading data, and addresses
    void ComputeInstanceClusterTiling(ClusterTessAccels& accels,
        const RTXMGScene& scene,
        uint32_t instanceIndex,
        uint32_t surfaceOffset,
        uint32_t surfaceCount,
        const nvrhi::BufferRange& tessCounterRange,
        nvrhi::ICommandList* commandList);
    void CopyClusterOffset(uint32_t instanceIndex, ClusterTessDispatchType dispatchType,
        const nvrhi::BufferRange& tessCounterRange, nvrhi::ICommandList* commandList);
    void UpdateTessRate(uint32_t tessCounterIndex, nvrhi::ICommandList* commandList);

protected:
    TessellatorConfig m_tessellatorConfig;
    donut::engine::ShaderFactory& m_shaderFactory;

    nvrhi::DeviceHandle m_device;
    nvrhi::DescriptorTableHandle m_descriptorTable;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    
    RTXMGBuffer<TessellationCounters> m_tessellationCountersBuffer;
    uint32_t m_buildAccelFrameIndex = 0; // substition for frameIndex since we don't necessarily build every frame

    // Pipeline descs
    nvrhi::BindingLayoutHandle m_bindlessBL;

    nvrhi::BindingLayoutHandle m_fillInstantiateTemplateBL;
    nvrhi::ComputePipelineHandle m_fillInstantiateTemplatePSO;

    nvrhi::BindingLayoutHandle m_fillBlasFromClasArgsBL;
    nvrhi::ComputePipelineHandle m_fillBlasFromClasArgsPSO;

    nvrhi::BindingLayoutHandle m_copyClusterOffsetBL;
    nvrhi::ComputePipelineHandle m_copyClusterOffsetPSO;

    nvrhi::BindingLayoutHandle m_updateTessRateBL;
    nvrhi::ComputePipelineHandle m_updateTessRatePSO;

    nvrhi::BindingLayoutHandle m_fillClustersBL;
    nvrhi::ComputePipelineHandle m_fillClustersPSOs[FillClustersPermutation::kCount];
    nvrhi::ComputePipelineHandle m_fillClustersTexcoordsPSO;

    nvrhi::BindingLayoutHandle m_computeClusterTilingBL;
    nvrhi::BindingLayoutHandle m_computeClusterTilingHizBL;
    nvrhi::ComputePipelineHandle m_computeClusterTilingPSOs[ComputeClusterTilingPermutation::kCount];
    
    RTXMGBuffer<uint3> m_fillClustersDispatchIndirectBuffer; // number of thread groups per each instance
    RTXMGBuffer<uint2> m_clusterOffsetCountsBuffer; // offset+count per each instance
    
    nvrhi::rt::cluster::OperationParams m_createBlasParams;
    nvrhi::rt::cluster::OperationSizeInfo m_createBlasSizeInfo;

    // Per input surface patch 
    RTXMGBuffer<GridSampler> m_gridSamplersBuffer;
    
    RTXMGBuffer<Cluster> m_clustersBuffer;
    RTXMGBuffer<nvrhi::rt::cluster::IndirectArgs> m_blasFromClasIndirectArgsBuffer;
    RTXMGBuffer<nvrhi::rt::cluster::nv::IndirectInstantiateTemplateArgs> m_clasIndirectArgDataBuffer;

    uint32_t m_numInstances = 0;
    uint32_t m_sceneSubdPatches = 0;
    uint32_t m_maxClusters = 0;
    uint32_t m_maxVertices = 0;
    uint64_t m_maxClasBytes = 0;

    struct TemplateBuffers
    {
        uint32_t                                quantNBits = 0;
        nvrhi::BufferHandle                     dataBuffer; // Holds the template data
        RTXMGBuffer<nvrhi::GpuVirtualAddress>   addressesBuffer; // Array of addresses within dataBuffer, one per template
        ClusterOperationSizesBuffer instantiationSizesBuffer; // Size to instanstiate each template
        std::vector<uint64_t> instantiationSizes;
    };
    TemplateBuffers m_templateBuffers; // Buffers used to Create templates. They are created once but need to be persistent throughout the app's run time.
    
    nvrhi::BufferHandle m_fillInstantiateTemplateArgsParamsBuffer; // constant buffer for filling indirect args for getting template sizes
    nvrhi::BufferHandle m_computeClusterTilingParamsBuffer; // constant buffer for compute cluster tiling
    nvrhi::BufferHandle m_copyClusterOffsetParamsBuffer; // constant buffer for copying cluster offsets
    nvrhi::BufferHandle m_fillClustersParamsBuffer; // constant buffer for fill clusters
    nvrhi::BufferHandle m_fillBlasFromClasArgsParamsBuffer; // constant buffer for filling indirect args to initialize blas from clas

#if ENABLE_SHADER_DEBUG
    RTXMGBuffer<ShaderDebugElement> m_debugBuffer;
#endif
};
