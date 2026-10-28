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

#include <donut/core/log.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/ShaderFactory.h>
#include <nvrhi/common/misc.h>
#include <nvrhi/utils.h>

#include <map>
#include <fstream>

#include "rtxmg/utils/bindless_layout.h"
#include "rtxmg/utils/buffer.h"

#include "rtxmg/cluster_tess/cluster_tess_accels.h"
#include "rtxmg/cluster_tess/cluster_tessellator.h"
#include "rtxmg/cluster_tess/fill_clusters_params.h"
#include "rtxmg/cluster_tess/copy_cluster_offset_params.h"
#include "rtxmg/cluster_tess/update_tess_rate_params.h"
#include "rtxmg/cluster_tess/fill_blas_from_clas_args_params.h"
#include "rtxmg/cluster_tess/fill_instantiate_template_args_params.h"
#include "rtxmg/cluster_tess/compute_cluster_tiling_params.h"
#include "rtxmg/cluster_tess/tessellation_counters.h"
#include "rtxmg/cluster_tess/tessellator_config.h"
#include "rtxmg/cluster_tess/tessellator_constants.h"

#include "rtxmg/hiz/zbuffer.h"

#include "rtxmg/profiler/statistics.h"

#include "rtxmg/scene/camera.h"
#include "rtxmg/scene/scene.h"

#include "rtxmg/subdivision/subdivision_surface.h"
#include "rtxmg/subdivision/topology_map.h"

using namespace donut;
using namespace nvrhi::rt;

constexpr uint32_t kNumTemplates = kMaxClusterEdgeSegments * kMaxClusterEdgeSegments;
constexpr uint32_t kClusterMaxTriangles = kMaxClusterEdgeSegments * kMaxClusterEdgeSegments * 2;
constexpr uint32_t kClusterMaxVertices = (kMaxClusterEdgeSegments + 1) * (kMaxClusterEdgeSegments + 1);
constexpr uint32_t kFrameCount = 4;

ClusterTessellator::ClusterTessellator(donut::engine::ShaderFactory& shaderFactory,
    std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
    nvrhi::DescriptorTableHandle descriptorTable,
    nvrhi::DeviceHandle device)
    : m_shaderFactory(shaderFactory)
    , m_descriptorTable(descriptorTable)
    , m_commonPasses(commonPasses)
    , m_device(device)
{
    // Vulkan requires BUILD_INPUT_READ_ONLY usage on the buffer the cluster-AS
    // build op reads its srcInfosCount from.
    nvrhi::BufferDesc tessellationCountersDesc =
        GetGenericDesc(kFrameCount, sizeof(TessellationCounters), "tesselation counters");
    tessellationCountersDesc.isAccelStructBuildInput = true;
    m_tessellationCountersBuffer.Create(tessellationCountersDesc, m_device);

#if ENABLE_SHADER_DEBUG
    m_debugBuffer.Create(512, "ClusterAccelDebug", m_device);
#endif


    //////////////////////////////////////////////////
    // Parameter buffers for shaders
    //////////////////////////////////////////////////
    m_fillInstantiateTemplateArgsParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(FillInstantiateTemplateArgsParams), "FillInstantiateTemplateArgsParams", engine::c_MaxRenderPassConstantBufferVersions));

    m_fillBlasFromClasArgsParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(FillBlasFromClasArgsParams), "FillBlasFromClasArgsParams", engine::c_MaxRenderPassConstantBufferVersions));

    //////////////////////////////////////////////////
    // Create common bindless binding layout
    //////////////////////////////////////////////////
    // The renderer's global descriptor table is bound against this layout, and
    // Vulkan requires the two to be compatible — hence the canonical desc.
    m_bindlessBL = m_device->createBindlessLayout(rtxmg::MakeGlobalBindlessLayoutDesc());
}

// Must match shader defines in compute_cluster_tiling.hlsl
inline char const* toString(TessellatorConfig::AdaptiveTessellationMode mode)
{
    switch (mode)
    {
    case TessellatorConfig::AdaptiveTessellationMode::UNIFORM: return "TESS_MODE_UNIFORM";
    case TessellatorConfig::AdaptiveTessellationMode::WORLD_SPACE_EDGE_LENGTH: return "TESS_MODE_WORLD_SPACE_EDGE_LENGTH";
    case TessellatorConfig::AdaptiveTessellationMode::SPHERICAL_PROJECTION: return "TESS_MODE_SPHERICAL_PROJECTION";
    default: return "UNKNOWN";
    }
}

inline char const* toString(TessellatorConfig::VisibilityMode mode)
{
    switch (mode)
    {
    case TessellatorConfig::VisibilityMode::VIS_SURFACE: return "VIS_MODE_SURFACE";
    case TessellatorConfig::VisibilityMode::VIS_LIMIT_EDGES: return "VIS_MODE_LIMIT_EDGES";
    default: return "UNKNOWN";
    }
}

constexpr auto kSurfaceTypeDefines = std::to_array<const char*>(
{
    "SURFACE_TYPE_PUREBSPLINE",
    "SURFACE_TYPE_REGULARBSPLINE",
    "SURFACE_TYPE_LIMIT",
    "SURFACE_TYPE_ALL"
});
static_assert(kSurfaceTypeDefines.size() == size_t(ShaderPermutationSurfaceType::Count));

inline char const* toString(ShaderPermutationSurfaceType surfaceType)
{
    return kSurfaceTypeDefines[uint32_t(surfaceType)];
}


void ClusterTessellator::FillInstantiateTemplateArgs(nvrhi::IBuffer* outArgs, nvrhi::IBuffer* templateAddresses, uint32_t numTemplates, nvrhi::ICommandList* commandList)
{
    FillInstantiateTemplateArgsParams params = {};
    params.numTemplates = numTemplates;
    params.pad = uint3();

    nvrhi::utils::ScopedMarker marker(commandList, "ClusterTessellator::FillInstantiateTemplateArgs");
    commandList->writeBuffer(m_fillInstantiateTemplateArgsParamsBuffer, &params, sizeof(params));


    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, templateAddresses))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, outArgs))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_fillInstantiateTemplateArgsParamsBuffer));

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(m_device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_fillInstantiateTemplateBL, bindingSet))
    {
        log::fatal("Failed to create binding set and layout for fill_instantiate_template_args.hlsl");
    }

    if (!m_fillInstantiateTemplatePSO)
    {
        nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader(
            GetClusterApiShaderPath(m_device, "cluster_tess/fill_instantiate_template_args.hlsl").c_str(), "main", nullptr, nvrhi::ShaderType::Compute);

        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(shader)
            .addBindingLayout(m_fillInstantiateTemplateBL);

        m_fillInstantiateTemplatePSO = m_device->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_fillInstantiateTemplatePSO)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    commandList->dispatch(div_ceil(numTemplates, kFillInstantiateTemplateArgsThreads), 1, 1);
}

void ClusterTessellator::FillBlasFromClasArgs(nvrhi::IBuffer* outArgs, nvrhi::IBuffer* clusterOffsets,
    nvrhi::GpuVirtualAddress clasPtrsBaseAddress, uint32_t numInstances, nvrhi::ICommandList* commandList)
{
    FillBlasFromClasArgsParams params = {};
    params.clasAddressesBaseAddress = clasPtrsBaseAddress;
    params.numInstances = numInstances;

    nvrhi::utils::ScopedMarker marker(commandList, "ClusterTessellator::FillBlasFromClasArgs");
    commandList->writeBuffer(m_fillBlasFromClasArgsParamsBuffer, &params, sizeof(params));

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, clusterOffsets))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, outArgs))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_fillBlasFromClasArgsParamsBuffer));

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(m_device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_fillBlasFromClasArgsBL, bindingSet))
    {
        log::fatal("Failed to create binding set and layout for fill_blas_from_clas_args.hlsl");
    }

    if (!m_fillBlasFromClasArgsPSO)
    {
        nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_tess/fill_blas_from_clas_args.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(shader)
            .addBindingLayout(m_fillBlasFromClasArgsBL);

        m_fillBlasFromClasArgsPSO = m_device->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_fillBlasFromClasArgsPSO)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    commandList->dispatch(div_ceil(numInstances, kFillBlasFromClasArgsThreads), 1, 1);
}

static TemplateGrids GenerateTemplateGrids()
{
    TemplateGrids result;

    // Offsets per template
    result.descs.resize(kNumTemplates);
    result.indices.reserve(kNumTemplates * kClusterMaxTriangles * 3);
    result.vertices.reserve(kNumTemplates * kClusterMaxVertices * 3);

    // Generate cluster topologies for 11x11 grid
    for (uint32_t i = 0; i < kNumTemplates; i++)
    {
        assert(i % kMaxClusterEdgeSegments < std::numeric_limits<TemplateGrids::IndexType>::max());
        assert(i / kMaxClusterEdgeSegments < std::numeric_limits<TemplateGrids::IndexType>::max());

        TemplateGridDesc& gridDesc = result.descs[i];
        gridDesc =
        {
            .xEdges = i % kMaxClusterEdgeSegments + 1,
            .yEdges = i / kMaxClusterEdgeSegments + 1,
            .indexOffset = static_cast<uint32_t>(result.indices.size() * sizeof(result.indices[0])),
            .vertexOffset = static_cast<uint32_t>(result.vertices.size() * sizeof(result.vertices[0]))
        };

        // x, y = lower - left vertex of quad
        // s: 0 is the first triangle, (left vertical edge), 1 is the second triangle (right vertical edge)
        auto TriIndices = [&gridDesc](uint32_t x, uint32_t y, uint32_t s)->std::array<uint32_t, 3>
        {
            uint32_t vs = gridDesc.getXVerts();  // vertex stride  (same as xVerts)
            uint32_t vid = y * vs + x;          // lower-left vertex id
            bool diag03 = ((x & 1) == (y & 1)); // is this triangle a 0-3 diagonal (true) or a 1-2 diagonal (false)

            assert(vid + vs + 1 < std::numeric_limits<TemplateGrids::IndexType>::max());

            // Example output for (xEdges = 3, yEdges = 1, x = {0..3}, y = 0)
            //      4_____5_____6_____7
            //      |    /|\    |    /|
            //      | a / | \ d | e / |
            //      |  /  |  \  |  /  |
            //      | / b | c \ | / f |
            //      |/____|____\|/____|
            //      0     1     2     3
            //
            //    (x,y,s)
            // a  (0,0,0) diag03 = (0, 5, 4) 
            // b  (0,0,1) diag03 = (0, 1, 5)
            // c  (1,0,0)        = (1, 2, 5)
            // d  (1,0,1)        = (2, 6, 5)
            // e  (2,0,0) diag03 = (2, 7, 6)
            // f  (2,0,1) diag03 = (2, 3, 7)

            if (diag03)
            {
                if (s == 0) return { vid, vid + 1 + vs, vid + vs };
                else        return { vid, vid + 1     , vid + 1 + vs };
            }
            else
            {
                if (s == 0) return { vid    , vid + 1     , vid + vs };
                else        return { vid + 1u, vid + 1 + vs, vid + vs };
            }
        };

        float xScale = 1.0f / gridDesc.xEdges;
        float yScale = 1.0f / gridDesc.yEdges;

        uint32_t xVerts = gridDesc.getXVerts();
        uint32_t yVerts = gridDesc.getYVerts();

        for (uint32_t y = 0; y < yVerts; y++)
        {
            for (uint32_t x = 0; x < xVerts; x++)
            {
                // Add triangles to index buffer
                if (x < gridDesc.xEdges && y < gridDesc.yEdges)
                {
                    for (uint32_t s = 0; s < 2; s++)
                    {
                        std::array<uint32_t, 3> triIndices = TriIndices(x, y, s);
                        std::transform(triIndices.begin(), triIndices.end(), std::back_inserter(result.indices), [](uint32_t e)
                            {
                                assert(e < std::numeric_limits<TemplateGrids::IndexType>::max());
                                return static_cast<TemplateGrids::IndexType>(e);
                            });
                    }
                }

                // Add verts
                result.vertices.push_back(x * xScale);
                result.vertices.push_back(y * yScale);
                result.vertices.push_back(0.0f);
            }
        }

        result.maxTriangles = std::max(result.maxTriangles, gridDesc.getNumTriangles());
        result.totalTriangles += gridDesc.getNumTriangles();

        result.maxVertices = std::max(result.maxVertices, gridDesc.getNumVerts());
        result.totalVertices += gridDesc.getNumVerts();
    }

    assert(result.maxVertices == kClusterMaxVertices);
    assert(result.maxTriangles == kClusterMaxTriangles);

    return result;
}

nvrhi::BufferHandle ClusterTessellator::GenerateStructuredClusterTemplateArgs(const TemplateGrids &grids, nvrhi::ICommandList* commandList)
{
    nvrhi::BufferDesc indexBufferDesc = {
        .byteSize = grids.indices.size() * sizeof(grids.indices[0]),
        .structStride = sizeof(grids.indices[0]),
        .debugName = "ClusterTemplateIndices",
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::AccelStructBuildInput,
        .keepInitialState = true,
    };

    nvrhi::BufferHandle indexBuffer = CreateBuffer(indexBufferDesc, m_device);
    if (grids.indices.size() > 0)
    {
        // writeBuffer is retains indexBuffer until the frame end
        commandList->writeBuffer(indexBuffer, grids.indices.data(), grids.indices.size() * sizeof(grids.indices[0]));
    }

    nvrhi::BufferDesc vertexBufferDesc = {
        .byteSize = grids.vertices.size() * sizeof(grids.vertices[0]),
        .debugName = "ClusterTemplateVertices",
        .format = nvrhi::Format::RGB32_FLOAT,
        .isVertexBuffer = true,
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::AccelStructBuildInput,
        .keepInitialState = true,
    };
    nvrhi::BufferHandle vertexBuffer = CreateBuffer(vertexBufferDesc, m_device);
    if (grids.vertices.size() > 0)
    {
        // writeBuffer is retains vertexBuffer until the frame end
        commandList->writeBuffer(vertexBuffer, grids.vertices.data(), grids.vertices.size() * sizeof(grids.vertices[0]));
    }

    nvrhi::GpuVirtualAddress indexBufferAddress = indexBuffer->getGpuVirtualAddress();
    nvrhi::GpuVirtualAddress vertexBufferAddress = vertexBuffer->getGpuVirtualAddress();

    std::vector<cluster::nv::IndirectTriangleTemplateArgs> createTemplateArgData(grids.descs.size());
    for (uint32_t i = 0; i < createTemplateArgData.size(); i++)
    {
        const TemplateGridDesc& grid = grids.descs[i];

        // Zero-initialize unused bit fields
        createTemplateArgData[i] = { };
        cluster::nv::IndirectTriangleTemplateArgs& arg = createTemplateArgData[i];
        arg.clusterId                         = 0;
        arg.clusterFlags                      = 0;
        arg.triangleCount                     = grid.getNumTriangles();
        arg.vertexCount                       = grid.getNumVerts();
        arg.positionTruncateBitCount          = 0;
        arg.indexFormat                       = uint32_t(TemplateGrids::kIndexFormat);
        arg.baseGeometryIndexAndFlags         = {};
        arg.indexBufferStride                 = sizeof(grids.indices[0]);
        arg.vertexBufferStride                = sizeof(grids.vertices[0]) * 3;
        arg.geometryIndexAndFlagsBufferStride = 0;
        arg.indexBuffer                       = indexBufferAddress + grid.indexOffset;
        arg.vertexBuffer                      = vertexBufferAddress + grid.vertexOffset;
        arg.geometryIndexAndFlagsBuffer       = 0;
        arg.instantiationBoundingBoxLimit     = 0;
    }

    nvrhi::BufferDesc clusterTemplateArgsDesc =
    {
        .debugName = "ClusterTemplateArgs",
        .isDrawIndirectArgs = true,
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::IndirectArgument,
        .keepInitialState = true,
    };

    return CreateAndUploadClusterArgs(createTemplateArgData, clusterTemplateArgsDesc, commandList);
}

void ClusterTessellator::InitStructuredClusterTemplates(nvrhi::ICommandList* commandList)
{
    // only initialize if quantNBits changes
    if (m_templateBuffers.dataBuffer.Get() != 0 &&
        m_templateBuffers.quantNBits == m_tessellatorConfig.quantNBits)
        return;

    nvrhi::utils::ScopedMarker marker(commandList, "InitStructuredClusterTemplates");
    m_templateBuffers.quantNBits = m_tessellatorConfig.quantNBits;

    TemplateGrids grids = GenerateTemplateGrids();
    
    // First compute the size of each template so we can build the address buffer
    // this will also act as the settings for further operations below.
    cluster::OperationParams operationParams =
    {
        .maxArgCount = kNumTemplates,
        .type = cluster::OperationType::ClasBuildTemplates,
        .mode = cluster::OperationMode::GetSizes,
        .flags = cluster::OperationFlags::None,
        .clas =
        {
            .vertexFormat = nvrhi::Format::RGB32_FLOAT,
            .maxGeometryIndex = 0,
            .maxUniqueGeometryCount = 1,
            .maxTriangleCount = kClusterMaxTriangles,
            .maxVertexCount = kClusterMaxVertices,
            .maxTotalTriangleCount = grids.totalTriangles,
            .maxTotalVertexCount = grids.totalVertices,
            .minPositionTruncateBitCount = m_tessellatorConfig.quantNBits,
            .indexFormat = TemplateGrids::kIndexFormat,
        }
    };
    cluster::OperationSizeInfo sizeInfo = m_device->getClusterOperationSizeInfo(operationParams);
        
    nvrhi::BufferHandle clusterTemplateArgsBuffer = GenerateStructuredClusterTemplateArgs(grids, commandList);
    
    ClusterOperationSizesBuffer templateSizesBuffer;
    templateSizesBuffer.Create(kNumTemplates, "ClusterTemplateSizes", m_device);

    cluster::OperationDesc templateGetSizesDesc =
    {
        .params = operationParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgsBuffer = clusterTemplateArgsBuffer,
        .inIndirectArgsOffsetInBytes = 0,
        .outSizesBuffer = templateSizesBuffer,
        .outSizesOffsetInBytes = 0
    };
    commandList->executeMultiIndirectClusterOperation(templateGetSizesDesc);

    // readback templateSizes
    std::vector<uint64_t> templateSizes = templateSizesBuffer.Download(commandList);
    
    if (m_tessellatorConfig.enableLogging)
    {
        templateSizesBuffer.Log(commandList);
    }

    size_t totalTemplateSize = 0;
    for (uint32_t i = 0; i < kNumTemplates; i++)
    {
        totalTemplateSize += templateSizes[i];
    }
    
    // Create template data buffer based off of totalSize of all templates
    nvrhi::BufferDesc destDataDesc = {
        .byteSize = totalTemplateSize,
        .debugName = "ClusterTemplateData",
        .canHaveUAVs = true,
        .isAccelStructStorage = true,
        .initialState = nvrhi::ResourceStates::AccelStructWrite,
        .keepInitialState = true,
    };
    m_templateBuffers.dataBuffer = CreateBuffer(destDataDesc, m_device);

    // Explicit Destination mode, calculate the address offset for each template to get a tight fit
    operationParams.type = cluster::OperationType::ClasBuildTemplates;
    operationParams.mode = cluster::OperationMode::ExplicitDestinations;

    nvrhi::GpuVirtualAddress baseAddress = m_templateBuffers.dataBuffer->getGpuVirtualAddress();
    std::vector<nvrhi::GpuVirtualAddress> addresses(kNumTemplates);
    totalTemplateSize = 0;
    for (size_t i = 0; i < addresses.size(); i++)
    {
        addresses[i] = baseAddress + totalTemplateSize;
        totalTemplateSize += templateSizes[i];
    }

    m_templateBuffers.addressesBuffer.Create(kNumTemplates, "ClusterTemplateDestAddressData", m_device);
    m_templateBuffers.addressesBuffer.Upload(addresses, commandList);
    m_templateBuffers.instantiationSizesBuffer.Create(kNumTemplates, "ClusterTemplateInstantiationSizes", m_device);

    cluster::OperationDesc createClusterTemplateDesc =
    {
        .params = operationParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgsBuffer = clusterTemplateArgsBuffer,
        .inIndirectArgsOffsetInBytes = 0,
        .inOutAddressesBuffer = m_templateBuffers.addressesBuffer,
        .inOutAddressesOffsetInBytes = 0,
        .outSizesBuffer = 0,
        .outSizesOffsetInBytes = 0,
        .outAccelerationStructuresBuffer = nullptr,
        .outAccelerationStructuresOffsetInBytes = 0
    };
    commandList->executeMultiIndirectClusterOperation(createClusterTemplateDesc);

    if (m_tessellatorConfig.enableLogging)
    {
        m_templateBuffers.addressesBuffer.Log(commandList);
    }

    // Create and fill out the instantiate args buffer from addressesBuffer
    nvrhi::BufferDesc instantiateTemplateArgsDesc = 
    {
        .byteSize = sizeof(cluster::nv::IndirectInstantiateTemplateArgs) * kNumTemplates,
        .structStride = sizeof(cluster::nv::IndirectInstantiateTemplateArgs),
        .debugName = "InstantiateTemplateArgs",
        .canHaveUAVs = true,
        .isDrawIndirectArgs = true,
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::IndirectArgument,
        .keepInitialState = true,
    };

    RTXMGBuffer<cluster::nv::IndirectInstantiateTemplateArgs> instantiateTemplateArgsBuffer(instantiateTemplateArgsDesc, m_device);
    FillInstantiateTemplateArgs(instantiateTemplateArgsBuffer, m_templateBuffers.addressesBuffer, kNumTemplates, commandList);

    if (m_tessellatorConfig.enableLogging)
    {
        instantiateTemplateArgsBuffer.Log(commandList, [](std::ostream& ss, auto e)
            {
                ss << "{ct: " << std::hex << e.clusterTemplate <<
                    " | vb: " << std::hex << e.vertexBuffer.startAddress << "}";
                return true;
            });
    }

    // Execute GetSizes mode to fill out destSizes
    operationParams.type = cluster::OperationType::ClasInstantiateTemplates;
    operationParams.mode = cluster::OperationMode::GetSizes;
    
    cluster::OperationDesc instantiateTemplateGetSizesDesc =
    {
        .params = operationParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgsBuffer = instantiateTemplateArgsBuffer,
        .inIndirectArgsOffsetInBytes = 0,
        .outSizesBuffer = m_templateBuffers.instantiationSizesBuffer,
        .outSizesOffsetInBytes = 0
    };
    commandList->executeMultiIndirectClusterOperation(instantiateTemplateGetSizesDesc);

    m_templateBuffers.instantiationSizes = m_templateBuffers.instantiationSizesBuffer.Download(commandList);

    if (m_tessellatorConfig.enableLogging)
    {
        m_templateBuffers.instantiationSizesBuffer.Log(commandList, { .wrap = false });
    }
}

void ClusterTessellator::BuildStructuredCLASes(ClusterTessAccels& accels,
    const nvrhi::BufferRange& tessCounterRange, nvrhi::ICommandList* commandList)
{
    nvrhi::utils::ScopedMarker marker(commandList, "ClusterTessellator::BuildStructuredCLASes");

    cluster::OperationParams instantiateClasParams =
    {
        .maxArgCount = m_maxClusters,
        .type = cluster::OperationType::ClasInstantiateTemplates,
        .mode = cluster::OperationMode::ExplicitDestinations,
        .flags = cluster::OperationFlags::None,
        .clas =
        {
            .vertexFormat = nvrhi::Format::RGB32_FLOAT,
            .maxGeometryIndex = 0,
            .maxUniqueGeometryCount = 1,
            .maxTriangleCount = kClusterMaxTriangles,
            .maxVertexCount = kClusterMaxVertices,
            .maxTotalTriangleCount = m_maxClusters * kClusterMaxTriangles,
            .maxTotalVertexCount = m_maxVertices,
            .minPositionTruncateBitCount = m_tessellatorConfig.quantNBits,
        }
    };

    cluster::OperationSizeInfo sizeInfo = m_device->getClusterOperationSizeInfo(instantiateClasParams);
    cluster::OperationDesc instantiateClasDesc =
    {
        .params = instantiateClasParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgCountBuffer = m_tessellationCountersBuffer,
        .inIndirectArgCountOffsetInBytes = tessCounterRange.byteOffset + kClusterCountByteOffset,
        .inIndirectArgsBuffer = m_clasIndirectArgDataBuffer,
        .inIndirectArgsOffsetInBytes = 0,
        .inOutAddressesBuffer = accels.clasPtrsBuffer,
        .inOutAddressesOffsetInBytes = 0,
        .outSizesBuffer = nullptr,
        .outSizesOffsetInBytes = 0,
        .outAccelerationStructuresBuffer = nullptr,
        .outAccelerationStructuresOffsetInBytes = 0
    };

    commandList->executeMultiIndirectClusterOperation(instantiateClasDesc);
}

void ClusterTessellator::FillInstanceClusters(const RTXMGScene& scene, ClusterTessAccels& accels, nvrhi::ICommandList* commandList)
{    
    const auto& subdMeshes = scene.GetSubdMeshes();
    const auto& instances = scene.GetSubdMeshInstances();

    nvrhi::utils::ScopedMarker marker(commandList, "FillInstanceClusters");
    stats::clusterAccelSamplers.fillClustersTime.Start(commandList);

    uint32_t surfaceOffset{ 0 };
    for (uint32_t instanceIndex = 0; instanceIndex < instances.size(); ++instanceIndex)
    {
        const auto& instance = instances[instanceIndex];
        const auto& subd = *subdMeshes[instance.meshID];
        
        const uint32_t surfaceCount = subd.SurfaceCount();

#if ENABLE_SHADER_DEBUG
        if (m_tessellatorConfig.debugSurfaceIndex >= 0 &&
            m_tessellatorConfig.debugClusterIndex >= 0 &&
            m_tessellatorConfig.debugLaneIndex >= 0)
        {
            commandList->clearBufferUInt(m_debugBuffer, 0);
        }
#endif

        FillClustersParams params = {};
        params.instanceIndex = instanceIndex;
        params.quantNBits = m_tessellatorConfig.quantNBits;
        params.isolationLevel = m_tessellatorConfig.isolationLevel;
        params.globalDisplacementScale = m_tessellatorConfig.displacementScale;
        params.clusterPattern = uint32_t(m_tessellatorConfig.clusterPattern);
        params.debugSurfaceIndex = uint32_t(m_tessellatorConfig.debugSurfaceIndex);
        params.debugClusterIndex = uint32_t(m_tessellatorConfig.debugClusterIndex);
        params.debugLaneIndex = uint32_t(m_tessellatorConfig.debugLaneIndex);
        commandList->writeBuffer(m_fillClustersParamsBuffer, &params, sizeof(FillClustersParams));

        auto bindingSetDesc = nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_gridSamplersBuffer,
                nvrhi::Format::UNKNOWN,
                nvrhi::BufferRange(surfaceOffset * sizeof(GridSampler), surfaceCount * sizeof(GridSampler))))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_clusterOffsetCountsBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_clustersBuffer))
            .addItem(scene.GetSubdPositions(3, instance))
            // Subd
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, subd.m_vertexDeviceData.surfaceDescriptors))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(5, subd.m_vertexDeviceData.controlPointIndices))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(6, subd.m_vertexDeviceData.patchPointsOffsets))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(7, subd.GetTopologyMap()->plansBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(8, subd.GetTopologyMap()->subpatchTreesArraysBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(9, subd.GetTopologyMap()->patchPointIndicesArraysBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(10, subd.GetTopologyMap()->stencilMatrixArraysBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(11, scene.GetSubdPatchPoints(instance)))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(12, subd.m_surfaceToMaterialIndexBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(13, scene.GetMaterialBuffer()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(14, subd.m_texcoordDeviceData.surfaceDescriptors))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(15, subd.m_texcoordDeviceData.controlPointIndices))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(16, subd.m_texcoordDeviceData.patchPointsOffsets))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(17, subd.m_texcoordDeviceData.patchPoints))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(18, subd.m_texcoordsBuffer))
            .addItem(nvrhi::BindingSetItem::Sampler(0, scene.GetDisplacementSampler()))

            // gatherer
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, accels.clusterVertexPositionsBuffer)) 
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, accels.clusterShadingDataBuffer))
#if ENABLE_SHADER_DEBUG
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_debugBuffer))
#endif
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(3, accels.clusterVertexNormalsBuffer))
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_fillClustersParamsBuffer));

        nvrhi::BindingSetHandle bindingSet;
        if (!nvrhi::utils::CreateBindingSetAndLayout(m_device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_fillClustersBL, bindingSet))
        {
            log::fatal("Failed to create binding set and layout for fill_clusters.hlsl");
        }

        auto GetFillClustersPSO = [this](const FillClustersPermutation& shaderPermutation)
            {
                if (!m_fillClustersPSOs[shaderPermutation.index()])
                {
                    std::vector<donut::engine::ShaderMacro> fillClustersMacros;
                    fillClustersMacros.push_back(donut::engine::ShaderMacro("DISPLACEMENT_MAPS", shaderPermutation.isDisplacementEnabled() ? "1" : "0"));
                    fillClustersMacros.push_back(donut::engine::ShaderMacro("VERTEX_NORMALS", shaderPermutation.isVertexNormalsEnabled() ? "1" : "0"));
                    fillClustersMacros.push_back(donut::engine::ShaderMacro("SURFACE_TYPE", toString(shaderPermutation.surfaceType())));
                    nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_tess/fill_clusters.hlsl", "FillClustersMain", &fillClustersMacros, nvrhi::ShaderType::Compute);

                    auto computePipelineDesc = nvrhi::ComputePipelineDesc()
                        .setComputeShader(shader)
                        .addBindingLayout(m_fillClustersBL)
                        .addBindingLayout(m_bindlessBL);

                    m_fillClustersPSOs[shaderPermutation.index()] = m_device->createComputePipeline(computePipelineDesc);
                }
                return m_fillClustersPSOs[shaderPermutation.index()];
            };
        
        if (!m_fillClustersTexcoordsPSO)
        {
            nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_tess/fill_clusters.hlsl", "FillClustersTexcoordsMain", nullptr, nvrhi::ShaderType::Compute);

            auto computePipelineDesc = nvrhi::ComputePipelineDesc()
                .setComputeShader(shader)
                .addBindingLayout(m_fillClustersBL)
                .addBindingLayout(m_bindlessBL);

            m_fillClustersTexcoordsPSO = m_device->createComputePipeline(computePipelineDesc);
        }

        auto state = nvrhi::ComputeState()
            .addBindingSet(bindingSet)
            .addBindingSet(m_descriptorTable)
            .setIndirectParams(m_fillClustersDispatchIndirectBuffer);

        if (m_tessellatorConfig.enableMonolithicClusterBuild)
        {
            FillClustersPermutation shaderPermutation = { subd.m_hasDisplacementMaterial, m_tessellatorConfig.enableVertexNormals, ShaderPermutationSurfaceType::All };
            state.setPipeline(GetFillClustersPSO(shaderPermutation));
            commandList->setComputeState(state);
            uint32_t dispatchIndirectArgsOffset = (instanceIndex * ClusterTessDispatchType::NumTypes + ClusterTessDispatchType::Limit) * uint32_t(m_fillClustersDispatchIndirectBuffer.GetElementBytes());
            commandList->dispatchIndirect(dispatchIndirectArgsOffset);
        }
        else
        {
            for (uint32_t i = 0; i <= uint32_t(ShaderPermutationSurfaceType::Limit); i++)
            {
                FillClustersPermutation shaderPermutation = { subd.m_hasDisplacementMaterial, m_tessellatorConfig.enableVertexNormals, ShaderPermutationSurfaceType(i) };
                state.setPipeline(GetFillClustersPSO(shaderPermutation));
                commandList->setComputeState(state);
                uint32_t dispatchIndirectArgsOffset = (instanceIndex * ClusterTessDispatchType::NumTypes + ClusterTessDispatchType(i)) * uint32_t(m_fillClustersDispatchIndirectBuffer.GetElementBytes());
                commandList->dispatchIndirect(dispatchIndirectArgsOffset);
            }
        }
        
        state.setPipeline(m_fillClustersTexcoordsPSO);
        commandList->setComputeState(state);
        uint32_t dispatchIndirectArgsOffset = (instanceIndex * ClusterTessDispatchType::NumTypes + ClusterTessDispatchType::All) * uint32_t(m_fillClustersDispatchIndirectBuffer.GetElementBytes());
        commandList->dispatchIndirect(dispatchIndirectArgsOffset);

        surfaceOffset += surfaceCount;

#if ENABLE_SHADER_DEBUG
        if (m_tessellatorConfig.debugSurfaceIndex >= 0 &&
            m_tessellatorConfig.debugClusterIndex >= 0 &&
            m_tessellatorConfig.debugLaneIndex >= 0)
        {
            donut::log::info("Fill Clusters Debug Instance:%d MeshID:%u (Surface:%u Cluster:%u Lane:%u)", instanceIndex, instance.node && instance.node->subdMeshInstance ? instance.node->subdMeshInstance->meshID : ~0u, m_tessellatorConfig.debugSurfaceIndex,
                m_tessellatorConfig.debugClusterIndex, m_tessellatorConfig.debugLaneIndex);

            auto debugOutput = m_debugBuffer.Download(commandList);
            uint numElements = debugOutput.front().payloadType;
            vectorlog::Log(debugOutput, ShaderDebugElement::OutputLambda, vectorlog::FormatOptions{ .wrap = false, .header = false, .elementIndex = false, .startIndex = 1, .count = numElements });
        }
#endif
    }

    stats::clusterAccelSamplers.fillClustersTime.Stop();
}

void ClusterTessellator::ComputeInstanceClusterTiling(ClusterTessAccels& accels, 
    const RTXMGScene& scene,
    uint32_t instanceIndex,
    uint32_t surfaceOffset,
    uint32_t surfaceCount,
    const nvrhi::BufferRange& tessCounterRange,
    nvrhi::ICommandList* commandList)
{

    const auto& subdMeshes = scene.GetSubdMeshes();
    const auto& instance = scene.GetSubdMeshInstances()[instanceIndex];

    const SubdivisionSurface& subdivisionSurface = *subdMeshes[instance.meshID];

    const donut::math::affine3& localToWorld = instance.localToWorld;

#if ENABLE_SHADER_DEBUG
    if (m_tessellatorConfig.debugSurfaceIndex >= 0 &&
        m_tessellatorConfig.debugLaneIndex >= 0)
    {
        commandList->clearBufferUInt(m_debugBuffer, 0);
    }
#endif

    ComputeClusterTilingParams params = {};
    params.debugSurfaceIndex = uint32_t(m_tessellatorConfig.debugSurfaceIndex);
    params.debugLaneIndex = uint32_t(m_tessellatorConfig.debugLaneIndex);
    params.matWorldToClip = m_tessellatorConfig.camera->GetProjectionMatrix() * m_tessellatorConfig.camera->GetViewMatrix();
    affineToColumnMajor(localToWorld, params.localToWorld.m_data); // params.localToWorld;
    params.viewportSize.x = float(m_tessellatorConfig.viewportSize.x);
    params.viewportSize.y = float(m_tessellatorConfig.viewportSize.y);
    params.isolationLevel = m_tessellatorConfig.isolationLevel;
    params.coarseTessellationRate = m_tessellatorConfig.coarseTessellationRate;
    params.fineTessellationRate = m_tessellatorConfig.fineTessellationRate;
    params.cameraPos = m_tessellatorConfig.camera->GetEye();
    params.aabb = scene.GetSubdAabb(instance) * localToWorld;
    params.enableBackfaceVisibility = m_tessellatorConfig.enableBackfaceVisibility;
    params.enableFrustumVisibility = m_tessellatorConfig.enableFrustumVisibility;
    params.enableHiZVisibility = m_tessellatorConfig.enableHiZVisibility && m_tessellatorConfig.zbuffer != nullptr;
    params.edgeSegments = m_tessellatorConfig.edgeSegments;
    params.globalDisplacementScale = m_tessellatorConfig.displacementScale;

    params.maxClasBlocks = uint32_t(m_maxClasBytes / size_t(cluster::kClasByteAlignment));
    params.maxClusters = m_maxClusters;
    params.maxVertices = m_maxVertices;
    params.clusterVertexPositionsBaseAddress = accels.clusterVertexPositionsBuffer.GetGpuVirtualAddress();
    params.clasDataBaseAddress = accels.clasBuffer.GetGpuVirtualAddress();

    if (m_tessellatorConfig.zbuffer)
    {
        params.numHiZLODs = m_tessellatorConfig.zbuffer->GetNumHiZLODs();
        params.invHiZSize = m_tessellatorConfig.zbuffer->GetInvHiZSize();
    }
    
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_computeClusterTilingParamsBuffer))
        .addItem(scene.GetSubdPositions(0, instance))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, subdivisionSurface.m_surfaceToMaterialIndexBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(2, scene.GetMaterialBuffer()))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, subdivisionSurface.m_vertexDeviceData.surfaceDescriptors))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, subdivisionSurface.m_vertexDeviceData.controlPointIndices))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(5, subdivisionSurface.m_vertexDeviceData.patchPointsOffsets))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(6, subdivisionSurface.GetTopologyMap()->plansBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(7, subdivisionSurface.GetTopologyMap()->subpatchTreesArraysBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(8, subdivisionSurface.GetTopologyMap()->patchPointIndicesArraysBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(9, subdivisionSurface.GetTopologyMap()->stencilMatrixArraysBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(10, m_templateBuffers.instantiationSizesBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(11, m_templateBuffers.addressesBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(12, subdivisionSurface.m_texcoordDeviceData.surfaceDescriptors))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(13, subdivisionSurface.m_texcoordDeviceData.controlPointIndices))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(14, subdivisionSurface.m_texcoordDeviceData.patchPointsOffsets))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(15, subdivisionSurface.m_texcoordsBuffer))
        .addItem(nvrhi::BindingSetItem::Sampler(0, scene.GetDisplacementSampler()))
        .addItem(nvrhi::BindingSetItem::Sampler(1, m_commonPasses->m_LinearClampSampler)) // hiZ sampler
        
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_gridSamplersBuffer,
            nvrhi::Format::UNKNOWN,
            nvrhi::BufferRange(surfaceOffset * sizeof(GridSampler), surfaceCount * sizeof(GridSampler))))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_tessellationCountersBuffer, nvrhi::Format::UNKNOWN, tessCounterRange))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_clustersBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(3, accels.clusterShadingDataBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(4, m_clasIndirectArgDataBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(5, accels.clasPtrsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(6, scene.GetSubdPatchPoints(instance)))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(7, subdivisionSurface.m_texcoordDeviceData.patchPoints));
#if ENABLE_SHADER_DEBUG
    bindingSetDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(8, m_debugBuffer));
#endif

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(m_device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_computeClusterTilingBL, bindingSet, true))
    {
        log::fatal("Failed to create binding set and layout for compute_cluster_tiling.hlsl");
    }

    nvrhi::BindingLayoutDesc hizLayoutDesc;
    nvrhi::BindingSetDesc hizSetDesc;
    m_tessellatorConfig.zbuffer->GetHiZDesc(&hizLayoutDesc, &hizSetDesc);
    
    if (!m_computeClusterTilingHizBL)
    {
        hizLayoutDesc
            .setVisibility(nvrhi::ShaderType::Compute)
            .setRegisterSpace(1)
            .setRegisterSpaceIsDescriptorSet(true);
        m_computeClusterTilingHizBL = m_device->createBindingLayout(hizLayoutDesc);
        if (!m_computeClusterTilingHizBL)
        {
            log::fatal("Failed to create hiz binding layout for compute_cluster_tiling.hlsl");
        }
    }

    nvrhi::BindingSetHandle hizSet = m_device->createBindingSet(hizSetDesc, m_computeClusterTilingHizBL);
    if (!hizSet)
    {
        log::fatal("Failed to create hiz binding set for compute_cluster_tiling.hlsl");
    }

    ComputeClusterTilingPermutation shaderPermutation(subdivisionSurface.m_hasDisplacementMaterial,
        m_tessellatorConfig.enableFrustumVisibility,
        m_tessellatorConfig.tessMode,
        m_tessellatorConfig.visMode,
        ShaderPermutationSurfaceType::PureBSpline);

    auto GetComputeClusterTilingPSO = [this](const ComputeClusterTilingPermutation& shaderPermutation)
        {
            if (!m_computeClusterTilingPSOs[shaderPermutation.index()])
            {
                std::vector<donut::engine::ShaderMacro> macros;
                macros.push_back(donut::engine::ShaderMacro("DISPLACEMENT_MAPS", shaderPermutation.isDisplacementEnabled() ? "1" : "0"));
                macros.push_back(donut::engine::ShaderMacro("TESS_MODE", toString(shaderPermutation.tessellationMode())));
                macros.push_back(donut::engine::ShaderMacro("ENABLE_FRUSTUM_VISIBILITY", shaderPermutation.isFrustumVisibilityEnabled() ? "1" : "0"));
                macros.push_back(donut::engine::ShaderMacro("VIS_MODE", toString(shaderPermutation.visibilityMode())));
                macros.push_back(donut::engine::ShaderMacro("SURFACE_TYPE", toString(shaderPermutation.surfaceType())));

                nvrhi::ShaderDesc tilingDesc(nvrhi::ShaderType::Compute);
                nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader(
                    GetClusterApiShaderPath(m_device, "cluster_tess/compute_cluster_tiling.hlsl").c_str(), "main", &macros, tilingDesc);

                auto computePipelineDesc = nvrhi::ComputePipelineDesc()
                    .setComputeShader(shader)
                    .addBindingLayout(m_computeClusterTilingBL)
                    .addBindingLayout(m_computeClusterTilingHizBL)
                    .addBindingLayout(m_bindlessBL);

                m_computeClusterTilingPSOs[shaderPermutation.index()] = m_device->createComputePipeline(computePipelineDesc);
            }
            return m_computeClusterTilingPSOs[shaderPermutation.index()];
        };

    auto state = nvrhi::ComputeState()
        .addBindingSet(bindingSet)
        .addBindingSet(hizSet)
        .addBindingSet(m_descriptorTable);

    if (m_tessellatorConfig.enableMonolithicClusterBuild)
    {
        // Skip no limit surfaces
        params.surfaceStart = 0;
        params.surfaceEnd = subdivisionSurface.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::NoLimit)];
        uint32_t dispatchCount = params.surfaceEnd - params.surfaceStart;

        commandList->writeBuffer(m_computeClusterTilingParamsBuffer, &params, sizeof(ComputeClusterTilingParams));
        ShaderPermutationSurfaceType shaderSurfaceType = ShaderPermutationSurfaceType::All;
        shaderPermutation.setSurfaceType(shaderSurfaceType);
        state.setPipeline(GetComputeClusterTilingPSO(shaderPermutation));
        commandList->setComputeState(state);

        commandList->dispatch(div_ceil(dispatchCount, kComputeClusterTilingWaves), 1, 1);

        // Save cluster offset for this instance
        ClusterTessDispatchType dispatchType = ClusterTessDispatchType::All;
        CopyClusterOffset(instanceIndex, dispatchType, tessCounterRange, commandList);
    }
    else
    {
        // Loop
        for (uint32_t i = 0; i <= uint32_t(ClusterTessDispatchType::Limit); i++)
        {
            SubdivisionSurface::SurfaceType subdSurfaceType = SubdivisionSurface::SurfaceType(i);

            // Skip no limit surfaces
            params.surfaceStart = subdivisionSurface.m_surfaceOffsets[uint32_t(subdSurfaceType)];
            params.surfaceEnd = subdivisionSurface.m_surfaceOffsets[uint32_t(subdSurfaceType) + 1];

            uint32_t dispatchCount = params.surfaceEnd - params.surfaceStart;
            if (dispatchCount)
            {
                commandList->writeBuffer(m_computeClusterTilingParamsBuffer, &params, sizeof(ComputeClusterTilingParams));

                ShaderPermutationSurfaceType shaderSurfaceType = ShaderPermutationSurfaceType(i);
                shaderPermutation.setSurfaceType(shaderSurfaceType);
                state.setPipeline(GetComputeClusterTilingPSO(shaderPermutation));
                commandList->setComputeState(state);

                commandList->dispatch(div_ceil(dispatchCount, kComputeClusterTilingWaves), 1, 1);
            }
            // Save cluster offset for this instance
            ClusterTessDispatchType dispatchType = ClusterTessDispatchType(i);
            CopyClusterOffset(instanceIndex, dispatchType, tessCounterRange, commandList);
        }
    }

#if ENABLE_SHADER_DEBUG
    if (m_tessellatorConfig.debugSurfaceIndex >= 0 &&
        m_tessellatorConfig.debugLaneIndex >= 0)
    {
        donut::log::info("Cluster Tiling Debug Instance:%d MeshID:%u (Surface:%d, Lane:%d)", instanceIndex, instance.node && instance.node->subdMeshInstance ? instance.node->subdMeshInstance->meshID : ~0u, m_tessellatorConfig.debugSurfaceIndex, m_tessellatorConfig.debugLaneIndex);

        auto debugOutput = m_debugBuffer.Download(commandList);
        uint numElements = debugOutput.front().payloadType;
        vectorlog::Log(debugOutput, ShaderDebugElement::OutputLambda, vectorlog::FormatOptions{ .wrap = false, .header = false, .elementIndex = false, .startIndex = 1, .count = numElements });
    }
#endif

    if (m_tessellatorConfig.enableLogging)
    {
        donut::log::info("Vertex PatchPoints:%d MeshID:%u", instanceIndex, instance.node && instance.node->subdMeshInstance ? instance.node->subdMeshInstance->meshID : ~0u);
        {
            auto readBackDesc = GetReadbackDesc(subdivisionSurface.m_vertexDeviceData.patchPoints->getDesc());
            auto readbackBuffer = commandList->getDevice()->createBuffer(readBackDesc);

            std::vector<float3> patchPoints;
            DownloadBuffer<float3>(subdivisionSurface.m_vertexDeviceData.patchPoints, patchPoints, readbackBuffer, false, commandList);
            vectorlog::Log(patchPoints);
        }

        donut::log::info("Texcoord PatchPoints:%d MeshID:%u", instanceIndex, instance.node && instance.node->subdMeshInstance ? instance.node->subdMeshInstance->meshID : ~0u);
        {
            auto readBackDesc = GetReadbackDesc(subdivisionSurface.m_texcoordDeviceData.patchPoints->getDesc());
            auto readbackBuffer = commandList->getDevice()->createBuffer(readBackDesc);

            std::vector<float2> patchPoints;
            DownloadBuffer<float2>(subdivisionSurface.m_texcoordDeviceData.patchPoints, patchPoints, readbackBuffer, false, commandList);
            vectorlog::Log(patchPoints);
        }
    }
}

void ClusterTessellator::CopyClusterOffset(uint32_t instanceIndex,
    ClusterTessDispatchType dispatchType, const nvrhi::BufferRange& tessCounterRange, nvrhi::ICommandList* commandList)
{
    nvrhi::utils::ScopedMarker marker(commandList, "ClusterTessellator::CopyClusterOffset");
    CopyClusterOffsetParams params;
    params.instanceIndex = instanceIndex;
    params.dispatchTypeIndex = uint32_t(dispatchType);
    commandList->writeBuffer(m_copyClusterOffsetParamsBuffer, &params, sizeof(CopyClusterOffsetParams));

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_tessellationCountersBuffer, nvrhi::Format::UNKNOWN, tessCounterRange))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_clusterOffsetCountsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_fillClustersDispatchIndirectBuffer))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_copyClusterOffsetParamsBuffer));

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(m_device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_copyClusterOffsetBL, bindingSet))
    {
        log::fatal("Failed to create binding set and layout for copy_cluster_offset shader");
    }

    if (!m_copyClusterOffsetPSO)
    {
        nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_tess/copy_cluster_offset.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(shader)
            .addBindingLayout(m_copyClusterOffsetBL);

        m_copyClusterOffsetPSO = m_device->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_copyClusterOffsetPSO)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    commandList->dispatch(1, 1, 1);
}

void ClusterTessellator::UpdateTessRate(uint32_t tessCounterIndex, nvrhi::ICommandList* commandList)
{
    nvrhi::utils::ScopedMarker marker(commandList, "ClusterTessellator::UpdateTessRate");

    // Uniform mode ignores the rates, so there is nothing to adapt.
    const UpdateTessRateParams params = {
        .slot = tessCounterIndex,
        .prevSlot = (tessCounterIndex + kFrameCount - 1) % kFrameCount,
        .prev2Slot = (tessCounterIndex + kFrameCount - 2) % kFrameCount,
        .numPrevBuilds = std::min(m_buildAccelFrameIndex, 2u),
        .maxClusters = m_maxClusters,
        .maxVertices = m_maxVertices,
        .maxClasBlocks = uint32_t(m_maxClasBytes / size_t(cluster::kClasByteAlignment)),
        .enable = m_tessellatorConfig.enableAdaptiveTessRate &&
            m_tessellatorConfig.tessMode != TessellatorConfig::AdaptiveTessellationMode::UNIFORM,
    };

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_tessellationCountersBuffer))
        .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(UpdateTessRateParams)));

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(m_device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_updateTessRateBL, bindingSet))
    {
        log::fatal("Failed to create binding set and layout for update_tess_rate shader");
    }

    if (!m_updateTessRatePSO)
    {
        nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_tess/update_tess_rate.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(shader)
            .addBindingLayout(m_updateTessRateBL);

        m_updateTessRatePSO = m_device->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_updateTessRatePSO)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    commandList->setPushConstants(&params, sizeof(params));
    commandList->dispatch(1, 1, 1);
}

void ClusterTessellator::BuildBlasFromClas(ClusterTessAccels& accels, std::span<Instance const> instances, nvrhi::ICommandList* commandList)
{
    //// Allocate and build BLASes
    nvrhi::utils::ScopedMarker marker(commandList, "Blas Build from Clas");
    stats::clusterAccelSamplers.buildBlasTime.Start(commandList);

    uint32_t numInstances = static_cast<uint32_t>(instances.size());
    nvrhi::GpuVirtualAddress clasPtrsBaseAddress = accels.clasPtrsBuffer.GetGpuVirtualAddress();
    FillBlasFromClasArgs(m_blasFromClasIndirectArgsBuffer, m_clusterOffsetCountsBuffer, clasPtrsBaseAddress, numInstances, commandList);

    if (m_tessellatorConfig.enableLogging)
    {
        m_blasFromClasIndirectArgsBuffer.Log(commandList, [](std::ostream& ss, const cluster::IndirectArgs& e)
            {
                ss << "{c: " << std::dec << e.clusterCount <<
                    " | addr: " << std::hex << e.clusterAddresses << "}";
                return true;
            });
    }

    //// Build Operation
    cluster::OperationDesc createBlasDesc =
    {
        .params = m_createBlasParams,
        .scratchSizeInBytes = m_createBlasSizeInfo.scratchSizeInBytes,
        .inIndirectArgCountBuffer = nullptr,
        .inIndirectArgCountOffsetInBytes = 0,
        .inIndirectArgsBuffer = m_blasFromClasIndirectArgsBuffer,
        .inIndirectArgsOffsetInBytes = 0,
        .inOutAddressesBuffer = accels.blasPtrsBuffer,
        .inOutAddressesOffsetInBytes = 0,
        .outSizesBuffer = accels.blasSizesBuffer,
        .outSizesOffsetInBytes = 0,
        .outAccelerationStructuresBuffer = accels.blasBuffer,
        .outAccelerationStructuresOffsetInBytes = 0,
    };
    commandList->executeMultiIndirectClusterOperation(createBlasDesc);

    stats::clusterAccelSamplers.buildBlasTime.Stop();
}
bool ClusterTessellator::ResizeAccelStorage(const RTXMGScene& scene, const TessellatorConfig& config,
    ClusterTessAccels& accels)
{
    m_tessellatorConfig = config;

    const auto& subdMeshes = scene.GetSubdMeshes();
    const auto& instances = scene.GetSubdMeshInstances();

    if (subdMeshes.empty() || instances.empty())
        return false;

    return UpdateMemoryAllocations(accels, uint32_t(instances.size()), scene.TotalSubdPatchCount());
}

bool ClusterTessellator::UpdateMemoryAllocations(ClusterTessAccels& accels, uint32_t numInstances, uint32_t sceneSubdPatches)
{
    uint32_t maxClusters = std::min(kMaxApiClusterCount, m_tessellatorConfig.memorySettings.maxClusters);
    maxClusters = std::max(1u, maxClusters);

    // Reallocate memory if settings changed
    size_t maxClusterBlocks = (m_tessellatorConfig.memorySettings.clasBufferBytes + (size_t(cluster::kClasByteAlignment) - 1ull)) / size_t(cluster::kClasByteAlignment);
    maxClusterBlocks = std::max<size_t>(1, maxClusterBlocks);
    size_t maxClasBytes = size_t(cluster::kClasByteAlignment) * maxClusterBlocks;

    // Calculate max vertices based on vertex buffer bytes (same for positions and normals since both are float3)
    uint32_t maxVertices = uint32_t(m_tessellatorConfig.memorySettings.vertexBufferBytes / sizeof(float3));
    maxVertices = std::max(kClusterMaxVertices, maxVertices);

    bool numInstancesChanged = m_numInstances != numInstances;
    bool sceneSubdPatchesChanged = m_sceneSubdPatches != sceneSubdPatches;
    bool numClustersChanged = m_maxClusters != maxClusters;
    bool clasBytesChanged = m_maxClasBytes != maxClasBytes;
    bool maxVerticesChanged = m_maxVertices != maxVertices;

    // Check if vertex normals setting changed by comparing current setting to buffer state
    bool prevVertexNormalsEnabled = accels.clusterVertexNormalsBuffer.GetBuffer() != nullptr && accels.clusterVertexNormalsBuffer.GetNumElements() == m_maxVertices;
    bool enableVertexNormalsChanged = (m_tessellatorConfig.enableVertexNormals != prevVertexNormalsEnabled);

    m_numInstances = numInstances;
    m_sceneSubdPatches = sceneSubdPatches;
    m_maxClusters = maxClusters;
    m_maxClasBytes = maxClasBytes;
    m_maxVertices = maxVertices;
    
    // No allocations needed
    if (!numInstancesChanged && !sceneSubdPatchesChanged && !numClustersChanged && !clasBytesChanged && !maxVerticesChanged && !enableVertexNormalsChanged)
    {
        return false;
    }

    // Wait for idle to ensure resources are not in use
    m_device->waitForIdle();

    if (numInstancesChanged)
    {
        if (m_copyClusterOffsetParamsBuffer)
            m_copyClusterOffsetParamsBuffer->Release();
        m_clusterOffsetCountsBuffer.Release();
        m_fillClustersDispatchIndirectBuffer.Release();
        m_blasFromClasIndirectArgsBuffer.Release();
        accels.blasPtrsBuffer.Release();
        accels.blasSizesBuffer.Release();

        m_copyClusterOffsetParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
            sizeof(CopyClusterOffsetParams), "CopyClusterOffsetParams", m_numInstances * ClusterTessDispatchType::NumTypes * kFrameCount));
        // Written per instance (tiling: per surface type) every frame; Vulkan needs a version for each in flight.
        m_computeClusterTilingParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
            sizeof(ComputeClusterTilingParams), "ComputeClusterTilingParams", m_numInstances * ClusterTessDispatchType::NumTypes * kFrameCount));
        m_fillClustersParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
            sizeof(FillClustersParams), "FillClustersParams", m_numInstances * kFrameCount));

        m_clusterOffsetCountsBuffer.Create(m_numInstances * ClusterTessDispatchType::NumTypes, "ClusterOffsets", m_device);
        nvrhi::BufferDesc dispatchIndirectDesc =
        {
            .byteSize = m_numInstances * ClusterTessDispatchType::NumTypes * m_fillClustersDispatchIndirectBuffer.GetElementBytes(),
            .structStride = uint32_t(m_fillClustersDispatchIndirectBuffer.GetElementBytes()),
            .debugName = "FillClustersIndirectArgs",
            .canHaveUAVs = true,
            .isDrawIndirectArgs = true,
            .initialState = nvrhi::ResourceStates::IndirectArgument,
            .keepInitialState = true,
        };
        m_fillClustersDispatchIndirectBuffer.Create(dispatchIndirectDesc, m_device);

        // Create and fill out the instantiate args buffer from addressesBuffer
        nvrhi::BufferDesc clusterIndirectArgsDesc = {
            .byteSize = sizeof(cluster::IndirectArgs) * m_numInstances,
            .structStride = sizeof(cluster::IndirectArgs),
            .debugName = "cluster::IndirectArgs",
            .canHaveUAVs = true,
            .isAccelStructBuildInput = true,
            .initialState = nvrhi::ResourceStates::ShaderResource,
            .keepInitialState = true,
        };
        m_blasFromClasIndirectArgsBuffer.Create(clusterIndirectArgsDesc, m_device);
        accels.blasPtrsBuffer.Create(m_numInstances, "BlasPtrs", m_device);
        accels.blasSizesBuffer.Create(m_numInstances, "BlasSizes", m_device);
    }

    if (sceneSubdPatchesChanged)
    {
        m_gridSamplersBuffer.Release();
        m_gridSamplersBuffer.Create(m_sceneSubdPatches, "GridSamplers", m_device);
    }

    if (numClustersChanged)
    {
        m_clustersBuffer.Release();
        m_clasIndirectArgDataBuffer.Release();
        accels.clusterShadingDataBuffer.Release();
        accels.clasPtrsBuffer.Release();

        m_clustersBuffer.Create(m_maxClusters, "clusters", m_device);

        // Vulkan requires BUILD_INPUT_READ_ONLY usage on the instantiate op's
        // srcInfosArray, same rule as the buffer above.
        nvrhi::BufferDesc clasIndirectArgDataDesc =
            GetGenericDesc(m_maxClusters, sizeof(cluster::nv::IndirectInstantiateTemplateArgs),
                           "indirect arg data");
        clasIndirectArgDataDesc.isAccelStructBuildInput = true;
        m_clasIndirectArgDataBuffer.Create(clasIndirectArgDataDesc, m_device);

        accels.clusterShadingDataBuffer.Create(m_maxClusters, "cluster shading data", m_device);
        accels.clasPtrsBuffer.Create(m_maxClusters, "ClasAddresses", m_device);
    }

    if (numClustersChanged || numInstancesChanged)
    {
        accels.blasBuffer.Release();
        m_createBlasParams =
        {
            .maxArgCount = m_numInstances,
            .type = cluster::OperationType::BlasBuild,
            .mode = cluster::OperationMode::ImplicitDestinations,
            .flags = cluster::OperationFlags::None,
            .blas =
            {
                .maxClasPerBlasCount = m_maxClusters,
                .maxTotalClasCount = m_maxClusters
            }
        };
        m_createBlasSizeInfo = m_device->getClusterOperationSizeInfo(m_createBlasParams);

        nvrhi::BufferDesc blasBufferDesc = {
            .byteSize = m_createBlasSizeInfo.resultMaxSizeInBytes,
            .debugName = "Blas Data",
            .canHaveUAVs = true,
            .isAccelStructStorage = true,
            .initialState = nvrhi::ResourceStates::AccelStructWrite,
            .keepInitialState = true,
        };
        accels.blasBuffer.Create(blasBufferDesc, m_device);
    }

    if (clasBytesChanged)
    {
        accels.clasBuffer.Release();

        nvrhi::BufferDesc clasDataDesc =
        {
            .byteSize = m_maxClasBytes,
            .debugName = "ClasData",
            .canHaveUAVs = true,
            .isAccelStructStorage = true,
            .initialState = nvrhi::ResourceStates::AccelStructWrite,
            .keepInitialState = true,
        };
        accels.clasBuffer.Create(clasDataDesc, m_device);
    }

    if (maxVerticesChanged)
    {
        accels.clusterVertexPositionsBuffer.Release();
        accels.clusterVertexPositionsBuffer.Create(m_maxVertices, "cluster vertex positions", m_device);
    }
        
    if (maxVerticesChanged || enableVertexNormalsChanged)
    {
        accels.clusterVertexNormalsBuffer.Release();
        accels.clusterVertexNormalsBuffer.Create(m_tessellatorConfig.enableVertexNormals ? m_maxVertices : 1, "cluster vertex normals", m_device);
    }

    return true;
}

uint64_t ClusterTessellator::GetWorkingBufferBytes(const ClusterTessAccels& accels) const
{
    return m_tessellationCountersBuffer.GetVramBytes()
         + m_fillClustersDispatchIndirectBuffer.GetVramBytes()
         + m_clusterOffsetCountsBuffer.GetVramBytes()
         + m_gridSamplersBuffer.GetVramBytes()
         + m_blasFromClasIndirectArgsBuffer.GetVramBytes()
         + rtxmg::BufferVramBytes(m_templateBuffers.dataBuffer)
         + m_templateBuffers.addressesBuffer.GetVramBytes()
         + m_templateBuffers.instantiationSizesBuffer.GetVramBytes()
         + accels.blasPtrsBuffer.GetVramBytes()
         + accels.blasSizesBuffer.GetVramBytes();
}

void ClusterTessellator::BuildAccel(const RTXMGScene& scene, const TessellatorConfig& config,
    ClusterTessAccels& accels, ClusterTessStatistics& stats, uint32_t frameIndex, nvrhi::ICommandList* commandList)
{
    m_tessellatorConfig = config;

    const auto& subdMeshes = scene.GetSubdMeshes();
    const auto& instances = scene.GetSubdMeshInstances();

    if (subdMeshes.empty() || instances.empty())
        return;

    // Storage was sized by ResizeAccelStorage at the top of the frame.
    InitStructuredClusterTemplates(commandList);
    
    nvrhi::utils::ScopedMarker marker(commandList, "ClusterTessellator::BuildAccel");

    uint32_t tessCounterIndex = (m_buildAccelFrameIndex % kFrameCount);
    nvrhi::BufferRange tessCounterRange = { m_tessellationCountersBuffer.GetElementBytes() * tessCounterIndex, m_tessellationCountersBuffer.GetElementBytes() };

    // Clear tessellation counters for this frame
    TessellationCounters tessCounters = {};
    m_tessellationCountersBuffer.UploadElement(tessCounters, tessCounterIndex, commandList);
    UpdateTessRate(tessCounterIndex, commandList);

    commandList->clearBufferUInt(m_clusterOffsetCountsBuffer, 0);
    commandList->clearBufferUInt(m_fillClustersDispatchIndirectBuffer, 0);

    {
        nvrhi::utils::ScopedMarker marker(commandList, "ComputeClusterTiling");
        stats::clusterAccelSamplers.clusterTilingTime.Start(commandList);
        uint32_t surfaceOffset = 0;
        for (uint32_t i = 0; i < instances.size(); ++i)
        {
            const auto& inst = instances[i];
            const auto& subd = *subdMeshes[inst.meshID];

            uint32_t surfaceCount{ subd.SurfaceCount() };

            ComputeInstanceClusterTiling(accels, scene, i, surfaceOffset, surfaceCount, tessCounterRange, commandList);

            surfaceOffset += surfaceCount;
        }
        stats::clusterAccelSamplers.clusterTilingTime.Stop();
    }

    if (m_tessellatorConfig.enableLogging)
    {
        // sync download to get current frame results
        m_tessellationCountersBuffer.Log(commandList, [](std::ostream& ss, const TessellationCounters& e)
            {
                ss << "{cluster: " << e.desiredClusters << "/" << e.clusters 
                    << ", vertices: " << e.desiredVertices
                    << ", tri: " << e.desiredTriangles
                    << ", clasBytes: " << e.DesiredClasBytes() << "}";
                return true;
            });
        TessellationCounters currentTessCounts = m_tessellationCountersBuffer.Download(commandList, false)[tessCounterIndex];

        log::info("tessellation counters: ");
        log::info("  clusters: %u", currentTessCounts.clusters);
        log::info("  desiredClusters: %u", currentTessCounts.desiredClusters);
        log::info("  desiredVertices: %u", currentTessCounts.desiredVertices);
        log::info("  desiredTriangles: %u", currentTessCounts.desiredTriangles);
        log::info("  desiredClasBytes: %llu", currentTessCounts.DesiredClasBytes());

        m_fillClustersDispatchIndirectBuffer.Log(commandList);
        m_clusterOffsetCountsBuffer.Log(commandList);
        
        m_gridSamplersBuffer.Log(commandList, [](std::ostream& ss, const GridSampler& e)
            {
                ss << "{" << e.edgeSegments.x << ", " << e.edgeSegments.y << ", " << e.edgeSegments.z << ", " << e.edgeSegments.w;
                return true;
            });
        
        m_clustersBuffer.Log(commandList, [](std::ostream& ss, const Cluster& e)
        {
              ss << "{surface:" << e.iSurface << ", vertexOffset:" << e.nVertexOffset
                  << ", offset:" << e.offset.x << ", " << e.offset.y << ", size: " << e.sizeX << ", " << e.sizeY << "}";
              return true;
        }, { .count = std::min(currentTessCounts.clusters, 64u) });

        accels.clusterShadingDataBuffer.Log(commandList, [](std::ostream& ss, auto& e)
        {
            ss << "{surface id: " << e.m_surfaceId
                << ", edges:" << e.m_edgeSegments.x << ", " << e.m_edgeSegments.y << ", " << e.m_edgeSegments.z << ", " << e.m_edgeSegments.w
                << ", texcoords: (" << e.m_texcoords[0].x << "," << e.m_texcoords[0].y << "), (" << e.m_texcoords[1].x << "," << e.m_texcoords[1].y << ")" << ", (" << e.m_texcoords[2].x << "," << e.m_texcoords[2].y << ")" << ", (" << e.m_texcoords[3].x << "," << e.m_texcoords[3].y << ")"
                << ", voffset:" << e.m_vertexOffset
                << ", clusterOffset:" << e.m_clusterOffset.x << "," << e.m_clusterOffset.y
                << ", cluster_size:" << e.m_clusterSizeX << "," << e.m_clusterSizeY
                << "}";
            return true;
        }, { .count = std::min(currentTessCounts.clusters, 64u) });

        m_clasIndirectArgDataBuffer.Log(commandList, [](std::ostream& ss, auto& e)
        {
            ss << "{clusterId:" << e.clusterIdOffset << ", geometryIndexOffset:" << e.geometryIndexOffset
                << ", clusterTemplate:0x" << std::hex << e.clusterTemplate
                << ", vertexBuffer:0x" << std::hex << e.vertexBuffer.startAddress
                << ", vertexBufferStride:" << std::dec << e.vertexBuffer.strideInBytes
                << "}";
            return true;
        }, { .count = std::min(currentTessCounts.clusters, 64u) });

        accels.clasPtrsBuffer.Log(commandList, { .count = std::min(currentTessCounts.clusters, 64u) });
    }

    FillInstanceClusters(scene, accels, commandList);

    // Build CLASes for all instances at once
    stats::clusterAccelSamplers.buildClasTime.Start(commandList);
    BuildStructuredCLASes(accels, tessCounterRange, commandList);
    stats::clusterAccelSamplers.buildClasTime.Stop();

    BuildBlasFromClas(accels, instances, commandList);
    
    // Async read of counters
    auto counterBufferData = m_tessellationCountersBuffer.Download(commandList, true);
    TessellationCounters counters = counterBufferData[(tessCounterIndex + 1) % kFrameCount];
    stats.tessRateScale = counters.rateScale > 0.f ? counters.rateScale : 1.f;
    stats.tessRateAdjusted = counters.rateAdjusted != 0;
    stats.clusterRateScale = counters.rateScale > 0.f ? counters.clusterRateScale : 1.f;
    stats.clasRateScale = counters.rateScale > 0.f ? counters.clasRateScale : 1.f;
    stats.vertexRateScale = counters.rateScale > 0.f ? counters.vertexRateScale : 1.f;

    // Record the desired required memory instead of the max
    stats.desired.m_numTriangles = counters.desiredTriangles;
    stats.desired.m_numClusters = counters.desiredClusters;
    stats.desired.m_vertexBufferSize = accels.clusterVertexPositionsBuffer.GetElementBytes() * counters.desiredVertices;
    stats.desired.m_vertexNormalsBufferSize = m_tessellatorConfig.enableVertexNormals ? 
        (accels.clusterVertexNormalsBuffer.GetElementBytes() * counters.desiredVertices) : 0;
    stats.desired.m_clasSize = counters.DesiredClasBytes();
    stats.desired.m_clusterDataSize = (m_clustersBuffer.GetElementBytes() +
        accels.clusterShadingDataBuffer.GetElementBytes() +
        accels.clasPtrsBuffer.GetElementBytes() +
        m_clasIndirectArgDataBuffer.GetElementBytes()) * counters.desiredClusters;
    stats.desired.m_blasSize = m_createBlasSizeInfo.resultMaxSizeInBytes;
    stats.desired.m_blasScratchSize = m_createBlasSizeInfo.scratchSizeInBytes;

    // Atomics are expensive so we don't track the number of allocated triangles
    stats.allocated.m_numTriangles = counters.desiredTriangles;
    stats.allocated.m_numClusters = m_maxClusters;
    stats.allocated.m_vertexBufferSize = accels.clusterVertexPositionsBuffer.GetBytes();
    stats.allocated.m_vertexNormalsBufferSize = accels.clusterVertexNormalsBuffer.GetBytes();
    stats.allocated.m_clasSize = accels.clasBuffer.GetBytes();
    stats.allocated.m_clusterDataSize = m_clustersBuffer.GetBytes() + accels.clusterShadingDataBuffer.GetBytes()
        + accels.clasPtrsBuffer.GetBytes() + m_clasIndirectArgDataBuffer.GetBytes();
    stats.allocated.m_blasSize = accels.blasBuffer.GetBytes();
    stats.allocated.m_blasScratchSize = m_createBlasSizeInfo.scratchSizeInBytes;

    m_buildAccelFrameIndex++;
}
