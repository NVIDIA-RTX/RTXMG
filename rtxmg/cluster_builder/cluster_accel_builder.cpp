//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <donut/core/log.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/ShaderFactory.h>
#include <nvrhi/common/misc.h>
#include <nvrhi/utils.h>

#include <map>
#include <fstream>

#include "rtxmg/utils/buffer.h"

#include "rtxmg/cluster_builder/cluster_accels.h"
#include "rtxmg/cluster_builder/cluster_accel_builder.h"
#include "rtxmg/cluster_builder/fill_clusters_params.h"
#include "rtxmg/cluster_builder/copy_cluster_offset_params.h"
#include "rtxmg/cluster_builder/fill_blas_from_clas_args_params.h"
#include "rtxmg/cluster_builder/fill_instantiate_template_args_params.h"
#include "rtxmg/cluster_builder/compute_cluster_tiling_params.h"
#include "rtxmg/cluster_builder/tessellation_counters.h"
#include "rtxmg/cluster_builder/tessellator_config.h"
#include "rtxmg/cluster_builder/tessellator_constants.h"

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
constexpr uint32_t kTessBufferFrameCount = 4;

ClusterAccelBuilder::ClusterAccelBuilder(donut::engine::ShaderFactory& shaderFactory,
    std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
    nvrhi::DescriptorTableHandle descriptorTable,
    nvrhi::DeviceHandle device)
    : m_shaderFactory(shaderFactory)
    , m_descriptorTable(descriptorTable)
    , m_commonPasses(commonPasses)
    , m_device(device)
{
    m_tessellationCountersBuffer.Create(kTessBufferFrameCount, "tesselation counters", m_device);
    m_debugBuffer.Create(65536, "debug", m_device, nvrhi::Format::RGBA32_FLOAT);
        
    //////////////////////////////////////////////////
    // Parameter buffers for shaders
    //////////////////////////////////////////////////
    m_fillInstantiateTemplateArgsParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(FillInstantiateTemplateArgsParams), "FillInstantiateTemplateArgsParams", engine::c_MaxRenderPassConstantBufferVersions));

    m_computeClusterTilingParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(ComputeClusterTilingParams), "ComputeClusterTilingParams", engine::c_MaxRenderPassConstantBufferVersions));

    m_fillClustersParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(FillClustersParams), "FillClustersParams", engine::c_MaxRenderPassConstantBufferVersions));

    m_copyClusterOffsetParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(CopyClusterOffsetParams), "CopyClusterOffsetParams", engine::c_MaxRenderPassConstantBufferVersions));

    m_fillBlasFromClasArgsParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(FillBlasFromClasArgsParams), "FillBlasFromClasArgsParams", engine::c_MaxRenderPassConstantBufferVersions));
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


void ClusterAccelBuilder::FillInstantiateTemplateArgs(nvrhi::IBuffer* outArgs, nvrhi::IBuffer* templateAddresses, uint32_t numTemplates, nvrhi::ICommandList* commandList)
{
    FillInstantiateTemplateArgsParams params = {};
    params.numTemplates = numTemplates;
    params.pad = uint3();

    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::FillInstantiateTemplateArgs");
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
        nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/fill_instantiate_template_args.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

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

void ClusterAccelBuilder::FillBlasFromClasArgs(nvrhi::IBuffer* outArgs, nvrhi::IBuffer* clusterOffsets,
    nvrhi::GpuVirtualAddress clasPtrsBaseAddress, uint32_t numInstances, nvrhi::ICommandList* commandList)
{
    FillBlasFromClasArgsParams params = {};
    params.clasAddressesBaseAddress = clasPtrsBaseAddress;
    params.numInstances = numInstances;

    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::FillBlasFromClasArgs");
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
        nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/fill_blas_from_clas_args.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

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

nvrhi::BufferHandle ClusterAccelBuilder::GenerateStructuredClusterTemplateArgs(const TemplateGrids &grids, nvrhi::ICommandList* commandList)
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

    uint32_t indexFormat = 0;
    switch (sizeof(TemplateGrids::IndexType))
    {
    case 1: indexFormat = (uint32_t)cluster::OperationIndexFormat::IndexFormat8bit; break;
    case 2: indexFormat = (uint32_t)cluster::OperationIndexFormat::IndexFormat16bit; break;
    case 4: indexFormat = (uint32_t)cluster::OperationIndexFormat::IndexFormat32bit; break;
    default: assert(false);
    }

    std::vector<cluster::IndirectTriangleTemplateArgs> createTemplateArgData(grids.descs.size());
    for (uint32_t i = 0; i < createTemplateArgData.size(); i++)
    {
        const TemplateGridDesc& grid = grids.descs[i];

        // Zero-initialize unused bit fields 
        createTemplateArgData[i] = { };
        createTemplateArgData[i] = cluster::IndirectTriangleTemplateArgs
        {
            .clusterId = 0,
            .clusterFlags = 0,
            .triangleCount = grid.getNumTriangles(),
            .vertexCount = grid.getNumVerts(),
            .positionTruncateBitCount = 0,
            .indexFormat = indexFormat,
            .baseGeometryIndexAndFlags = 0,
            .indexBufferStride = sizeof(grids.indices[0]),
            .vertexBufferStride = sizeof(grids.vertices[0]) * 3,
            .geometryIndexAndFlagsBufferStride = 0,
            .indexBuffer = indexBufferAddress + grid.indexOffset,
            .vertexBuffer = vertexBufferAddress + grid.vertexOffset,
            .geometryIndexAndFlagsBuffer = 0,
            .instantiationBoundingBoxLimit = 0
        };
    }

    nvrhi::BufferDesc clusterTemplateArgsDesc =
    {
        .byteSize = createTemplateArgData.size() * sizeof(createTemplateArgData[0]),
        .structStride = sizeof(createTemplateArgData[0]),
        .debugName = "ClusterTemplateArgs",
        .isDrawIndirectArgs = true,
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::IndirectArgument,
        .keepInitialState = true,
    };

    return CreateAndUploadBuffer(createTemplateArgData, clusterTemplateArgsDesc, commandList);
}

void ClusterAccelBuilder::InitStructuredClusterTemplates(uint32_t maxGeometryCountPerMesh, nvrhi::ICommandList* commandList)
{
    // only initialize if maxGeometryCount or quantNBits changes
    if (m_templateBuffers.dataBuffer.Get() != 0 && 
        m_templateBuffers.maxGeometryCountPerMesh == maxGeometryCountPerMesh &&
        m_templateBuffers.quantNBits == m_tessellatorConfig.quantNBits)
        return;

    nvrhi::utils::ScopedMarker marker(commandList, "InitStructuredClusterTemplates");
    m_templateBuffers.maxGeometryCountPerMesh = maxGeometryCountPerMesh;
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
            .maxGeometryIndex = maxGeometryCountPerMesh,
            .maxUniqueGeometryCount = 1,
            .maxTriangleCount = kClusterMaxTriangles,
            .maxVertexCount = kClusterMaxVertices,
            .maxTotalTriangleCount = grids.totalTriangles,
            .maxTotalVertexCount = grids.totalVertices,
            .minPositionTruncateBitCount = m_tessellatorConfig.quantNBits,
        }
    };
    cluster::OperationSizeInfo sizeInfo = m_device->getClusterOperationSizeInfo(operationParams);
        
    nvrhi::BufferHandle clusterTemplateArgsBuffer = GenerateStructuredClusterTemplateArgs(grids, commandList);
    
    RTXMGBuffer<uint32_t> templateSizesBuffer;
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
    std::vector<uint32_t> templateSizes = templateSizesBuffer.Download(commandList);
    
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
        .byteSize = sizeof(cluster::IndirectInstantiateTemplateArgs) * kNumTemplates,
        .structStride = sizeof(cluster::IndirectInstantiateTemplateArgs),
        .debugName = "InstantiateTemplateArgs",
        .canHaveUAVs = true,
        .isDrawIndirectArgs = true,
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::IndirectArgument,
        .keepInitialState = true,
    };

    RTXMGBuffer<cluster::IndirectInstantiateTemplateArgs> instantiateTemplateArgsBuffer(instantiateTemplateArgsDesc, m_device);
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

void ClusterAccelBuilder::BuildStructuredCLASes(ClusterAccels& accels, uint32_t maxGeometryCountPerMesh, 
    const nvrhi::BufferRange& tessCounterRange, nvrhi::ICommandList* commandList)
{
    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::BuildStructuredCLASes");

    cluster::OperationParams instantiateClasParams =
    {
        .maxArgCount = m_maxClusters,
        .type = cluster::OperationType::ClasInstantiateTemplates,
        .mode = cluster::OperationMode::ExplicitDestinations,
        .flags = cluster::OperationFlags::None,
        .clas =
        {
            .vertexFormat = nvrhi::Format::RGB32_FLOAT,
            .maxGeometryIndex = maxGeometryCountPerMesh,
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

void ClusterAccelBuilder::FillInstanceClusters(const RTXMGScene& scene, ClusterAccels& accels, nvrhi::ICommandList* commandList)
{
#if ENABLE_ACCEL_BUILDER_DEBUG
    commandList->clearBufferUInt(m_debugBuffer, 0);
#endif

    const auto& subdMeshes = scene.GetSubdMeshes();
    const auto& instances = scene.GetSubdMeshInstances();

    nvrhi::utils::ScopedMarker marker(commandList, "FillInstanceClusters");
    stats::clusterAccelSamplers.fillClustersTime.Start(commandList);

    uint32_t surfaceOffset{ 0 };
    for (uint32_t instanceIndex = 0; instanceIndex < instances.size(); ++instanceIndex)
    {
        const auto& instance = instances[instanceIndex];
        assert(instance.meshInstance.get());
        const auto& donutMeshInfo = instance.meshInstance->GetMesh();
        assert(donutMeshInfo.get());
        uint32_t firstGeometryIndex = donutMeshInfo->geometries[0]->globalGeometryIndex;

        const auto& subd = *subdMeshes[instance.meshID];
        
        const uint32_t surfaceCount = subd.SurfaceCount();

        FillClustersParams params = {};
        params.instanceIndex = instanceIndex;
        params.quantNBits = m_tessellatorConfig.quantNBits;
        params.isolationLevel = subd.m_dynamicIsolationLevel;
        params.globalDisplacementScale = m_tessellatorConfig.displacementScale;
        params.clusterPattern = uint32_t(m_tessellatorConfig.clusterPattern);
        params.firstGeometryIndex = firstGeometryIndex;
        commandList->writeBuffer(m_fillClustersParamsBuffer, &params, sizeof(FillClustersParams));

        auto bindingSetDesc = nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_gridSamplersBuffer,
                nvrhi::Format::UNKNOWN,
                nvrhi::BufferRange(surfaceOffset * sizeof(GridSampler), surfaceCount * sizeof(GridSampler))))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_clusterOffsetCountsBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_clustersBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, subd.m_positionsBuffer))
            // Subd
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, subd.m_vertexDeviceData.surfaceDescriptors))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(5, subd.m_vertexDeviceData.controlPointIndices))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(6, subd.m_vertexDeviceData.patchPointsOffsets))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(7, subd.GetTopologyMap()->plansBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(8, subd.GetTopologyMap()->subpatchTreesArraysBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(9, subd.GetTopologyMap()->patchPointIndicesArraysBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(10, subd.GetTopologyMap()->stencilMatrixArraysBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(11, subd.m_vertexDeviceData.patchPoints))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(12, scene.GetGeometryBuffer()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(13, scene.GetMaterialBuffer()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(14, subd.m_surfaceToGeometryIndexBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(15, subd.m_texcoordDeviceData.surfaceDescriptors))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(16, subd.m_texcoordDeviceData.controlPointIndices))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(17, subd.m_texcoordDeviceData.patchPointsOffsets))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(18, subd.m_texcoordDeviceData.patchPoints))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(19, subd.m_texcoordsBuffer))
            .addItem(nvrhi::BindingSetItem::Sampler(0, scene.GetDisplacementSampler()))

            // gatherer
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, accels.clusterVertexPositionsBuffer)) 
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, accels.clusterShadingDataBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_debugBuffer))
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_fillClustersParamsBuffer));

        nvrhi::BindingSetHandle bindingSet;
        if (!nvrhi::utils::CreateBindingSetAndLayout(m_device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_fillClustersBL, bindingSet))
        {
            log::fatal("Failed to create binding set and layout for fill_clusters.hlsl");
        }

        if (!m_fillClustersBindlessBL)
        {
            nvrhi::BindlessLayoutDesc bindlessLayoutDesc;
            bindlessLayoutDesc.visibility = nvrhi::ShaderType::All;
            bindlessLayoutDesc.firstSlot = 0;
            bindlessLayoutDesc.maxCapacity = 1024;
            bindlessLayoutDesc.registerSpaces = {
                nvrhi::BindingLayoutItem::Texture_SRV(2) };
            m_fillClustersBindlessBL = m_device->createBindlessLayout(bindlessLayoutDesc);
        }

        auto GetFillClustersPSO = [this](const FillClustersPermutation& shaderPermutation)
            {
                if (!m_fillClustersPSOs[shaderPermutation.index()])
                {
                    std::vector<donut::engine::ShaderMacro> fillClustersMacros;
                    fillClustersMacros.push_back(donut::engine::ShaderMacro("DISPLACEMENT_MAPS", shaderPermutation.isDisplacementEnabled() ? "1" : "0"));
                    fillClustersMacros.push_back(donut::engine::ShaderMacro("SURFACE_TYPE", toString(shaderPermutation.surfaceType())));
                    nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/fill_clusters.hlsl", "FillClustersMain", &fillClustersMacros, nvrhi::ShaderType::Compute);

                    auto computePipelineDesc = nvrhi::ComputePipelineDesc()
                        .setComputeShader(shader)
                        .addBindingLayout(m_fillClustersBL)
                        .addBindingLayout(m_fillClustersBindlessBL);

                    m_fillClustersPSOs[shaderPermutation.index()] = m_device->createComputePipeline(computePipelineDesc);
                }
                return m_fillClustersPSOs[shaderPermutation.index()];
            };
        
        if (!m_fillClustersTexcoordsPSO)
        {
            nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/fill_clusters.hlsl", "FillClustersTexcoordsMain", nullptr, nvrhi::ShaderType::Compute);

            auto computePipelineDesc = nvrhi::ComputePipelineDesc()
                .setComputeShader(shader)
                .addBindingLayout(m_fillClustersBL)
                .addBindingLayout(m_fillClustersBindlessBL);

            m_fillClustersTexcoordsPSO = m_device->createComputePipeline(computePipelineDesc);
        }

        auto state = nvrhi::ComputeState()
            .addBindingSet(bindingSet)
            .addBindingSet(m_descriptorTable)
            .setIndirectParams(m_fillClustersDispatchIndirectBuffer);

        if (m_tessellatorConfig.enableMonolithicClusterBuild)
        {
            FillClustersPermutation shaderPermutation = { subd.m_hasDisplacementMaterial, ShaderPermutationSurfaceType::All };
            state.setPipeline(GetFillClustersPSO(shaderPermutation));
            commandList->setComputeState(state);
            uint32_t dispatchIndirectArgsOffset = (instanceIndex * ClusterDispatchType::NumTypes + ClusterDispatchType::Limit) * uint32_t(m_fillClustersDispatchIndirectBuffer.GetElementBytes());
            commandList->dispatchIndirect(dispatchIndirectArgsOffset);
        }
        else
        {
            for (uint32_t i = 0; i <= uint32_t(ShaderPermutationSurfaceType::Limit); i++)
            {
                FillClustersPermutation shaderPermutation = { subd.m_hasDisplacementMaterial, ShaderPermutationSurfaceType(i) };
                state.setPipeline(GetFillClustersPSO(shaderPermutation));
                commandList->setComputeState(state);
                uint32_t dispatchIndirectArgsOffset = (instanceIndex * ClusterDispatchType::NumTypes + ClusterDispatchType(i)) * uint32_t(m_fillClustersDispatchIndirectBuffer.GetElementBytes());
                commandList->dispatchIndirect(dispatchIndirectArgsOffset);
            }
        }
        
        state.setPipeline(m_fillClustersTexcoordsPSO);
        commandList->setComputeState(state);
        uint32_t dispatchIndirectArgsOffset = (instanceIndex * ClusterDispatchType::NumTypes + ClusterDispatchType::All) * uint32_t(m_fillClustersDispatchIndirectBuffer.GetElementBytes());
        commandList->dispatchIndirect(dispatchIndirectArgsOffset);

        surfaceOffset += surfaceCount;
    }

    stats::clusterAccelSamplers.fillClustersTime.Stop();

#if ENABLE_ACCEL_BUILDER_DEBUG
    std::vector<math::float4> data;
    data = m_debugBuffer.Download(commandList);
    for (int idx = 0; idx < 12; idx++)
    {
        donut::log::info("%d: %f %f %f %f", idx, data[idx].x, data[idx].y, data[idx].z, data[idx].w);
    }
#endif
}

void ClusterAccelBuilder::ComputeInstanceClusterTiling(uint32_t instanceIndex, 
    const SubdivisionSurface& subdivisionSurface,
    ClusterAccels& accels,
    uint32_t firstGeometryIndex,
    nvrhi::IBuffer* geometryBuffer,
    nvrhi::IBuffer* materialBuffer,
    nvrhi::ISampler* displacementSampler,
    const donut::math::affine3& localToWorld,
    uint32_t surfaceOffset,
    uint32_t surfaceCount,
    const nvrhi::BufferRange& tessCounterRange,
    nvrhi::ICommandList* commandList)
{
    ComputeClusterTilingParams params = {};
    params.matWorldToClip = m_tessellatorConfig.camera->GetProjectionMatrix() * m_tessellatorConfig.camera->GetViewMatrix();
    affineToColumnMajor(localToWorld, params.localToWorld.m_data); // params.localToWorld;
    params.viewportSize.x = float(m_tessellatorConfig.viewportSize.x);
    params.viewportSize.y = float(m_tessellatorConfig.viewportSize.y);
    params.firstGeometryIndex = firstGeometryIndex;
    params.isolationLevel = subdivisionSurface.m_dynamicIsolationLevel;
    params.coarseTessellationRate = m_tessellatorConfig.coarseTessellationRate;
    params.fineTessellationRate = m_tessellatorConfig.fineTessellationRate;
    params.cameraPos = m_tessellatorConfig.camera->GetEye();
    params.aabb = subdivisionSurface.m_aabb * localToWorld;
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
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, subdivisionSurface.m_positionsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, geometryBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(2, materialBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, subdivisionSurface.m_surfaceToGeometryIndexBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, subdivisionSurface.m_vertexDeviceData.surfaceDescriptors))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(5, subdivisionSurface.m_vertexDeviceData.controlPointIndices))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(6, subdivisionSurface.m_vertexDeviceData.patchPointsOffsets))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(7, subdivisionSurface.GetTopologyMap()->plansBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(8, subdivisionSurface.GetTopologyMap()->subpatchTreesArraysBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(9, subdivisionSurface.GetTopologyMap()->patchPointIndicesArraysBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(10, subdivisionSurface.GetTopologyMap()->stencilMatrixArraysBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(11, m_templateBuffers.instantiationSizesBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(12, m_templateBuffers.addressesBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(13, subdivisionSurface.m_texcoordDeviceData.surfaceDescriptors))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(14, subdivisionSurface.m_texcoordDeviceData.controlPointIndices))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(15, subdivisionSurface.m_texcoordDeviceData.patchPointsOffsets))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(16, subdivisionSurface.m_texcoordsBuffer))
        .addItem(nvrhi::BindingSetItem::Sampler(0, displacementSampler))
        .addItem(nvrhi::BindingSetItem::Sampler(1, m_commonPasses->m_LinearClampSampler)) // hiZ sampler
        
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_gridSamplersBuffer,
            nvrhi::Format::UNKNOWN,
            nvrhi::BufferRange(surfaceOffset * sizeof(GridSampler), surfaceCount * sizeof(GridSampler))))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_tessellationCountersBuffer, nvrhi::Format::UNKNOWN, tessCounterRange))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_clustersBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(3, accels.clusterShadingDataBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(4, m_clasIndirectArgDataBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(5, accels.clasPtrsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(6, subdivisionSurface.m_vertexDeviceData.patchPoints))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(7, subdivisionSurface.m_texcoordDeviceData.patchPoints))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(8, m_debugBuffer));

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(m_device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_computeClusterTilingBL, bindingSet))
    {
        log::fatal("Failed to create binding set and layout for compute_cluster_tiling.hlsl");
    }

    auto hizDesc = ZBuffer::GetHiZDesc(m_tessellatorConfig.zbuffer, m_commonPasses->m_BlackTexture);

    nvrhi::BindingSetHandle hizSet;
    nvrhi::BindingLayoutHandle hizLayout;
    if (!nvrhi::utils::CreateBindingSetAndLayout(m_device, nvrhi::ShaderType::Compute, 3, hizDesc, m_computeClusterTilingHizBL, hizSet))
    {
        log::fatal("Failed to create hiz binding set and layout for compute_cluster_tiling.hlsl");
    }

    ComputeClusterTilingPermutation shaderPermutation(subdivisionSurface.m_hasDisplacementMaterial,
        m_tessellatorConfig.enableFrustumVisibility,
        m_tessellatorConfig.tessMode,
        m_tessellatorConfig.visMode,
        ShaderPermutationSurfaceType::PureBSpline);

    if (!m_computeClusterTilingBindlessBL)
    {
        nvrhi::BindlessLayoutDesc bindlessLayoutDesc;
        bindlessLayoutDesc.visibility = nvrhi::ShaderType::All;
        bindlessLayoutDesc.firstSlot = 0;
        bindlessLayoutDesc.maxCapacity = 1024;
        bindlessLayoutDesc.registerSpaces = {
            nvrhi::BindingLayoutItem::Texture_SRV(2) };
        m_computeClusterTilingBindlessBL = m_device->createBindlessLayout(bindlessLayoutDesc);
    }

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
                nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/compute_cluster_tiling.hlsl", "main", &macros, tilingDesc);

                auto computePipelineDesc = nvrhi::ComputePipelineDesc()
                    .setComputeShader(shader)
                    .addBindingLayout(m_computeClusterTilingBL)
                    .addBindingLayout(m_computeClusterTilingHizBL)
                    .addBindingLayout(m_computeClusterTilingBindlessBL);

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
        ClusterDispatchType dispatchType = ClusterDispatchType::All;
        CopyClusterOffset(instanceIndex, dispatchType, tessCounterRange, commandList);
    }
    else
    {
        // Loop
        for (uint32_t i = 0; i <= uint32_t(ClusterDispatchType::Limit); i++)
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
            ClusterDispatchType dispatchType = ClusterDispatchType(i);
            CopyClusterOffset(instanceIndex, dispatchType, tessCounterRange, commandList);
        }
    }
    
}

void ClusterAccelBuilder::CopyClusterOffset(uint32_t instanceIndex,
    ClusterDispatchType dispatchType, const nvrhi::BufferRange& tessCounterRange, nvrhi::ICommandList* commandList)
{
    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::CopyClusterOffset");
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
        nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/copy_cluster_offset.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

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

void ClusterAccelBuilder::BuildBlasFromClas(ClusterAccels& accels, std::span<Instance const> instances, nvrhi::ICommandList* commandList)
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
void ClusterAccelBuilder::UpdateMemoryAllocations(ClusterAccels& accels, uint32_t numInstances, uint32_t sceneSubdPatches)
{
    uint32_t maxClusters = std::min(kMaxApiClusterCount, m_tessellatorConfig.memorySettings.maxClusters);
    maxClusters = std::max(uint32_t(1), maxClusters);

    // Reallocate memory if settings changed
    size_t maxClusterBlocks = (m_tessellatorConfig.memorySettings.clasBufferBytes + (size_t(cluster::kClasByteAlignment) - 1ull)) / size_t(cluster::kClasByteAlignment);
    maxClusterBlocks = std::max(size_t(1), maxClusterBlocks);
    size_t maxClasBytes = size_t(cluster::kClasByteAlignment) * maxClusterBlocks;

    uint32_t maxVertices = uint32_t(m_tessellatorConfig.memorySettings.vertexBufferBytes / sizeof(float3));
    maxVertices = std::max(kClusterMaxVertices, maxVertices);

    bool numInstancesChanged = m_numInstances != numInstances;
    bool sceneSubdPatchesChanged = m_sceneSubdPatches != sceneSubdPatches;
    bool numClustersChanged = m_maxClusters != maxClusters;
    bool clasBytesChanged = m_maxClasBytes != maxClasBytes;
    bool maxVerticesChanged = m_maxVertices != maxVertices;

    m_numInstances = numInstances;
    m_sceneSubdPatches = sceneSubdPatches;
    m_maxClusters = maxClusters;
    m_maxClasBytes = maxClasBytes;
    m_maxVertices = maxVertices;
    
    // No allocations needed
    if (!numInstancesChanged && !sceneSubdPatchesChanged && !numClustersChanged && !clasBytesChanged && !maxVerticesChanged)
    {
        return;
    }

    // Wait for idle to ensure resources are not in use
    m_device->waitForIdle();

    if (numInstancesChanged)
    {
        m_clusterOffsetCountsBuffer.Release();
        m_fillClustersDispatchIndirectBuffer.Release();
        m_blasFromClasIndirectArgsBuffer.Release();
        accels.blasPtrsBuffer.Release();
        accels.blasSizesBuffer.Release();

        m_clusterOffsetCountsBuffer.Create(m_numInstances * ClusterDispatchType::NumTypes, "ClusterOffsets", m_device);
        nvrhi::BufferDesc dispatchIndirectDesc =
        {
            .byteSize = m_numInstances * ClusterDispatchType::NumTypes * m_fillClustersDispatchIndirectBuffer.GetElementBytes(),
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
        m_clasIndirectArgDataBuffer.Create(m_maxClusters, "indirect arg data", m_device);

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
}

void ClusterAccelBuilder::BuildAccel(const RTXMGScene& scene, const TessellatorConfig& config,
    ClusterAccels& accels, ClusterStatistics& stats, uint32_t frameIndex, nvrhi::ICommandList* commandList)
{
    m_tessellatorConfig = config;

    const auto& subdMeshes = scene.GetSubdMeshes();
    const auto& instances = scene.GetSubdMeshInstances();

    if (subdMeshes.empty() || instances.empty())
        return;

    UpdateMemoryAllocations(accels, uint32_t(instances.size()), scene.TotalSubdPatchCount());

    const uint32_t maxGeometryCountPerMesh = uint32_t(scene.GetSceneGraph()->GetMaxGeometryCountPerMesh());
    InitStructuredClusterTemplates(maxGeometryCountPerMesh, commandList);
    
    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::BuildAccel");

    uint32_t tessCounterIndex = (m_buildAccelFrameIndex % kTessBufferFrameCount);
    nvrhi::BufferRange tessCounterRange = { m_tessellationCountersBuffer.GetElementBytes() * tessCounterIndex, m_tessellationCountersBuffer.GetElementBytes() };

    // Clear tessellation counters for this frame
    TessellationCounters tessCounters = {};
    m_tessellationCountersBuffer.UploadElement(tessCounters, tessCounterIndex, commandList);

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

            assert(inst.meshInstance.get());
            const auto& donutMeshInfo = inst.meshInstance->GetMesh();
            assert(donutMeshInfo.get());
            uint32_t firstGeometryIndex = donutMeshInfo->geometries[0]->globalGeometryIndex;
            uint32_t surfaceCount{ subd.SurfaceCount() };

            ComputeInstanceClusterTiling(i, subd, accels, firstGeometryIndex, scene.GetGeometryBuffer(), scene.GetMaterialBuffer(), scene.GetDisplacementSampler(),
                inst.localToWorld, surfaceOffset, surfaceCount, tessCounterRange, commandList);

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
    BuildStructuredCLASes(accels, maxGeometryCountPerMesh, tessCounterRange, commandList);
    stats::clusterAccelSamplers.buildClasTime.Stop();

    BuildBlasFromClas(accels, instances, commandList);
    
    // Async read of counters
    auto counterBufferData = m_tessellationCountersBuffer.Download(commandList, true);
    TessellationCounters counters = counterBufferData[(tessCounterIndex + 1) % kTessBufferFrameCount];

    // Record the desired required memory instead of the max
    stats.desired.m_numTriangles = counters.desiredTriangles;
    stats.desired.m_numClusters = counters.desiredClusters;
    stats.desired.m_vertexBufferSize = accels.clusterVertexPositionsBuffer.GetElementBytes() * counters.desiredVertices;
    stats.desired.m_clasSize = counters.DesiredClasBytes();
    stats.desired.m_clusterDataSize = (m_clustersBuffer.GetElementBytes() + 
        accels.clusterShadingDataBuffer.GetElementBytes() +
        accels.clasPtrsBuffer.GetElementBytes()) * counters.desiredClusters;
    stats.desired.m_blasSize = m_createBlasSizeInfo.resultMaxSizeInBytes;
    stats.desired.m_blasScratchSize = m_createBlasSizeInfo.scratchSizeInBytes;

    // Atomics are expensive so we don't track the number of allocated triangles
    stats.allocated.m_numTriangles = counters.desiredTriangles;
    stats.allocated.m_numClusters = m_maxClusters;
    stats.allocated.m_vertexBufferSize = accels.clusterVertexPositionsBuffer.GetBytes();
    stats.allocated.m_clasSize = accels.clasBuffer.GetBytes();
    stats.allocated.m_clusterDataSize = m_clustersBuffer.GetBytes() + accels.clusterShadingDataBuffer.GetBytes() + accels.clasPtrsBuffer.GetBytes();
    stats.allocated.m_blasSize = accels.blasBuffer.GetBytes();
    stats.allocated.m_blasScratchSize = m_createBlasSizeInfo.scratchSizeInBytes;

    m_buildAccelFrameIndex++;
}
