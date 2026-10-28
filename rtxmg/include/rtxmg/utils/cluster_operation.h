/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/*-------------------------------------------------------------------------------------------------
Cluster-operation layouts that differ per device. NVAPI and Vulkan use nvrhi::rt::cluster::nv and
32-bit result sizes; DXR clusters use nvrhi::rt::cluster::dxr and 64-bit sizes. A D3D12 device picks
its API at creation, so host code fills the nv layout and repacks it on upload, and shaders that
use the layouts have a "_dxr" variant (see donut/shaders/cluster_args.hlsli).
-------------------------------------------------------------------------------------------------*/

#pragma once

#include <rtxmg/utils/buffer.h>

#include <algorithm>
#include <string>
#include <vector>

inline nvrhi::ClusterApi GetClusterApi(nvrhi::IDevice* device)
{
    nvrhi::RayTracingClustersFeatureInfo info = {};
    device->queryFeatureSupport(nvrhi::Feature::RayTracingClusters, &info, sizeof(info));
    return info.api;
}

inline bool UsesDxrClusterOperations(nvrhi::IDevice* device)
{
    return GetClusterApi(device) == nvrhi::ClusterApi::Dxr;
}

// Picks the "_dxr" build of a shader that has one, e.g. "dir/name.hlsl" -> "dir/name_dxr.hlsl".
inline std::string GetClusterApiShaderPath(nvrhi::IDevice* device, const std::string& path)
{
    if (!UsesDxrClusterOperations(device))
        return path;
    const size_t extension = path.rfind(".hlsl");
    return path.substr(0, extension) + "_dxr" + path.substr(extension);
}

inline uint32_t GetClusterOperationSizeBytes(nvrhi::IDevice* device)
{
    return UsesDxrClusterOperations(device) ? uint32_t(sizeof(nvrhi::rt::cluster::dxr::OperationSize))
                                              : uint32_t(sizeof(nvrhi::rt::cluster::nv::OperationSize));
}

inline uint32_t GetTriangleClasArgsBytes(nvrhi::IDevice* device)
{
    return UsesDxrClusterOperations(device) ? uint32_t(sizeof(nvrhi::rt::cluster::dxr::IndirectTriangleClasArgs))
                                              : uint32_t(sizeof(nvrhi::rt::cluster::nv::IndirectTriangleClasArgs));
}

// Only the member order differs, so buffers of these are sized the same on every device.
static_assert(sizeof(nvrhi::rt::cluster::nv::IndirectInstantiateTemplateArgs) ==
              sizeof(nvrhi::rt::cluster::dxr::IndirectInstantiateTemplateArgs));

inline nvrhi::rt::cluster::dxr::IndirectTriangleClasArgs ToDxr(const nvrhi::rt::cluster::nv::IndirectTriangleClasArgs& a)
{
    nvrhi::rt::cluster::dxr::IndirectTriangleClasArgs d = {};
    d.clusterId                         = a.clusterId;
    d.clusterFlags                      = a.clusterFlags;
    d.triangleCount                     = uint16_t(a.triangleCount);
    d.vertexCount                       = uint16_t(a.vertexCount);
    d.baseGeometryIndexAndFlags         = a.baseGeometryIndexAndFlags;
    d.vertexBufferStride                = a.vertexBufferStride;
    d.indexBufferStride                 = a.indexBufferStride;
    d.opacityMicromapIndexBufferStride  = a.opacityMicromapIndexBufferStride;
    d.geometryIndexAndFlagsBufferStride = a.geometryIndexAndFlagsBufferStride;
    d.positionTruncateBitCount          = uint16_t(a.positionTruncateBitCount);
    d.vertexBuffer                      = a.vertexBuffer;
    d.indexBuffer                       = a.indexBuffer;
    d.geometryIndexAndFlagsBuffer       = a.geometryIndexAndFlagsBuffer;
    d.opacityMicromapArray              = a.opacityMicromapArray;
    d.opacityMicromapIndexBuffer        = a.opacityMicromapIndexBuffer;
    return d;
}

inline nvrhi::rt::cluster::dxr::IndirectTriangleTemplateArgs ToDxr(const nvrhi::rt::cluster::nv::IndirectTriangleTemplateArgs& a)
{
    nvrhi::rt::cluster::dxr::IndirectTriangleTemplateArgs d = {};
    d.clusterId                         = a.clusterId;
    d.clusterFlags                      = a.clusterFlags;
    d.triangleCount                     = uint16_t(a.triangleCount);
    d.vertexCount                       = uint16_t(a.vertexCount);
    d.baseGeometryIndexAndFlags         = a.baseGeometryIndexAndFlags;
    d.vertexBufferStride                = a.vertexBufferStride;
    d.indexBufferStride                 = a.indexBufferStride;
    d.opacityMicromapIndexBufferStride  = a.opacityMicromapIndexBufferStride;
    d.geometryIndexAndFlagsBufferStride = a.geometryIndexAndFlagsBufferStride;
    d.positionTruncateBitCount          = uint16_t(a.positionTruncateBitCount);
    d.vertexBuffer                      = a.vertexBuffer;
    d.indexBuffer                       = a.indexBuffer;
    d.geometryIndexAndFlagsBuffer       = a.geometryIndexAndFlagsBuffer;
    d.opacityMicromapArray              = a.opacityMicromapArray;
    d.opacityMicromapIndexBuffer        = a.opacityMicromapIndexBuffer;
    d.instantiationBoundingBoxLimit     = a.instantiationBoundingBoxLimit;
    return d;
}

// Creates a buffer from NVAPI-layout args, repacked when the device takes DXR clusters.
// desc supplies everything but byteSize and structStride.
template <typename Args>
nvrhi::BufferHandle CreateAndUploadClusterArgs(const std::vector<Args>& args, nvrhi::BufferDesc desc,
                                               nvrhi::ICommandList* commandList)
{
    auto upload = [&](const auto& data)
    {
        const size_t stride = sizeof(data[0]);
        desc.byteSize       = std::max<size_t>(data.size(), 1) * stride;
        desc.structStride   = uint32_t(stride);
        return CreateAndUploadBuffer(data, desc, commandList);
    };

    if (!UsesDxrClusterOperations(commandList->getDevice()))
        return upload(args);

    std::vector<decltype(ToDxr(args[0]))> converted;
    converted.reserve(args.size());
    for (const Args& a : args)
        converted.push_back(ToDxr(a));
    return upload(converted);
}

// Result sizes of a cluster operation, stored at the width the device writes them.
class ClusterOperationSizesBuffer
{
public:
    operator nvrhi::IBuffer*() const { return GetBuffer().Get(); }
    nvrhi::BufferHandle GetBuffer() const { return m_wide.GetBuffer() ? m_wide.GetBuffer() : m_narrow.GetBuffer(); }
    size_t GetBytes() const { return m_wide.GetBytes() + m_narrow.GetBytes(); }
    size_t GetVramBytes() const { return m_wide.GetVramBytes() + m_narrow.GetVramBytes(); }
    void Release() { m_wide.Release(); m_narrow.Release(); }

    void Create(size_t nElements, const char* name, nvrhi::IDevice* device)
    {
        Create(GetGenericDesc(nElements, GetClusterOperationSizeBytes(device), name), device);
    }

    void Create(const nvrhi::BufferDesc& desc, nvrhi::IDevice* device)
    {
        if (desc.structStride == sizeof(uint64_t))
            m_wide.Create(desc, device);
        else
            m_narrow.Create(desc, device);
    }

    std::vector<uint64_t> Download(nvrhi::ICommandList* commandList, bool async = false)
    {
        if (m_wide.GetBuffer())
            return m_wide.Download(commandList, async);
        const std::vector<uint32_t> narrow = m_narrow.Download(commandList, async);
        return std::vector<uint64_t>(narrow.begin(), narrow.end());
    }

    void Log(nvrhi::ICommandList* commandList, vectorlog::FormatOptions options = {})
    {
        if (m_wide.GetBuffer())
            m_wide.Log(commandList, options);
        else
            m_narrow.Log(commandList, options);
    }

private:
    RTXMGBuffer<uint64_t> m_wide;
    RTXMGBuffer<uint32_t> m_narrow;
};
