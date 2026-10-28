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

#include <donut/core/log.h>
#include <donut/core/math/math.h>

#include <nvrhi/utils.h>
#include <vector>

#include <rtxmg/utils/vectorlog.h>
#include <rtxmg/utils/vram_bytes.h>

nvrhi::BufferDesc GetGenericDesc(size_t nElements, uint32_t elementSize, const char* name, nvrhi::Format format = nvrhi::Format::UNKNOWN);
nvrhi::BufferDesc GetReadbackDesc(const nvrhi::BufferDesc& desc);

inline nvrhi::BufferHandle CreateBuffer(const nvrhi::BufferDesc& desc, nvrhi::IDevice* device)
{
    nvrhi::BufferHandle buffer = device->createBuffer(desc);
    return buffer;
}

inline nvrhi::BufferHandle CreateBuffer(size_t nElements, uint32_t elementSize, const char* name, nvrhi::IDevice *device, nvrhi::Format format = nvrhi::Format::UNKNOWN)
{
    return CreateBuffer(GetGenericDesc(nElements, elementSize, name, format), device);
}

template <typename T>
nvrhi::BufferHandle
CreateAndUploadBuffer(const std::vector<T>& data, const nvrhi::BufferDesc &desc,
    nvrhi::ICommandList* commandList)
{
    assert(desc.byteSize > 0);
    nvrhi::BufferHandle buffer = commandList->getDevice()->createBuffer(desc);

    size_t minSize = std::min(desc.byteSize, data.size() * sizeof(T));
    if (minSize > 0)
    {
        commandList->writeBuffer(buffer, data.data(), minSize);
    }
    return buffer;
}

template <typename T>
nvrhi::BufferHandle
CreateAndUploadBuffer(const std::vector<T>& data, const char* name,
    nvrhi::ICommandList* commandList, nvrhi::Format format = nvrhi::Format::UNKNOWN)
{
    nvrhi::BufferDesc desc = GetGenericDesc(data.size(), sizeof(T), name, format);
    return CreateAndUploadBuffer(data, desc, commandList);
}

void DownloadBuffer(nvrhi::IBuffer* src, void* dest, nvrhi::IBuffer* staging, bool async, nvrhi::ICommandList* commandList);

template <typename T>
void DownloadBuffer(nvrhi::IBuffer* src, std::vector<T>& vec, nvrhi::IBuffer* staging, bool async, nvrhi::ICommandList* commandList)
{
    size_t nelems = src->getDesc().byteSize / sizeof(T);
    vec.clear();
    vec.resize(nelems);

    DownloadBuffer(src, (void*)&vec[0], staging, async, commandList);
}

template<typename T>
class RTXMGBuffer
{
protected:
    nvrhi::BufferHandle m_buffer;
    nvrhi::BufferHandle m_readbackBuffer;
public:
    typedef vectorlog::OutputLambda<T>::Type OutputLambdaType;
    typedef T ElementType;
     
    operator nvrhi::BufferHandle() const { return m_buffer; }
    operator nvrhi::IBuffer*() { return m_buffer.Get(); }
    operator nvrhi::IBuffer*() const { return m_buffer.Get(); }
    operator nvrhi::IBuffer&() { return *m_buffer.Get(); }
    nvrhi::BufferHandle GetBuffer() const { return m_buffer; }

    size_t GetElementBytes() const { return sizeof(T); }
    uint32_t GetNumElements() const { return m_buffer ? uint32_t(m_buffer->getDesc().byteSize / sizeof(T)) : 0; }
    size_t GetBytes() const { return m_buffer ? m_buffer->getDesc().byteSize : 0u; }
    size_t GetVramBytes() const { return size_t(rtxmg::BufferVramBytes(m_buffer.Get())); }
    nvrhi::GpuVirtualAddress GetGpuVirtualAddress() const { return m_buffer->getGpuVirtualAddress(); }
    void Release() { m_buffer = nullptr; }

    void Create(const nvrhi::BufferDesc& desc, nvrhi::IDevice* device)
    {
        if (GetBytes() >= desc.byteSize)
        {
            // reuse
            return;
        }
        m_buffer = device->createBuffer(desc);
    }

    void Create(size_t nElements, const char* name, nvrhi::IDevice* device, nvrhi::Format format = nvrhi::Format::UNKNOWN)
    {
        auto desc = GetGenericDesc(nElements, sizeof(T), name, format);
        Create(desc, device);
    }

    RTXMGBuffer() {}
    RTXMGBuffer(const nvrhi::BufferDesc& desc, nvrhi::IDevice* device)
    {
        Create(desc, device);
    }

    void Upload(const std::vector<T>& data, nvrhi::ICommandList* commandList) const
    {
        if (data.size() > 0)
        {
            commandList->writeBuffer(m_buffer, data.data(), data.size() * sizeof(T));
        }
    }

    void UploadElement(const T& data, uint32_t index, nvrhi::ICommandList* commandList) const
    {
        commandList->writeBuffer(m_buffer, &data, sizeof(T), index * sizeof(T));
    }

    std::vector<T> Download(nvrhi::ICommandList* commandList, bool async = false)
    {
        auto readBackDesc = GetReadbackDesc(m_buffer->getDesc());
        if (!m_readbackBuffer || readBackDesc.byteSize > m_readbackBuffer->getDesc().byteSize)
        {
            m_readbackBuffer = commandList->getDevice()->createBuffer(readBackDesc);
        }
        
        std::vector<T> outData;
        DownloadBuffer<T>(m_buffer, outData, m_readbackBuffer, async, commandList);

        return outData;
    }

    void OutputStream(nvrhi::ICommandList* commandList, OutputLambdaType outputElementLambda, std::ostream* optOutputStream = nullptr, vectorlog::FormatOptions options = {})
    {
        auto data = Download(commandList);

        if (options.header)
        {
            std::stringstream ss;

            const nvrhi::BufferDesc& desc = m_buffer->getDesc();
            size_t numElements = data.size();

            ss << desc.debugName << "<" << typeid(T).name() << ">"
                << "(n:" << data.size() << " bytes: " << data.size() * sizeof(data[0]) << ")"
                << " printing " << options.startIndex << " to " << (options.startIndex + options.count);

            vectorlog::EndLine(ss, optOutputStream);
        }
        
        vectorlog::OutputStream(data, outputElementLambda, optOutputStream, options);
    }

    void OutputStream(nvrhi::ICommandList* commandList, std::ostream& outputStream, vectorlog::FormatOptions options = {})
    {
        OutputStream(commandList, vectorlog::OutputElement<T>, &outputStream, options);
    }

    void Log(nvrhi::ICommandList* commandList, OutputLambdaType outputElementLambda, vectorlog::FormatOptions options = {})
    {
        OutputStream(commandList, outputElementLambda, nullptr, options);
    }

    void Log(nvrhi::ICommandList* commandList, vectorlog::FormatOptions options = {})
    {
        Log(commandList, vectorlog::OutputElement<T>, options);
    }
};
