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

#include "rtxmg/utils/buffer.h"

nvrhi::BufferDesc GetGenericDesc(size_t nElements, uint32_t elementSize, const char* name, nvrhi::Format format)
{
    nElements = std::max<size_t>(1, nElements);
    return nvrhi::BufferDesc()
        .setByteSize(nElements * elementSize)
        .setCanHaveTypedViews(true)
        .setCanHaveUAVs(true)
        .setDebugName(name)
        .setFormat(format)
        .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
        .setKeepInitialState(true)
        .setStructStride(elementSize)
        .setCanHaveRawViews(true);
}

nvrhi::BufferDesc GetReadbackDesc(const nvrhi::BufferDesc& desc)
{
    nvrhi::BufferDesc readbackBufferDesc = nvrhi::BufferDesc()
        .setByteSize(desc.byteSize)
        .setCpuAccess(nvrhi::CpuAccessMode::Read)
        .setDebugName(desc.debugName + " Readback")
        .setFormat(desc.format)
        .setInitialState(nvrhi::ResourceStates::CopyDest)
        .setKeepInitialState(true);

    return readbackBufferDesc;
}

void DownloadBuffer(nvrhi::IBuffer* src, void* dest, nvrhi::IBuffer* staging, bool async, nvrhi::ICommandList* commandList)
{
    size_t numBytes = src->getDesc().byteSize;
    commandList->copyBuffer(staging, 0, src, 0, numBytes);

    if (!async)
    {
        commandList->close();
        commandList->getDevice()->executeCommandList(commandList);
        commandList->getDevice()->waitForIdle();
    }
    void* mappedBuffer = commandList->getDevice()->mapBuffer(staging, nvrhi::CpuAccessMode::Read);
    if (mappedBuffer)
        memcpy(dest, mappedBuffer, numBytes);
    else
        memset(dest, 0, numBytes);
    commandList->getDevice()->unmapBuffer(staging);

    if (!async)
    {
        commandList->open();
    }
}