/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "rtxmg/utils/buffer.h"

nvrhi::BufferDesc GetGenericDesc(size_t nElements, uint32_t elementSize, const char* name, nvrhi::Format format)
{
    nElements = std::max(size_t(1), nElements);
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

    if (!async)
    {
        commandList->getDevice()->unmapBuffer(staging);
        commandList->open();
    }
}