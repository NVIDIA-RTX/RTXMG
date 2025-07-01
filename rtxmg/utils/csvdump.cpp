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

#include "rtxmg/utils/debug.h"

#include <cmath>
#include <iomanip>
#include <fstream>



void WriteTexToCSV(nvrhi::ICommandList* commandList, nvrhi::ITexture* tex, char const filename[])
{
    nvrhi::TextureDesc desc = tex->getDesc();
    nvrhi::StagingTextureHandle staging = commandList->getDevice()->createStagingTexture(desc, nvrhi::CpuAccessMode::Read);

    commandList->copyTexture(staging, nvrhi::TextureSlice(), tex, nvrhi::TextureSlice());
    commandList->close();
    commandList->getDevice()->executeCommandList(commandList);

    size_t rowPitch = 0;
    float const* pData = static_cast<float const*>(commandList->getDevice()->mapStagingTexture(
        staging, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &rowPitch));

    std::ofstream debugDump(filename);

    for (uint32_t y = 0; y < desc.height; y++)
    {
        for (uint32_t x = 0; x < desc.width; x++)
        {
            float z = pData[y * rowPitch / sizeof(float) + x];
            if (std::isinf(z))
                z = -1.0f;
            debugDump << std::setw(8) << std::right << z;
            if (x < desc.width - 1)
                debugDump << ", ";
        }
        debugDump << std::endl;
    }

    commandList->open();
}

void WriteBufferToCSV(nvrhi::ICommandList* commandList, RTXMGBuffer<float>& buf, char const filename[], int width, int height)
{
    auto values = buf.Download(commandList);

    std::ofstream debugDump(filename);

    for (uint32_t i = 0; i < values.size(); i++)
    {
        debugDump << std::setw(8) << std::right << values[i];
        if (i < values.size() - 1)
        {
            if (i % width == width - 1)
                debugDump << std::endl;
            else
                debugDump << ", ";
        }
    }
}
