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

#include <cmath>
#include <iomanip>
#include <fstream>

#include "rtxmg/utils/debug.h"

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
