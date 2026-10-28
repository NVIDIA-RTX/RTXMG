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

#include "rtxmg/utils/vram_bytes.h"
#include "rtxmg/utils/texture_bytes.h"

#include <mutex>
#include <unordered_map>

namespace rtxmg {

namespace {

struct TextureShape
{
    nvrhi::TextureDimension dimension;
    nvrhi::Format           format;
    uint32_t                width, height, depth, arraySize, mipLevels, sampleCount;
    bool                    isRenderTarget, isUAV, isTypeless;

    bool operator==(const TextureShape&) const = default;
};

struct TextureShapeHash
{
    size_t operator()(const TextureShape& s) const
    {
        uint64_t h = 1469598103934665603ull;
        auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
        mix(uint64_t(s.dimension));
        mix(uint64_t(s.format));
        mix(s.width);
        mix(s.height);
        mix(s.depth);
        mix(s.arraySize);
        mix(s.mipLevels);
        mix(s.sampleCount);
        mix(uint64_t(s.isRenderTarget) | uint64_t(s.isUAV) << 1 | uint64_t(s.isTypeless) << 2);
        return size_t(h);
    }
};

}  // namespace

uint64_t TextureVramBytes(nvrhi::IDevice* device, nvrhi::ITexture* texture)
{
    if (!texture)
        return 0;

    const nvrhi::TextureDesc& d = texture->getDesc();
    const TextureShape shape = { d.dimension, d.format, d.width, d.height, d.depth, d.arraySize,
                                 d.mipLevels, d.sampleCount, d.isRenderTarget, d.isUAV, d.isTypeless };

    static std::mutex                                                    s_mutex;
    static std::unordered_map<TextureShape, uint64_t, TextureShapeHash> s_bytes;

    std::lock_guard<std::mutex> lock(s_mutex);
    if (auto it = s_bytes.find(shape); it != s_bytes.end())
        return it->second;

    const nvrhi::MemoryRequirements req = device ? device->getTextureMemoryRequirements(texture)
                                                 : nvrhi::MemoryRequirements{};
    const uint64_t size  = req.size ? req.size : TextureGpuBytes(d);
    const uint64_t bytes = (size + kTextureVramGranularity - 1) & ~(kTextureVramGranularity - 1);
    s_bytes.emplace(shape, bytes);
    return bytes;
}

}  // namespace rtxmg
