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

#include <nvrhi/nvrhi.h>

#include <cstdint>

namespace rtxmg {

// Steps the driver's usage counter grows by, measured on NVIDIA under D3D12 and VK;
// D3D12's 64 KB buffer alignment is address space only, VK images are 64 KB paged.
inline constexpr uint64_t kBufferVramGranularity  = 4ull << 10;
inline constexpr uint64_t kTextureVramGranularity = 64ull << 10;

// Device-local bytes a buffer occupies.  0 for CPU-visible (upload/readback)
// and volatile buffers, which live in system memory, and for virtual buffers,
// whose heap is the owner's to count.
inline uint64_t BufferVramBytes(const nvrhi::IBuffer* buffer)
{
    if (!buffer)
        return 0;
    const nvrhi::BufferDesc& d = buffer->getDesc();
    if (d.cpuAccess != nvrhi::CpuAccessMode::None || d.isVolatile || d.isVirtual)
        return 0;
    return (d.byteSize + kBufferVramGranularity - 1) & ~(kBufferVramGranularity - 1);
}

// Device-local bytes a committed texture occupies.  The size depends only on the
// desc, so the device is queried once per distinct desc and cached; safe to call
// per frame and from any thread.
uint64_t TextureVramBytes(nvrhi::IDevice* device, nvrhi::ITexture* texture);

}  // namespace rtxmg
