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

#pragma once

#include <cstdint>

namespace rtxmg {

struct SystemMemory
{
    uint64_t totalBytes     = 0;
    uint64_t availableBytes = 0;
};

// Physical RAM of the machine.  Returns false if the OS query fails.
bool QuerySystemMemory(SystemMemory& out);

} // namespace rtxmg
