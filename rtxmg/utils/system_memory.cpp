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

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cstdio>
#endif

#include "rtxmg/utils/system_memory.h"

namespace rtxmg {

bool QuerySystemMemory(SystemMemory& out)
{
#ifdef _WIN32
    MEMORYSTATUSEX status = {};
    status.dwLength       = sizeof(status);
    if (!GlobalMemoryStatusEx(&status))
        return false;
    out.totalBytes     = status.ullTotalPhys;
    out.availableBytes = status.ullAvailPhys;
    return true;
#else
    // MemAvailable counts reclaimable page cache, matching ullAvailPhys.
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f)
        return false;
    unsigned long long totalKiB = 0, availKiB = 0;
    char line[256];
    while (fgets(line, sizeof(line), f))
    {
        sscanf(line, "MemTotal: %llu kB", &totalKiB);
        sscanf(line, "MemAvailable: %llu kB", &availKiB);
    }
    fclose(f);
    if (totalKiB == 0)
        return false;
    out.totalBytes     = uint64_t(totalKiB) << 10;
    out.availableBytes = uint64_t(availKiB) << 10;
    return true;
#endif
}

} // namespace rtxmg
