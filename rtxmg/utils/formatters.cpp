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

#include "rtxmg/utils/formatters.h"

#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <iterator>

int HumanFormatter(double value, char* buff, int bufsize, void*)
{
    static char const* scale[] = { "", "K", "M", "B", "T", "Q", };

    int ndigits = static_cast<int>(value == 0 ? 0 : 1 + std::floor(log10l(std::abs(value))));

    int exp = ndigits <= 4 ? 0 : 3 * ((ndigits - 1) / 3);

    if ((exp / 3) >= std::size(scale))
        return false;

    double n = static_cast<double>(value / powl(10, exp));

    bool decimals = value - n == 0. ? false : true;

    return std::snprintf(buff, bufsize, decimals ? "%.1f%s" : "%.0f%s", n, scale[exp / 3]);
}

int MetricFormatter(double value, char* buff, int bufsize, void* data)
{
    const char* unit = (const char*)data;
    static double v[] = { 1000000000,1000000,1000,1,0.001,0.000001,0.000000001 };
    static const char* p[] = { "G","M","k","","m","u","n" };
    if (value == 0)
    {
        return snprintf(buff, bufsize, "0 %s", unit);
    }
    for (int i = 0; i < 7; ++i)
    {
        if (fabs(value) >= v[i])
        {
            return snprintf(buff, bufsize, "%g %s%s", value / v[i], p[i], unit);
        }
    }
    return snprintf(buff, bufsize, "%g %s%s", value / v[6], p[6], unit);
}

int MegabytesFormatter(double value, char* buff, int bufsize, void*)
{
    double mbsize = value / (1024 * 1024);
    return snprintf(buff, bufsize, "%.1f MB", mbsize);
}

int MemoryFormatter(double value, char* buff, int bufsize, void*)
{
    static char const* suffixes[] = { "B", "KB",  "MB",  "GB", "TB" };

    uint8_t s = 0;
    for (; value >= 1024; ++s)
        value /= 1024;

    assert(s < std::size(suffixes));

    if (value - std::floor(value) == 0.)
        snprintf(buff, bufsize, "%d %s", (int)value, suffixes[s]);
    else
        snprintf(buff, bufsize, "%.1f %s", value, suffixes[s]);
    return 0;
}
