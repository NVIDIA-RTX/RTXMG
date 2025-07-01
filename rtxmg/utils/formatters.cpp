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

#include "rtxmg/utils/formatters.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cmath>

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
