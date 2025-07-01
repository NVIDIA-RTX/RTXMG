//
// Copyright (c) 2012-2016, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>

#include <donut/core/math/math.h>

using namespace donut::math;

// some 'fast' string parsing helpers : use safer std functions
// instead wherever possible

inline bool IsWhiteSpace(char c)
{
    return ((c == ' ') || (c == '\t') || (c == '\r'));
}

inline bool IsNewLine(char c) { return c == '\n'; }

inline bool IsDigit(char c) { return ((c >= '0') && (c <= '9')); }

inline bool IsExponent(char c) { return ((c == 'e') || (c == 'E')); }

inline char const* SkipWhiteSpace(char const* ptr)
{
    while (IsWhiteSpace(*ptr))
        ++ptr;
    return ptr;
}

inline char const* SkipLine(char const* ptr)
{
    while (!IsNewLine(*ptr++))
        ;
    return ptr;
}

inline char const* SkipWord(char const* ptr)
{
    while ((!IsWhiteSpace(*ptr)) && (*ptr != '\n') && (*ptr != '\0'))
        ++ptr;
    return ptr;
}

inline char const* SkipWords(char const* ptr, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        ptr = SkipWord(ptr);
        ptr = SkipWhiteSpace(ptr);
    }
    return ptr;
}

inline char const* FindSubstring(char const* str, char const* substr,
    uint32_t num)
{
    while ((*str != '\0') && (num >= 0))
    {
        if (!std::memcmp(str, substr, num))
            return str;
        ++str;
    }
    return nullptr;
}

char const* ParseInt(char const* ptr, int* value);

char const* ParseDouble(char const* ptr, double* value);
char const* ParseFloat(char const* ptr, float* value);

char const* ParseString(char const* ptr, std::string* value);

std::string ReadASCIIFile(char const* m_filepath);
char const* sgets(char* s, int size, char** stream);


std::unique_ptr<uint8_t[]> ReadBigFile(std::filesystem::path const& m_filepath,
    uint64_t* size = nullptr);

std::istream& operator>>(std::istream& is, float3& v);
std::ostream& operator<<(std::ostream& os, float3& v);
std::ostream& operator<<(std::ostream& os, box3& v);
