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

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <donut/core/log.h>

#include "rtxmg/utils/file_mapping.h"

namespace rtxmg {

FileReadMapping::FileReadMapping(FileReadMapping&& o) noexcept
    : m_fileHandle(o.m_fileHandle)
    , m_mappingHandle(o.m_mappingHandle)
    , m_data(o.m_data)
    , m_size(o.m_size)
{
    o.m_fileHandle    = nullptr;
    o.m_mappingHandle = nullptr;
    o.m_data          = nullptr;
    o.m_size          = 0;
}

FileReadMapping& FileReadMapping::operator=(FileReadMapping&& o) noexcept
{
    if (this != &o)
    {
        Close();
        m_fileHandle    = o.m_fileHandle;
        m_mappingHandle = o.m_mappingHandle;
        m_data          = o.m_data;
        m_size          = o.m_size;
        o.m_fileHandle    = nullptr;
        o.m_mappingHandle = nullptr;
        o.m_data          = nullptr;
        o.m_size          = 0;
    }
    return *this;
}

#ifdef _WIN32

bool FileReadMapping::Open(const std::filesystem::path& path)
{
    Close();

    HANDLE hFile = CreateFileW(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);

    if (hFile == INVALID_HANDLE_VALUE)
    {
        donut::log::error("FileReadMapping: failed to open '%s' (error %lu)",
                          path.string().c_str(), GetLastError());
        return false;
    }

    LARGE_INTEGER fileSize{};
    if (!GetFileSizeEx(hFile, &fileSize) || fileSize.QuadPart == 0)
    {
        CloseHandle(hFile);
        return false;
    }

    HANDLE hMapping = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (hMapping == nullptr)
    {
        donut::log::error("FileReadMapping: CreateFileMapping failed for '%s' (error %lu)",
                          path.string().c_str(), GetLastError());
        CloseHandle(hFile);
        return false;
    }

    void* view = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr)
    {
        donut::log::error("FileReadMapping: MapViewOfFile failed for '%s' (error %lu)",
                          path.string().c_str(), GetLastError());
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return false;
    }

    m_fileHandle    = hFile;
    m_mappingHandle = hMapping;
    m_data          = view;
    m_size          = static_cast<size_t>(fileSize.QuadPart);
    return true;
}

void FileReadMapping::Close()
{
    if (m_data)
    {
        UnmapViewOfFile(m_data);
        m_data = nullptr;
    }
    if (m_mappingHandle)
    {
        CloseHandle(m_mappingHandle);
        m_mappingHandle = nullptr;
    }
    if (m_fileHandle)
    {
        CloseHandle(m_fileHandle);
        m_fileHandle = nullptr;
    }
    m_size = 0;
}

#else

// The mapping outlives the descriptor, so no OS handle is kept open.
bool FileReadMapping::Open(const std::filesystem::path& path)
{
    Close();

    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        donut::log::error("FileReadMapping: failed to open '%s' (%s)",
                          path.string().c_str(), strerror(errno));
        return false;
    }

    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size == 0)
    {
        close(fd);
        return false;
    }

    void* view = mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (view == MAP_FAILED)
    {
        donut::log::error("FileReadMapping: mmap failed for '%s' (%s)",
                          path.string().c_str(), strerror(errno));
        return false;
    }

    m_data = view;
    m_size = size_t(st.st_size);
    return true;
}

void FileReadMapping::Close()
{
    if (m_data)
    {
        munmap(m_data, m_size);
        m_data = nullptr;
    }
    m_size = 0;
}

#endif

} // namespace rtxmg
