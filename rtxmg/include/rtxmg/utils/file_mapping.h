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

// Read-only memory-mapped file wrapper.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace rtxmg {

// Maps a file read-only into the process address space; data() stays valid
// until close().
class FileReadMapping
{
public:
    FileReadMapping()  = default;
    ~FileReadMapping() { Close(); }

    // Non-copyable, movable.
    FileReadMapping(const FileReadMapping&)            = delete;
    FileReadMapping& operator=(const FileReadMapping&) = delete;
    FileReadMapping(FileReadMapping&& o) noexcept;
    FileReadMapping& operator=(FileReadMapping&& o) noexcept;

    // Map the file at `path` read-only.  Returns true on success.
    bool Open(const std::filesystem::path& path);

    // Unmap and release all OS handles.
    void Close();

    bool        Valid() const { return m_data != nullptr; }
    const void* Data()  const { return m_data; }
    size_t      Size()  const { return m_size; }

private:
    void*   m_fileHandle    = nullptr;   // HANDLE (WIN32)
    void*   m_mappingHandle = nullptr;   // HANDLE (WIN32)
    void*   m_data          = nullptr;   // mapped view base
    size_t  m_size          = 0;
};

} // namespace rtxmg
