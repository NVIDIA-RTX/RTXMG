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

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define RTXMG_GETPID _getpid
#else
#include <unistd.h>
#define RTXMG_GETPID getpid
#endif

#include <donut/core/log.h>

#include "rtxmg/cluster_lod/cache.h"
#include "rtxmg/cluster_lod/baked_geometry.h"
#include "rtxmg/cluster_lod/serialization.h"

using namespace donut;

// ---------------------------------------------------------------------------
// Internal: aligned FILE write helpers
// ---------------------------------------------------------------------------

namespace {

static const uint8_t s_pad[serialization::ALIGNMENT] = {};

// Write `dataSize` bytes to file with trailing alignment padding.
// Adds written bytes (aligned) to `accumulated`.
bool FileWriteAligned(uint64_t& accumulated, FILE* f, size_t dataSize, const void* data)
{
    assert(accumulated % serialization::ALIGNMENT == 0);

    if (fwrite(data, dataSize, 1, f) != 1)
        return false;

    uint64_t aligned   = (dataSize + serialization::ALIGN_MASK) & ~serialization::ALIGN_MASK;
    uint64_t padBytes  = aligned - dataSize;
    if (padBytes && fwrite(s_pad, padBytes, 1, f) != 1)
        return false;

    accumulated += aligned;
    return true;
}

// Write a span: 16-byte count header + data (aligned).
template<typename T>
void FileWriteSpan(bool& ok, uint64_t& accumulated, FILE* f, const std::span<const T>& view)
{
    assert(accumulated % serialization::ALIGNMENT == 0);
    if (!ok) return;

    union { uint64_t count; uint8_t countData[serialization::ALIGNMENT]; };
    memset(countData, 0, serialization::ALIGNMENT);
    count = view.size();

    if (fwrite(countData, serialization::ALIGNMENT, 1, f) != 1) { ok = false; return; }
    accumulated += serialization::ALIGNMENT;

    if (view.size() && !FileWriteAligned(accumulated, f, view.size_bytes(), view.data()))
        ok = false;
}

} // namespace

// ---------------------------------------------------------------------------
// StoreCachedGeometry — serialise one view into a pre-allocated buffer
// ---------------------------------------------------------------------------

bool StoreCachedGeometry(const GeometryView& view, uint64_t dataSize, void* data)
{
    uint64_t addr    = reinterpret_cast<uint64_t>(data);
    uint64_t addrEnd = addr + dataSize;

    bool ok = (addr % serialization::ALIGNMENT == 0)
           && (addr + sizeof(GeometryBase) <= addrEnd);

    if (ok)
    {
        memcpy(reinterpret_cast<void*>(addr),
               static_cast<const GeometryBase*>(&view), sizeof(GeometryBase));
        addr += (sizeof(GeometryBase) + serialization::ALIGN_MASK) & ~serialization::ALIGN_MASK;
    }

    if (ok)
    {
        serialization::StoreAndAdvance(ok, addr, addrEnd, view.groupData);
        serialization::StoreAndAdvance(ok, addr, addrEnd, view.groupInfos);
        serialization::StoreAndAdvance(ok, addr, addrEnd, view.lodLevels);
        serialization::StoreAndAdvance(ok, addr, addrEnd, view.lodStats);
        serialization::StoreAndAdvance(ok, addr, addrEnd, view.lodNodes);
        serialization::StoreAndAdvance(ok, addr, addrEnd, view.lodNodeBboxes);
        serialization::StoreAndAdvance(ok, addr, addrEnd, view.localMaterialIDs);
        serialization::StoreAndAdvance(ok, addr, addrEnd, view.localMaterialStateBits);
    }

    return ok;
}

// ---------------------------------------------------------------------------
// LoadCachedGeometry — deserialise one view from raw bytes (zero-copy)
// ---------------------------------------------------------------------------

bool LoadCachedGeometry(GeometryView& view, uint64_t dataSize, const void* data)
{
    uint64_t addr    = reinterpret_cast<uint64_t>(data);
    uint64_t addrEnd = addr + dataSize;

    if (addr % serialization::ALIGNMENT != 0 || addr + sizeof(GeometryBase) > addrEnd)
    {
        view = {};
        return false;
    }

    memcpy(static_cast<GeometryBase*>(&view), data, sizeof(GeometryBase));
    addr += (sizeof(GeometryBase) + serialization::ALIGN_MASK) & ~serialization::ALIGN_MASK;

    bool ok = true;
    serialization::LoadAndAdvance(ok, addr, addrEnd, view.groupData);
    serialization::LoadAndAdvance(ok, addr, addrEnd, view.groupInfos);
    serialization::LoadAndAdvance(ok, addr, addrEnd, view.lodLevels);
    serialization::LoadAndAdvance(ok, addr, addrEnd, view.lodStats);
    serialization::LoadAndAdvance(ok, addr, addrEnd, view.lodNodes);
    serialization::LoadAndAdvance(ok, addr, addrEnd, view.lodNodeBboxes);
    serialization::LoadAndAdvance(ok, addr, addrEnd, view.localMaterialIDs);
    serialization::LoadAndAdvance(ok, addr, addrEnd, view.localMaterialStateBits);

    return ok;
}

// ---------------------------------------------------------------------------
// CacheFileView::Init — from mapped pointer
// ---------------------------------------------------------------------------

bool CacheFileView::Init(uint64_t dataSize, const void* data)
{
    m_dataSize  = 0;
    m_dataBytes = reinterpret_cast<const uint8_t*>(data);

    const uint64_t minSize = sizeof(CacheFileHeader) + sizeof(uint64_t);
    if (dataSize <= minSize)
        return false;

    const auto* hdr = reinterpret_cast<const CacheFileHeader*>(data);
    if (!hdr->IsValid())
    {
        log::warning("CacheFileView: invalid or mismatched cache header");
        return false;
    }

    m_geometryCount = *reinterpret_cast<const uint64_t*>(m_dataBytes + dataSize - sizeof(uint64_t));

    if (m_geometryCount == 0
        || dataSize <= sizeof(CacheFileHeader) + sizeof(uint64_t) * (m_geometryCount * 2 + 1))
    {
        return false;
    }

    m_tableStart = dataSize - sizeof(uint64_t) * (m_geometryCount * 2 + 1);
    m_dataSize   = dataSize;
    return true;
}

// ---------------------------------------------------------------------------
// CacheFileView::Init — open and memory-map a file
// ---------------------------------------------------------------------------

bool CacheFileView::Init(const std::filesystem::path& path)
{
    Deinit();
    if (!m_mapping.Open(path))
        return false;
    if (!Init(m_mapping.Size(), m_mapping.Data()))
    {
        m_mapping.Close();
        return false;
    }
    return true;
}

void CacheFileView::Deinit()
{
    m_mapping.Close();
    m_dataSize      = 0;
    m_tableStart    = 0;
    m_dataBytes     = nullptr;
    m_geometryCount = 0;
}

// ---------------------------------------------------------------------------
// CacheFileView::GetConfigHash
// ---------------------------------------------------------------------------

uint64_t CacheFileView::GetConfigHash() const
{
    assert(IsValid());
    return reinterpret_cast<const CacheFileHeader*>(m_dataBytes)->configHash;
}

// ---------------------------------------------------------------------------
// CacheFileView::GetGeometryView — zero-copy
// ---------------------------------------------------------------------------

bool CacheFileView::GetGeometryView(GeometryView& view, uint64_t idx) const
{
    if (idx >= m_geometryCount)
    {
        assert(false);
        return false;
    }

    const auto* table  = Ptr<uint64_t>(m_tableStart, m_geometryCount * 2);
    uint64_t    base   = table[idx * 2 + 0];
    uint64_t    geoSz  = table[idx * 2 + 1];

    if (base + sizeof(GeometryBase) > m_tableStart)
    {
        assert(false);
        return false;
    }

    const void* geoData = Ptr<uint8_t>(base, geoSz);
    return LoadCachedGeometry(view, geoSz, geoData);
}

// ---------------------------------------------------------------------------
// SaveCache — write a complete .nvsngeo file
// ---------------------------------------------------------------------------

bool SaveCache(const std::filesystem::path&     path,
               const std::vector<GeometryView>& geometries,
               const BakerConfig&               config)
{
    const uint64_t geomCount = geometries.size();

    // Compute per-geometry offsets.
    std::vector<uint64_t> offsetTable;
    offsetTable.reserve(geomCount * 2 + 1);

    uint64_t offset = sizeof(CacheFileHeader);
    for (const GeometryView& g : geometries)
    {
        uint64_t sz = g.GetCachedSize();
        offsetTable.push_back(offset);
        offsetTable.push_back(sz);
        offset += sz;
    }
    offsetTable.push_back(geomCount);

    // Total file size.
    const uint64_t tableBytes  = offsetTable.size() * sizeof(uint64_t);
    const uint64_t totalBytes  = offset + tableBytes;

    // Build the header.
    CacheFileHeader hdr;
    hdr.configHash = config.SemanticHash();

    // Allocate file data in memory, write header + geometry blobs + table.
    std::vector<uint8_t> fileData(totalBytes, 0);
    uint8_t* base = fileData.data();

    memcpy(base, &hdr, sizeof(hdr));

    bool ok = true;
    for (uint64_t i = 0; i < geomCount; i++)
    {
        uint64_t geoOffset = offsetTable[i * 2 + 0];
        uint64_t geoSize   = offsetTable[i * 2 + 1];
        if (!StoreCachedGeometry(geometries[i], geoSize, base + geoOffset))
        {
            log::error("SaveCache: failed to serialise geometry %llu", i);
            ok = false;
        }
    }

    memcpy(base + offset, offsetTable.data(), tableBytes);

    if (!ok)
        return false;

    // Write to disk.
    FILE* f = fopen(path.string().c_str(), "wb");
    if (!f)
    {
        log::error("SaveCache: cannot open '%s' for writing", path.string().c_str());
        return false;
    }

    bool wrote = (fwrite(fileData.data(), fileData.size(), 1, f) == 1);
    fclose(f);

    if (!wrote)
    {
        log::error("SaveCache: write failed for '%s'", path.string().c_str());
        return false;
    }

    log::info("SaveCache: wrote %llu geometries to '%s' (%.1f MB)",
              geomCount,
              path.string().c_str(),
              double(totalBytes) / (1024.0 * 1024.0));
    return true;
}

// ---------------------------------------------------------------------------
// SaveShardAtomic — write one geometry to its own shard file (temp + rename)
// ---------------------------------------------------------------------------

bool SaveShardAtomic(const std::filesystem::path& path,
                     const GeometryView&          view,
                     const BakerConfig&           config,
                     uint64_t                     sourceHash,
                     uint64_t                     sourceSize,
                     int64_t                      sourceMtime)
{
    namespace fs = std::filesystem;

    const uint64_t blobSize = view.GetCachedSize();
    const uint64_t total    = sizeof(ShardHeader) + blobSize;

    ShardHeader hdr;
    hdr.configHash  = config.SemanticHash();
    hdr.sourceHash  = sourceHash;
    hdr.sourceSize  = sourceSize;
    hdr.sourceMtime = sourceMtime;
    hdr.blobSize    = blobSize;

    // Build the file image in RAM, then serialise the blob 16-aligned after the
    // header (vector data is max-align'd and sizeof(ShardHeader) % 16 == 0).
    std::vector<uint8_t> fileData(total, 0);
    memcpy(fileData.data(), &hdr, sizeof(hdr));
    if (!StoreCachedGeometry(view, blobSize, fileData.data() + sizeof(ShardHeader)))
    {
        log::error("SaveShardAtomic: failed to serialise geometry for '%s'",
                   path.string().c_str());
        return false;
    }

    // Atomic publish: unique temp sibling, then rename into place.  Only
    // *.shard files are ever treated as valid, so a crash mid-write leaves a
    // stray *.tmp that is ignored / overwritten.  The name carries the process
    // id as well as a counter: the cache dir is designed to be shared across
    // runs, and a bare counter restarts at 0 in every process, so two
    // concurrent bakes of the same content hash would write the same temp file.
    static std::atomic<uint64_t> s_tmpCounter{ 0 };
    fs::path tmp = path;
    tmp += ".tmp" + std::to_string(RTXMG_GETPID()) + "_" +
           std::to_string(s_tmpCounter.fetch_add(1, std::memory_order_relaxed));

    FILE* f = fopen(tmp.string().c_str(), "wb");
    if (!f)
    {
        log::error("SaveShardAtomic: cannot open '%s' for writing", tmp.string().c_str());
        return false;
    }
    bool wrote = (fwrite(fileData.data(), fileData.size(), 1, f) == 1);
    fclose(f);

    std::error_code ec;
    if (!wrote)
    {
        fs::remove(tmp, ec);
        log::error("SaveShardAtomic: write failed for '%s'", tmp.string().c_str());
        return false;
    }

    fs::rename(tmp, path, ec);
    if (ec)
    {
        // Replace an existing destination if the platform's rename won't.
        fs::remove(path, ec);
        fs::rename(tmp, path, ec);
        if (ec)
        {
            std::error_code ec2;
            fs::remove(tmp, ec2);
            log::error("SaveShardAtomic: rename to '%s' failed: %s",
                       path.string().c_str(), ec.message().c_str());
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// ShardCache
// ---------------------------------------------------------------------------

bool ShardCache::ReadHeader(const std::filesystem::path& path, ShardHeader& outHeader)
{
    rtxmg::FileReadMapping m;
    if (!m.Open(path) || m.Size() < sizeof(ShardHeader))
        return false;
    memcpy(&outHeader, m.Data(), sizeof(ShardHeader));
    return outHeader.IsValidFormat();
}

bool ShardCache::LoadGeometryView(const std::filesystem::path& path,
                                  const BakerConfig&           config,
                                  GeometryView&                outView)
{
    std::string key = path.string();

    size_t idx;
    auto   it = m_pathToIndex.find(key);
    if (it != m_pathToIndex.end())
    {
        idx = it->second;
    }
    else
    {
        rtxmg::FileReadMapping mapping;
        if (!mapping.Open(path))
            return false;
        idx = m_mappings.size();
        m_mappings.push_back(std::move(mapping));
        m_pathToIndex.emplace(std::move(key), idx);
    }

    // Note: a later push_back may move the FileReadMapping objects, but the
    // mapped OS address (which spans point into) is preserved across the move,
    // so views handed out by earlier calls stay valid.
    const rtxmg::FileReadMapping& mapping = m_mappings[idx];
    if (mapping.Size() < sizeof(ShardHeader))
        return false;

    const auto* hdr = reinterpret_cast<const ShardHeader*>(mapping.Data());
    if (!hdr->IsValidFormat())
        return false;
    if (hdr->configHash != config.SemanticHash())
        return false;
    if (sizeof(ShardHeader) + hdr->blobSize > mapping.Size())
        return false;

    const void* blob =
        reinterpret_cast<const uint8_t*>(mapping.Data()) + sizeof(ShardHeader);
    if (!LoadCachedGeometry(outView, hdr->blobSize, blob))
        return false;

    // Consumers index lodLevels[lodLevelsCount - 1], which underflows to
    // 0xFFFFFFFF at 0, so a shard with no hierarchy is not loadable.
    if (outView.lodLevelsCount == 0 || outView.lodLevelsCount > outView.lodLevels.size())
    {
        outView = {};
        return false;
    }
    return true;
}
