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

#include <algorithm>
#include <bit>
#include <cstring>
#include <cstdio>
#include <string>

#include <donut/core/log.h>

#include "rtxmg/cluster_lod/streaming.h"

// --debug-clusterlod-only logging and readbacks for ClusterLodStreaming.
// DebugDumpAllocatorFreelist and DebugReadbackMoveArgs issue blocking
// Download()s from inside an open command list, so both stall the frame.

namespace rtxmg {

bool ClusterLodStreaming::DebugClusterLodLoggingEnabled() const
{
    return m_config.debugClusterLod;
}

void ClusterLodStreaming::LogDebugRendererBeginFrame() const
{
    if (!DebugClusterLodLoggingEnabled())
        return;

    donut::log::info("streaming: renderer begin frame %u", m_frameIndex);
}

void ClusterLodStreaming::LogDebugUpdateApply(uint32_t taskIndex) const
{
    if (!DebugClusterLodLoggingEnabled())
        return;

    const shaderio::StreamingUpdate& update = m_shaderData.update;
    if (!update.patchGroupsCount && !update.patchCachedBlasCount)
        return;

    const uint32_t updateLoadCount =
        update.patchGroupsCount - update.patchUnloadGroupsCount;
    donut::log::info("ClusterLodStreaming update-apply frame=%u task=%u patchLoad=%u patchUnload=%u patchGroups=%u newClas=%u",
                     m_frameIndex,
                     taskIndex,
                     updateLoadCount,
                     update.patchUnloadGroupsCount,
                     update.patchGroupsCount,
                     update.newClasCount);
}

void ClusterLodStreaming::LogDebugResidentActive() const
{
    if (!DebugClusterLodLoggingEnabled())
        return;

    std::vector<GeometryGroup> activeGroups;
    m_resident.CollectActiveGeometryGroups(activeGroups);
    std::sort(activeGroups.begin(), activeGroups.end(),
              [](GeometryGroup a, GeometryGroup b) { return a.key < b.key; });

    std::string geometry0Ids;
    geometry0Ids.reserve(activeGroups.size() * 4);
    uint32_t geometry0Count = 0;
    for (const GeometryGroup group : activeGroups)
    {
        if (group.geometryID != 0)
            continue;

        if (!geometry0Ids.empty())
            geometry0Ids += ',';

        geometry0Ids += std::to_string(group.groupID);
        ++geometry0Count;
    }

    donut::log::info("ClusterLodStreaming resident-active frame=%u count=%zu geometry0Count=%u geometry0Ids=[%s]",
                     m_frameIndex,
                     activeGroups.size(),
                     geometry0Count,
                     geometry0Ids.c_str());
}

void ClusterLodStreaming::LogDebugRequestReadback(uint64_t requestFrame,
                                                  uint32_t taskIndex,
                                                  uint32_t loadCount,
                                                  uint32_t unloadCount,
                                                  const StreamingRequests::TaskInfo& request) const
{
    if (!DebugClusterLodLoggingEnabled())
        return;

    donut::log::info("ClusterLodStreaming request frame=%llu task=%u rawLoad=%u rawUnload=%u consumedLoad=%u consumedUnload=%u maxLoad=%u maxUnload=%u",
                     (unsigned long long)requestFrame,
                     taskIndex,
                     request.shaderData->loadCounter,
                     request.shaderData->unloadCounter,
                     loadCount,
                     unloadCount,
                     request.shaderData->maxLoads,
                     request.shaderData->maxUnloads);

    if (loadCount == 0)
        return;

    char preview[512] = {};
    size_t cursor = 0;
    const uint32_t previewCount = std::min(loadCount, 8u);
    for (uint32_t i = 0; i < previewCount && cursor < sizeof(preview); ++i)
    {
        const GeometryGroup group = request.loadGeometryGroups[i];
        const int written = std::snprintf(preview + cursor, sizeof(preview) - cursor,
                                          "%s{%u,%u}", i ? " " : "",
                                          group.geometryID, group.groupID);
        if (written <= 0)
            break;
        cursor += size_t(written);
    }
    donut::log::info("ClusterLodStreaming request frame=%llu firstLoadGroups=[%s]%s",
                     (unsigned long long)requestFrame,
                     preview,
                     loadCount > previewCount ? " ..." : "");
}

void ClusterLodStreaming::DebugDumpAllocatorFreelist(nvrhi::ICommandList* commandList,
                                                     const shaderio::StreamingUpdate& update)
{
    if (!DebugClusterLodLoggingEnabled())
        return;

    const std::vector<uint8_t> mgmtData =
        m_clasAllocator.GetManagementBufferTyped().Download(commandList);
    const std::vector<shaderio::SceneStreaming> shaderData =
        m_shaderBuffer.Download(commandList);

    const uint8_t* mgmt = mgmtData.empty() ? nullptr : mgmtData.data();
    const shaderio::StreamingAllocator* hdr =
        shaderData.empty() ? nullptr : &shaderData[0].clasAllocator;

    if (mgmt && hdr)
    {
        const shaderio::StreamingAllocator& a = m_shaderData.clasAllocator;
        // Bounds-guarded loads: corrupt GPU-written freelist offsets/counts
        // would drive these reads past the downloaded buffer, so the dump
        // reports the corruption instead of crashing on it.
        const size_t mgmtSize = mgmtData.size();
        uint32_t oobCount = 0;
        uint32_t oobFirstOffset = 0xFFFFFFFFu;
        auto inBounds = [&](uint32_t byteOffset, uint32_t width) -> bool {
            if (size_t(byteOffset) + width <= mgmtSize)
                return true;
            if (oobFirstOffset == 0xFFFFFFFFu)
                oobFirstOffset = byteOffset;
            ++oobCount;
            return false;
        };
        auto loadU32 = [&](uint32_t byteOffset) -> uint32_t {
            uint32_t v = 0;
            if (inBounds(byteOffset, sizeof(v)))
                std::memcpy(&v, mgmt + byteOffset, sizeof(v));
            return v;
        };
        auto loadI32 = [&](uint32_t byteOffset) -> int32_t {
            int32_t v = 0;
            if (inBounds(byteOffset, sizeof(v)))
                std::memcpy(&v, mgmt + byteOffset, sizeof(v));
            return v;
        };
        auto loadU16 = [&](uint32_t byteOffset) -> uint32_t {
            uint16_t v = 0;
            if (inBounds(byteOffset, sizeof(v)))
                std::memcpy(&v, mgmt + byteOffset, sizeof(v));
            return uint32_t(v);
        };
        const uint32_t usedWordCount = a.sectorCount << a.sectorSizeShift;
        uint32_t nonzeroUsedWords = 0;
        uint32_t usedGranules     = 0;
        uint32_t firstUsedWord    = 0xFFFFFFFFu;
        for (uint32_t i = 0; i < usedWordCount; ++i)
        {
            const uint32_t bits = loadU32(a.usedBitsByteOffset + i * 4u);
            if (bits)
            {
                if (firstUsedWord == 0xFFFFFFFFu)
                    firstUsedWord = i;
                ++nonzeroUsedWords;
                usedGranules += uint32_t(std::popcount(bits));
            }
        }

        const uint32_t sectorWords = (a.sectorCount + 31u) / 32u;
        std::string usedSectorWords;
        for (uint32_t i = 0; i < std::min(sectorWords, 8u); ++i)
        {
            char tmp[24];
            std::snprintf(tmp, sizeof(tmp), "%s%08x", i ? "," : "",
                          loadU32(a.usedSectorBitsByteOffset + i * 4u));
            usedSectorWords += tmp;
        }

        const uint32_t maxBin = a.maxAllocationSize ? a.maxAllocationSize - 1u : 0u;
        const int32_t maxBinCount =
            loadI32(a.freeSizeRangesByteOffset + maxBin * sizeof(shaderio::AllocatorRange) + 0u);
        const uint32_t maxBinOffset =
            loadU32(a.freeSizeRangesByteOffset + maxBin * sizeof(shaderio::AllocatorRange) + 4u);
        const uint32_t maxBinEntryCount =
            maxBinCount > 0 ? uint32_t(maxBinCount) : 0u;

        std::string nonzeroBins;
        uint32_t nonzeroBinCount = 0;
        for (uint32_t s = 0; s < a.maxAllocationSize; ++s)
        {
            const int32_t count = loadI32(a.freeSizeRangesByteOffset
                                          + s * sizeof(shaderio::AllocatorRange) + 0u);
            if (!count)
                continue;
            const uint32_t offset = loadU32(a.freeSizeRangesByteOffset
                                            + s * sizeof(shaderio::AllocatorRange) + 4u);
            ++nonzeroBinCount;
            if (nonzeroBinCount <= 16u)
            {
                char tmp[48];
                std::snprintf(tmp, sizeof(tmp), "%s%u:%d@%u",
                              nonzeroBins.empty() ? "" : " ",
                              s + 1u, count, offset);
                nonzeroBins += tmp;
            }
        }

        const uint32_t rawGapCount = std::min(hdr->freeGapsCounter, usedWordCount);
        std::string rawGaps;
        for (uint32_t i = 0; i < std::min(rawGapCount, 16u); ++i)
        {
            const uint32_t pos  = loadU32(a.freeGapsPosByteOffset + i * 4u);
            const uint32_t size = loadU16(a.freeGapsSizeByteOffset + i * 2u);
            char tmp[32];
            std::snprintf(tmp, sizeof(tmp), "%s%u:%u", rawGaps.empty() ? "" : " ", pos, size);
            rawGaps += tmp;
        }

        std::string maxBinEntries;
        for (uint32_t i = 0; i < std::min(maxBinEntryCount, 16u); ++i)
        {
            const uint32_t pos = loadU32(a.freeGapsPosBinnedByteOffset + (maxBinOffset + i) * 4u);
            char tmp[24];
            std::snprintf(tmp, sizeof(tmp), "%s%u", maxBinEntries.empty() ? "" : " ", pos);
            maxBinEntries += tmp;
        }

        donut::log::warning("ClusterLodStreaming freelist frame=%u task=%u freeGapsCounter=%u maxAlloc=%u granularity=%u sectorCount=%u sectorMaxSized=%u usedWords=%u/%u usedGranules=%u firstUsedWord=%u usedSectorWords=[%s]",
                            m_frameIndex,
                            update.taskIndex,
                            hdr->freeGapsCounter,
                            a.maxAllocationSize,
                            1u << a.granularityByteShift,
                            a.sectorCount,
                            a.sectorMaxAllocationSized,
                            nonzeroUsedWords,
                            usedWordCount,
                            usedGranules,
                            firstUsedWord,
                            usedSectorWords.c_str());
        donut::log::warning("ClusterLodStreaming freelist bins nonzero=%u first=[%s]%s",
                            nonzeroBinCount,
                            nonzeroBins.c_str(),
                            nonzeroBinCount > 16u ? " ..." : "");
        donut::log::warning("ClusterLodStreaming freelist raw[0..%u]=[%s]%s",
                            std::min(rawGapCount, 16u),
                            rawGaps.c_str(),
                            rawGapCount > 16u ? " ..." : "");
        donut::log::warning("ClusterLodStreaming freelist maxBin size=%u count=%d offset=%u entries[0..%u]=[%s]%s",
                            a.maxAllocationSize,
                            maxBinCount,
                            maxBinOffset,
                            std::min(maxBinEntryCount, 16u),
                            maxBinEntries.c_str(),
                            maxBinEntryCount > 16u ? " ..." : "");
        if (oobCount)
            donut::log::error("ClusterLodStreaming freelist CORRUPT frame=%u task=%u: %u out-of-bounds mgmt reads (mgmtSize=%zu, firstBadOffset=%u); GPU-written freelist offset/count is garbage",
                              m_frameIndex, update.taskIndex, oobCount, mgmtSize, oobFirstOffset);
    }
    else
    {
        donut::log::warning("ClusterLodStreaming freelist frame=%u: failed to map allocator readback",
                            m_frameIndex);
    }

}

void ClusterLodStreaming::DebugReadbackMoveArgs(nvrhi::ICommandList* commandList,
                                                RTXMGBuffer<uint64_t>& moveSrcBuffer,
                                                RTXMGBuffer<uint64_t>& moveDstBuffer,
                                                const shaderio::StreamingUpdate& update)
{
    if (!DebugClusterLodLoggingEnabled() || !moveSrcBuffer.GetBuffer() ||
        !moveDstBuffer.GetBuffer() || !update.newClasCount)
        return;

    const std::vector<uint64_t> src = moveSrcBuffer.Download(commandList);
    const std::vector<uint64_t> dst = moveDstBuffer.Download(commandList);
    const uint32_t inspectedCount = std::min<uint32_t>(
        update.newClasCount,
        std::min<uint32_t>(uint32_t(src.size()), uint32_t(dst.size())));

    if (inspectedCount < update.newClasCount)
    {
        donut::log::warning("ClusterLodStreaming move-args frame=%u: requested %u entries but read src=%zu dst=%zu",
                            m_frameIndex,
                            update.newClasCount,
                            src.size(),
                            dst.size());
    }

    const uint64_t srcBase = m_clasScratchBuffer->getGpuVirtualAddress();
    const uint64_t srcEnd  = srcBase + m_clasScratchBuffer->getDesc().byteSize;
    const uint64_t dstBase = m_shaderData.resident.clasBaseAddress;
    const uint64_t dstEnd  = dstBase + m_shaderData.resident.clasMaxSize;

    uint32_t srcZero       = 0;
    uint32_t dstZero       = 0;
    uint32_t srcOutOfRange = 0;
    uint32_t dstOutOfRange = 0;
    uint32_t badLogged     = 0;

    for (uint32_t i = 0; i < inspectedCount; ++i)
    {
        const uint64_t s = src[i];
        const uint64_t d = dst[i];
        const bool srcBad = (s == 0) || (s < srcBase) || (s >= srcEnd);
        const bool dstBad = (d == 0) || (d < dstBase) || (d >= dstEnd);

        srcZero       += s == 0;
        dstZero       += d == 0;
        srcOutOfRange += (s != 0) && ((s < srcBase) || (s >= srcEnd));
        dstOutOfRange += (d != 0) && ((d < dstBase) || (d >= dstEnd));

        if ((srcBad || dstBad) && badLogged < 8)
        {
            donut::log::warning("ClusterLodStreaming move-args BAD frame=%u idx=%u src=0x%016llx dst=0x%016llx srcBad=%u dstBad=%u",
                                m_frameIndex,
                                i,
                                (unsigned long long)s,
                                (unsigned long long)d,
                                srcBad ? 1u : 0u,
                                dstBad ? 1u : 0u);
            ++badLogged;
        }
    }

    donut::log::info("ClusterLodStreaming move-args frame=%u task=%u newClas=%u inspected=%u src=%s srcRange=[0x%016llx,0x%016llx) dstRange=[0x%016llx,0x%016llx) srcZero=%u dstZero=%u srcOOR=%u dstOOR=%u",
                     m_frameIndex,
                     update.taskIndex,
                     update.newClasCount,
                     inspectedCount,
                     "moveClasSrcAddresses",
                     (unsigned long long)srcBase,
                     (unsigned long long)srcEnd,
                     (unsigned long long)dstBase,
                     (unsigned long long)dstEnd,
                     srcZero,
                     dstZero,
                     srcOutOfRange,
                     dstOutOfRange);

    const uint32_t sampleCount = std::min(inspectedCount, 16u);
    for (uint32_t i = 0; i < sampleCount; ++i)
    {
        donut::log::info("ClusterLodStreaming move-args[%u] src=0x%016llx dst=0x%016llx",
                         i,
                         (unsigned long long)src[i],
                         (unsigned long long)dst[i]);
    }
}

}  // namespace rtxmg
