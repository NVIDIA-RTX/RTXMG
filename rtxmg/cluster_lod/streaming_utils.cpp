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
#include <cassert>
#include <cstring>
#include <vector>

#include "rtxmg/cluster_lod/streaming_utils.h"

namespace rtxmg {

//////////////////////////////////////////////////////////////////////////
//
// StreamingRequests

void StreamingRequests::Init(nvrhi::IDevice*       device,
                             const StreamingConfig& config,
                             uint32_t              groupCountAlignment,
                             uint32_t              /*clusterCountAlignment*/)
{
    m_device     = device;
    m_shaderData = {};
    m_shaderData.maxLoads   = config.maxPerFrameLoadRequests;
    m_shaderData.maxUnloads = config.maxPerFrameUnloadRequests;

    // some values are aligned up for easier gpu kernel access
    const uint64_t loadBytes   = sizeof(GeometryGroup)
                                 * rtxmg::align_up(config.maxPerFrameLoadRequests,   groupCountAlignment);
    const uint64_t unloadBytes = sizeof(GeometryGroup)
                                 * rtxmg::align_up(config.maxPerFrameUnloadRequests, groupCountAlignment);

    // Per-slot layout: [loadGroups][unloadGroups], 8-byte aligned per region.
    m_loadGroupsOffset   = 0;
    m_unloadGroupsOffset = rtxmg::align_up(m_loadGroupsOffset + loadBytes, uint64_t(8));
    m_requestSize        = rtxmg::align_up(m_unloadGroupsOffset + unloadBytes, uint64_t(8));

    // Shader-visible per-task stride (in uint2 elements): the shaders bind the
    // whole ring and derive their slot base from `taskIndex`.
    constexpr uint64_t kUint2Bytes = sizeof(uint32_t) * 2;
    assert((m_requestSize        % kUint2Bytes) == 0);
    assert((m_unloadGroupsOffset % kUint2Bytes) == 0);
    m_shaderData.taskSlotStride          = uint32_t(m_requestSize        / kUint2Bytes);
    m_shaderData.unloadGroupsOffsetElems = uint32_t(m_unloadGroupsOffset / kUint2Bytes);

    // Shader-data header snapshots come after all per-slot data.
    m_shaderDataOffset = m_requestSize * kStreamingMaxActiveTasks;

    const uint64_t totalBytes = m_shaderDataOffset
                              + sizeof(shaderio::StreamingFrameRequest) * kStreamingMaxActiveTasks;

    // Device-side buffer: written by streaming compute shaders, source for the
    // readback copy.  The uint2 stride lets the group ring be bound as
    // StructuredBuffer<uint2>; raw views cover the byte-addressed header writes.
    {
        auto desc = nvrhi::BufferDesc()
                        .setByteSize(totalBytes)
                        .setStructStride(sizeof(uint32_t) * 2)  // uint2
                        .setCanHaveUAVs(true)
                        .setCanHaveRawViews(true)
                        .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
                        .setKeepInitialState(true)
                        .setDebugName("ClusterLodStreamingRequests");
        m_requestBuffer.Create(desc, device);
    }

    // Host-side readback buffer, persistently mapped for the buffer's lifetime.
    {
        auto desc = nvrhi::BufferDesc()
                        .setByteSize(totalBytes)
                        .setCpuAccess(nvrhi::CpuAccessMode::Read)
                        .setInitialState(nvrhi::ResourceStates::CopyDest)
                        .setKeepInitialState(true)
                        .setDebugName("ClusterLodStreamingRequestsHost");
        m_requestHostBuffer.Create(desc, device);
    }

    m_hostMapping = device->mapBuffer(m_requestHostBuffer.GetBuffer(), nvrhi::CpuAccessMode::Read);
    assert(m_hostMapping && "failed to persistently-map StreamingRequests host buffer");

    // Per-slot host pointers, dereferenceable once that slot's readback has
    // completed on the GPU timeline.
    for (uint32_t c = 0; c < kStreamingMaxActiveTasks; ++c)
    {
        TaskInfo& task = m_taskInfos[c];
        task = {};

        task.shaderData = reinterpret_cast<const shaderio::StreamingFrameRequest*>(
            uintptr_t(m_hostMapping) + m_shaderDataOffset + sizeof(shaderio::StreamingFrameRequest) * c);
        task.loadGeometryGroups = reinterpret_cast<const GeometryGroup*>(
            uintptr_t(m_hostMapping) + m_requestSize * c + m_loadGroupsOffset);
        task.unloadGeometryGroups = reinterpret_cast<const GeometryGroup*>(
            uintptr_t(m_hostMapping) + m_requestSize * c + m_unloadGroupsOffset);
    }
}

void StreamingRequests::Deinit()
{
    if (m_hostMapping && m_device)
    {
        m_device->unmapBuffer(m_requestHostBuffer.GetBuffer());
    }
    m_hostMapping = nullptr;

    m_requestBuffer.Release();
    m_requestHostBuffer.Release();
    m_device = nullptr;

    for (uint32_t c = 0; c < kStreamingMaxActiveTasks; ++c)
    {
        m_taskInfos[c] = {};
    }
    m_shaderData         = {};
    m_requestSize        = 0;
    m_shaderDataOffset   = 0;
    m_loadGroupsOffset   = 0;
    m_unloadGroupsOffset = 0;
}

size_t StreamingRequests::GetOperationsSize() const
{
    return m_requestBuffer.GetVramBytes();
}

void StreamingRequests::ApplyTask(shaderio::StreamingFrameRequest& shaderData,
                                  uint32_t                          taskIndex,
                                  uint32_t                          frameIndex)
{
    shaderData            = m_shaderData;
    shaderData.taskIndex  = taskIndex;
    shaderData.frameIndex = frameIndex;
}

void StreamingRequests::DownloadFrameRequests(nvrhi::ICommandList*                   commandList,
                                              const shaderio::StreamingFrameRequest& shaderData,
                                              nvrhi::IBuffer*                        srcBuffer,
                                              uint64_t                               srcBufferOffset)
{
    const uint32_t taskIndex = shaderData.taskIndex;

    // 1) Copy the newly-requested indices (load + unload group arrays) to host.
    commandList->copyBuffer(m_requestHostBuffer.GetBuffer(), m_requestSize * uint64_t(taskIndex),
                            m_requestBuffer.GetBuffer(),     m_requestSize * uint64_t(taskIndex),
                            m_requestSize);

    // 2) Snapshot the live shaderio header; only its counters are really
    // needed, but the whole header is cheap and useful for comparisons.
    commandList->copyBuffer(m_requestHostBuffer.GetBuffer(),
                            m_shaderDataOffset + sizeof(shaderio::StreamingFrameRequest) * uint64_t(taskIndex),
                            srcBuffer, srcBufferOffset,
                            sizeof(shaderio::StreamingFrameRequest));
}

//////////////////////////////////////////////////////////////////////////
//
// StreamingResident

void StreamingResident::Init(nvrhi::IDevice*       device,
                             const StreamingConfig& config,
                             uint32_t              groupCountAlignment,
                             uint32_t              clusterCountAlignment)
{
    m_groupAllocator.init(config.maxGroups);
    m_clusterAllocator.init(config.maxClusters);

    // some values are aligned up for easier gpu kernel access
    m_maxClusters = rtxmg::align_up(config.maxClusters, clusterCountAlignment);
    m_maxGroups   = rtxmg::align_up(config.maxGroups,   groupCountAlignment);

    m_lowDetailGroupsCount      = 0;
    m_lowDetailClustersCount    = 0;
    m_lowDetailTrianglesCount   = 0;
    m_lowDetailMaxGroupClusters = 0;

    m_activeGroupsCount    = 0;
    m_activeClustersCount  = 0;
    m_activeTrianglesCount = 0;

    m_groupIndicesUpdateRange    = {};
    m_groups                     = {};
    m_activeGroupIndices         = {};
    m_mapGeometryGroup2Residency = {};

    m_mapGeometryGroup2Residency.reserve(m_maxGroups);
    m_groups.resize(m_maxGroups);
    m_activeGroupIndices.resize(m_maxGroups);

    m_groupsBuffer.Create      (m_maxGroups,   "ClusterLodStreamingResidentGroups",   device);
    m_groupIDsBuffer.Create    (m_maxGroups,   "ClusterLodStreamingResidentGroupIDs", device);
    m_clustersBuffer.Create    (m_maxClusters, "ClusterLodStreamingResidentClusters", device);
    // +groupCountAlignment of slack: the streaming kernels load
    // u_ActiveGroups[persistentGroupsCount + threadID] before guarding on the
    // real count, which can reach up to a thread group past m_maxGroups.
    m_activeGroupsBuffer.Create(m_maxGroups + groupCountAlignment,
                                "ClusterLodStreamingActiveGroups", device);
    m_activeUpdateBuffer.Create(m_maxGroups * kStreamingMaxActiveTasks,
                                  "ClusterLodStreamingActiveUpdate", device);

    // CPU-write-mapped staging buffer for the per-frame active[] uploads.
    {
        const uint64_t bytes = sizeof(uint32_t) * uint64_t(m_maxGroups) * kStreamingMaxActiveTasks;
        auto desc = nvrhi::BufferDesc()
                        .setByteSize(bytes)
                        .setCpuAccess(nvrhi::CpuAccessMode::Write)
                        .setDebugName("ClusterLodStreamingActiveHost")
                        .setStructStride(sizeof(uint32_t));
        m_activeHostBuffer.Create(desc, device);
    }

    m_shaderData = {};
    // clasBaseAddress / clasMaxSize get populated by initClas.
}

void StreamingResident::Deinit()
{
    // Free any still-allocated IDs so IDPool::deinit's assert passes.
    m_groupAllocator.destroyAll();
    m_clusterAllocator.destroyAll();
    m_groupAllocator.deinit();
    m_clusterAllocator.deinit();

    m_groupsBuffer.Release();
    m_groupIDsBuffer.Release();
    m_clustersBuffer.Release();
    m_activeGroupsBuffer.Release();
    m_activeUpdateBuffer.Release();
    m_activeHostBuffer.Release();

    // Release CLAS pool too (no-op if initClas was never called).
    DeinitClas();

    m_groups.clear();
    m_activeGroupIndices.clear();
    m_mapGeometryGroup2Residency.clear();
    m_groupIndicesUpdateRange = {};

    m_maxGroups                 = 0;
    m_maxClusters               = 0;
    m_lowDetailGroupsCount      = 0;
    m_lowDetailClustersCount    = 0;
    m_lowDetailTrianglesCount   = 0;
    m_lowDetailMaxGroupClusters = 0;
    m_activeGroupsCount         = 0;
    m_activeClustersCount       = 0;
    m_activeTrianglesCount      = 0;
    m_shaderData                = {};
}

size_t StreamingResident::GetOperationsSize() const
{
    return m_groupsBuffer.GetVramBytes()
         + m_groupIDsBuffer.GetVramBytes()
         + m_clustersBuffer.GetVramBytes()
         + m_activeGroupsBuffer.GetVramBytes()
         + m_activeUpdateBuffer.GetVramBytes();
}

void StreamingResident::InitClas(nvrhi::IDevice*              device,
                                 const StreamingConfig&       config,
                                 shaderio::StreamingResident& outShaderData)
{
    m_maxClasBytes = config.maxClasMegaBytes * 1024ull * 1024ull;

    // Typed structured-buffer UAV metadata arrays.
    m_clasAddressesBuffer.Create(m_maxClusters, "ClusterLodStreamingClasAddresses", device);
    m_clasSizesBuffer.Create    (m_maxClusters, "ClusterLodStreamingClasSizes",     device);
    if (config.usePersistentClasAllocator)
    {
        m_groupClasSizesBuffer.Create(m_maxGroups, "ClusterLodStreamingGroupClasSizes", device);
    }

    // Single-element GPU-persistent allocator scalars: the host zero-inits this
    // once and never re-uploads, so stream_dispatch_setup's cursor/budget survive the
    // wholesale SceneStreaming upload.
    m_residentPersistentBuffer.Create(1, "ClusterLodStreamingResidentPersistent", device);

    // Raw CLAS payload — also the backing storage the CLAS build writes into
    // (ExplicitDestinations).
    {
        auto desc = nvrhi::BufferDesc()
                        .setByteSize(m_maxClasBytes)
                        .setFormat(nvrhi::Format::UNKNOWN)
                        .setCanHaveUAVs(true)
                        .setCanHaveRawViews(true)
                        .setIsAccelStructStorage(true)
                        .setInitialState(nvrhi::ResourceStates::AccelStructWrite)
                        .setKeepInitialState(true)
                        .setDebugName("ClusterLodStreamingClasData");
        m_clasDataBuffer.Create(desc, device);
    }

    m_shaderData.clasBaseAddress = m_clasDataBuffer.GetGpuVirtualAddress();
    m_shaderData.clasMaxSize     = m_maxClasBytes;

    outShaderData = m_shaderData;
}

size_t StreamingResident::GetClasOperationsSize() const
{
    // Management metadata only — m_clasDataBuffer is the payload pool, not
    // "operations" bytes.
    return m_clasAddressesBuffer.GetVramBytes()
         + m_clasSizesBuffer.GetVramBytes()
         + m_groupClasSizesBuffer.GetVramBytes()
         + m_residentPersistentBuffer.GetVramBytes();
}

StreamingResident::ResidencyStats StreamingResident::GetStats() const
{
    ResidencyStats stats;
    stats.residentGroups      = m_activeGroupsCount;
    stats.residentClusters    = m_activeClustersCount;
    stats.residentTriangles   = m_activeTrianglesCount;
    stats.persistentGroups    = m_lowDetailGroupsCount;
    stats.persistentClusters  = m_lowDetailClustersCount;
    stats.persistentTriangles = m_lowDetailTrianglesCount;
    return stats;
}

void StreamingResident::DeinitClas()
{
    m_clasAddressesBuffer.Release();
    m_clasSizesBuffer.Release();
    m_groupClasSizesBuffer.Release();
    m_residentPersistentBuffer.Release();
    m_clasDataBuffer.Release();

    m_maxClasBytes               = 0;
    m_shaderData.clasBaseAddress = 0;
    m_shaderData.clasMaxSize     = 0;
}

void StreamingResident::Reset(shaderio::StreamingResident& shaderData)
{
    // free everything past the persistent (low-detail) prefix
    for (uint32_t activeGroup = m_lowDetailGroupsCount; activeGroup < m_activeGroupsCount; ++activeGroup)
    {
        Group& group = m_groups[m_activeGroupIndices[activeGroup]];

        m_mapGeometryGroup2Residency.erase(group.geometryGroup.key);
        m_groupAllocator.destroyID(group.groupResidentID);
        m_clusterAllocator.destroyRangeID(group.clusterResidentID, group.clusterCount);
    }

    m_activeGroupsCount    = m_lowDetailGroupsCount;
    m_activeClustersCount  = m_lowDetailClustersCount;
    m_activeTrianglesCount = m_lowDetailTrianglesCount;
    m_residencyEpoch++;

    m_groupIndicesUpdateRange = {};

    // The CBV carries only the active counts above the persistent prefix.
    m_shaderData.activeGroupsCount      = m_activeGroupsCount   - m_lowDetailGroupsCount;
    m_shaderData.activeClustersCount    = m_activeClustersCount - m_lowDetailClustersCount;
    m_shaderData.persistentGroupsCount  = m_lowDetailGroupsCount;
    shaderData = m_shaderData;
}

void StreamingResident::UploadInitialState(nvrhi::ICommandList*         commandList,
                                           shaderio::StreamingResident& outShaderData)
{
    // All groups + clusters added so far become the persistent low-detail state.
    m_lowDetailGroupsCount      = m_activeGroupsCount;
    m_lowDetailClustersCount    = m_activeClustersCount;
    m_lowDetailTrianglesCount   = m_activeTrianglesCount;
    m_lowDetailMaxGroupClusters = 0;

    if (m_lowDetailGroupsCount > 0)
    {
        std::vector<shaderio::StreamingGroup> shaderGroups(m_lowDetailGroupsCount);
        for (uint32_t g = 0; g < m_lowDetailGroupsCount; ++g)
        {
            const Group& group = m_groups[g];
            assert(group.groupResidentID == g);

            shaderio::StreamingGroup& shaderGroup = shaderGroups[g];
            shaderGroup.geometryID    = group.geometryGroup.geometryID;
            shaderGroup.lodLevel      = uint16_t(group.lodLevel);
            shaderGroup.age           = uint16_t(0x1234);  // never-visited sentinel
            shaderGroup.groupAddress  = group.groupAddress;

            m_lowDetailMaxGroupClusters = std::max(m_lowDetailMaxGroupClusters, uint32_t(group.clusterCount));
        }

        commandList->writeBuffer(m_groupsBuffer.GetBuffer(),
                                 shaderGroups.data(),
                                 shaderGroups.size() * sizeof(shaderio::StreamingGroup));
    }

    // Resident cluster-address entries for the persistent prefix: each one points
    // at its Cluster header, from which the hit shader derives payload offsets.
    // Dynamically loaded groups are patched by stream_update_scene instead.
    if (m_activeClustersCount > 0)
    {
        std::vector<shaderio::ClusterAddress> clusters(
            m_activeClustersCount,
            shaderio::ClusterAddress{ shaderio::kStreamingInvalidSrvIndex, 0u });
        for (uint32_t g = 0; g < m_lowDetailGroupsCount; ++g)
        {
            const Group& group = m_groups[g];
            assert(group.groupAddress.srvIndex != shaderio::kStreamingInvalidSrvIndex);
            for (uint32_t c = 0; c < group.clusterCount; ++c)
            {
                clusters[group.clusterResidentID + c] = shaderio::ClusterAddress{
                    group.groupAddress.srvIndex,
                    group.groupAddress.byteOffset
                        + uint32_t(sizeof(shaderio::Group))
                        + uint32_t(sizeof(shaderio::Cluster)) * c,
                };
            }
        }

        commandList->writeBuffer(m_clustersBuffer.GetBuffer(),
                                 clusters.data(),
                                 clusters.size() * sizeof(shaderio::ClusterAddress));
    }

    // The shader-visible counts skip the persistent prefix; persistentGroupsCount
    // is the element base shaders add to reach the dynamic activeGroups region.
    m_shaderData.activeGroupsCount      = m_activeGroupsCount   - m_lowDetailGroupsCount;
    m_shaderData.activeClustersCount    = m_activeClustersCount - m_lowDetailClustersCount;
    m_shaderData.persistentGroupsCount  = m_lowDetailGroupsCount;
    outShaderData = m_shaderData;
}

size_t StreamingResident::UploadActiveGroupsDelta(nvrhi::ICommandList* commandList, uint32_t taskIndex)
{
    TaskInfo& task = m_taskInfos[taskIndex];

    const uint32_t taskOffset = m_maxGroups * taskIndex;

    task.region     = {};
    task.shaderData = m_shaderData;
    // skip persistent in the shader-visible active counts
    task.shaderData.activeGroupsCount   = m_activeGroupsCount   - m_lowDetailGroupsCount;
    task.shaderData.activeClustersCount = m_activeClustersCount - m_lowDetailClustersCount;

    const uint32_t deltaCount = m_groupIndicesUpdateRange.Count();
    if (deltaCount == 0)
    {
        return 0;
    }

    // 1) memcpy the modified-index window into the host-mapped staging buffer
    //    at this task slot's offset (so concurrent slots don't trample each other).
    nvrhi::IDevice* device  = commandList->getDevice();
    void*           hostPtr = device->mapBuffer(m_activeHostBuffer.GetBuffer(), nvrhi::CpuAccessMode::Write);
    if (!hostPtr)
    {
        // Recording the copy anyway would push stale staging bytes into the live
        // active-groups list; leaving the range set retries the delta next frame.
        assert(!"failed to map StreamingResident host buffer");
        return 0;
    }

    uint32_t* dst = reinterpret_cast<uint32_t*>(hostPtr) + taskOffset;
    std::memcpy(dst,
                m_activeGroupIndices.data() + m_groupIndicesUpdateRange.lo,
                sizeof(uint32_t) * deltaCount);
    device->unmapBuffer(m_activeHostBuffer.GetBuffer());

    const uint64_t bytes           = sizeof(uint32_t) * deltaCount;
    const uint64_t updateBufOffset = sizeof(uint32_t) * uint64_t(taskOffset);

    // 2) GPU-copy host staging → activeUpdate buffer at the task-slot offset.
    commandList->copyBuffer(m_activeUpdateBuffer.GetBuffer(), updateBufOffset,
                            m_activeHostBuffer.GetBuffer(),   updateBufOffset,
                            bytes);

    // 3) Record where the second copy (CommitActiveGroupsDelta) should land in activeGroups.
    task.region.size      = bytes;
    task.region.srcOffset = updateBufOffset;  // src in m_activeUpdateBuffer
    task.region.dstOffset = sizeof(uint32_t) * uint64_t(m_groupIndicesUpdateRange.lo);  // dst in m_activeGroupsBuffer

    m_groupIndicesUpdateRange = {};
    return bytes;
}

void StreamingResident::ApplyTask(shaderio::StreamingResident& shaderData, uint32_t taskIndex, uint32_t frameIndex)
{
    shaderData            = m_taskInfos[taskIndex].shaderData;
    shaderData.taskIndex  = taskIndex;
    shaderData.frameIndex = frameIndex;
}

void StreamingResident::CommitActiveGroupsDelta(nvrhi::ICommandList* commandList, uint32_t taskIndex)
{
    TaskInfo& task = m_taskInfos[taskIndex];
    if (task.region.size != 0)
    {
        commandList->copyBuffer(m_activeGroupsBuffer.GetBuffer(), task.region.dstOffset,
                                m_activeUpdateBuffer.GetBuffer(), task.region.srcOffset,
                                task.region.size);
    }
}

uint32_t StreamingResident::GetLoadActiveGroupsOffset() const
{
    return m_activeGroupsCount - m_lowDetailGroupsCount;
}

uint32_t StreamingResident::GetLoadActiveClustersOffset() const
{
    return m_activeClustersCount - m_lowDetailClustersCount;
}

bool StreamingResident::CanAllocateGroup(uint32_t numClusters) const
{
    return m_groupAllocator.isRangeAvailable(1) && m_clusterAllocator.isRangeAvailable(numClusters);
}

const StreamingResident::Group* StreamingResident::FindGroup(GeometryGroup geometryGroup) const
{
    auto it = m_mapGeometryGroup2Residency.find(geometryGroup.key);
    if (it == m_mapGeometryGroup2Residency.end())
    {
        return nullptr;
    }
    else
    {
        return &m_groups[it->second];
    }
}

void StreamingResident::CollectActiveGeometryGroups(std::vector<GeometryGroup>& out) const
{
    out.reserve(out.size() + m_activeGroupsCount);
    for (uint32_t activeIndex = 0; activeIndex < m_activeGroupsCount; ++activeIndex)
    {
        const uint32_t residentID = m_activeGroupIndices[activeIndex];
        assert(residentID < m_groups.size());
        out.push_back(m_groups[residentID].geometryGroup);
    }
}

void StreamingResident::DownloadResidentClasBytes(
    nvrhi::ICommandList* commandList,
    uint32_t granularityByteShift,
    std::vector<std::array<uint64_t, shaderio::kMaxLodLevels>>& out)
{
    // Compaction allocator leaves the per-group CLAS-sizes buffer null.
    if (!m_groupClasSizesBuffer.GetBuffer())
        return;

    // uint2 per groupResidentID: (allocSize, wastedByteSize).  Async ⇒ returns
    // the prior frame's copy and schedules a new one; empty until ready.
    std::vector<donut::math::uint2> sizes = m_groupClasSizesBuffer.Download(commandList, /*async*/ true);
    if (sizes.empty())
        return;  // not ready yet — keep the last good aggregate

    // groupClasSizes.x is allocSize in allocator UNITS; convert to bytes with
    // the same shift the shader uses (allocByteSize = allocSize << shift).
    const uint32_t shift = granularityByteShift;
    for (auto& geom : out)
        geom.fill(0);
    for (uint32_t activeIndex = 0; activeIndex < m_activeGroupsCount; ++activeIndex)
    {
        const Group& g = m_groups[m_activeGroupIndices[activeIndex]];
        if (g.groupResidentID >= sizes.size() || g.lodLevel >= shaderio::kMaxLodLevels)
            continue;
        const uint32_t geometryID = g.geometryGroup.geometryID;
        if (geometryID >= out.size())
            out.resize(geometryID + 1, std::array<uint64_t, shaderio::kMaxLodLevels>{});
        out[geometryID][g.lodLevel] += uint64_t(sizes[g.groupResidentID].x) << shift;
    }
}

StreamingResident::Group* StreamingResident::AddGroup(GeometryGroup geometryGroup, uint32_t clusterCount, uint32_t triangleCount)
{
    // createID leaves its out-param untouched on failure, so an assert-only
    // guard indexes m_groups with garbage in Release.  Refuse instead.
    uint32_t groupResidentID   = 0;
    uint32_t clusterResidentID = 0;
    if (!m_groupAllocator.createID(groupResidentID))
        return nullptr;
    if (!m_clusterAllocator.createRangeID(clusterResidentID, clusterCount))
    {
        m_groupAllocator.destroyID(groupResidentID);
        return nullptr;
    }
    // Both counts are stored as uint16_t below.
    if (clusterCount > UINT16_MAX || triangleCount > UINT16_MAX)
    {
        m_clusterAllocator.destroyRangeID(clusterResidentID, clusterCount);
        m_groupAllocator.destroyID(groupResidentID);
        return nullptr;
    }

    StreamingResident::Group& group = m_groups[groupResidentID];

    assert(m_mapGeometryGroup2Residency.find(geometryGroup.key) == m_mapGeometryGroup2Residency.end());
    m_mapGeometryGroup2Residency.insert({ geometryGroup.key, groupResidentID });

    group.activeIndex       = m_activeGroupsCount++;
    group.geometryGroup     = geometryGroup;
    group.groupResidentID   = groupResidentID;
    group.clusterResidentID = clusterResidentID;
    group.clusterCount      = static_cast<uint16_t>(clusterCount);
    group.triangleCount     = static_cast<uint16_t>(triangleCount);
    group.deviceAddress     = shaderio::kStreamingInvalidAddressStart;
    group.groupAddress      = shaderio::GroupAddress{ shaderio::kStreamingInvalidSrvIndex, 0u };

    m_activeGroupIndices[group.activeIndex] = groupResidentID;

    // no update range needed: the Update task uploads all newly added groups

    m_activeClustersCount += clusterCount;
    m_activeTrianglesCount += triangleCount;
    m_residencyEpoch++;

    return &m_groups[groupResidentID];
}

void StreamingResident::RemoveGroup(uint32_t groupResidentID)
{
    StreamingResident::Group& group = m_groups[groupResidentID];
    assert(m_mapGeometryGroup2Residency.find(group.geometryGroup.key) != m_mapGeometryGroup2Residency.end());
    m_mapGeometryGroup2Residency.erase(group.geometryGroup.key);

    {
        // remove group from compact indices list
        uint32_t activeIndex = group.activeIndex;

        // classic swapping our position in the active list with last element
        if (activeIndex + 1 != m_activeGroupsCount)
        {
            uint32_t lastResidentID              = m_activeGroupIndices[m_activeGroupsCount - 1];
            m_groups[lastResidentID].activeIndex = activeIndex;
            m_activeGroupIndices[activeIndex]    = lastResidentID;

            // we track those changes so that we later minimize the upload of
            // changed indices
            m_groupIndicesUpdateRange.Update(activeIndex);
        }
        m_activeGroupsCount--;
    }

    m_activeClustersCount -= group.clusterCount;
    m_activeTrianglesCount -= group.triangleCount;
    m_residencyEpoch++;

    m_groupAllocator.destroyID(groupResidentID);
    m_clusterAllocator.destroyRangeID(group.clusterResidentID, group.clusterCount);

    group = {};
}

//////////////////////////////////////////////////////////////////////////
//
// StreamingAllocator

namespace {

// Sub-region carver for the management buffer's layout.
struct BufferRanges
{
    uint64_t tempOffset = 0;

    uint64_t Append(uint64_t size, uint64_t alignment)
    {
        tempOffset = (tempOffset + alignment - 1ull) & ~(alignment - 1ull);
        uint64_t offset = tempOffset;
        tempOffset += size;
        return offset;
    }

    uint64_t GetSize(uint64_t alignment = 4) const
    {
        return (tempOffset + alignment - 1ull) & ~(alignment - 1ull);
    }
};

}  // namespace

void StreamingAllocator::Init(nvrhi::IDevice*               device,
                              size_t                        totalMegaBytes,
                              uint32_t                      maxAllocationByteSize,
                              uint32_t                      granularityByteSize,
                              uint32_t                      sectorSizeShift,
                              shaderio::StreamingAllocator& shaderData)
{
    granularityByteSize = std::max(1u, granularityByteSize);

    // at least 2 warps
    assert(sectorSizeShift > 5 && granularityByteSize <= 0xFFFF);

    uint32_t granularityByteShift = 0;
    while ((1u << granularityByteShift) < granularityByteSize && granularityByteShift <= 16)
    {
        granularityByteShift++;
    }
    // want power of two
    assert(granularityByteShift <= 16 && granularityByteSize == (1u << granularityByteShift));

    size_t sectorSize32s = size_t(1) << sectorSizeShift;
    size_t memoryBits    = size_t(totalMegaBytes) * 1024 * 1024 / granularityByteSize;
    size_t memory32s     = memoryBits / 32;
    size_t sectorCount   = memory32s / sectorSize32s;

    m_shaderData                      = {};
    m_shaderData.freeGapsCounter      = 0;
    m_shaderData.granularityByteShift = granularityByteShift;
    // align up to be multiple of 32
    m_shaderData.maxAllocationSize = (((maxAllocationByteSize + granularityByteSize - 1) / granularityByteSize) + 31) & (~31);
    m_shaderData.sectorSizeShift          = sectorSizeShift;
    m_shaderData.sectorMaxAllocationSized = uint32_t(sectorSize32s * 32 / m_shaderData.maxAllocationSize);
    m_shaderData.sectorCount              = uint32_t(sectorCount);

    // can only manage memory in multiple of sectorSize
    // so there might be some initial waste
    m_shaderData.baseWastedSize = uint32_t(memory32s - (sectorCount * sectorSize32s));

    // reset to multiples of sectors
    memory32s = sectorCount * sectorSize32s;

    // Capacity of the used-bits map, NOT upstream's live population count of
    // set bits - nothing here maintains one.  It only bounds the host-side
    // freelist dump's scan.  See review K-16 before porting upstream's
    // allocatedSize-vs-usedBits invariant against it.
    m_shaderData.usedBitsCount = uint32_t(memory32s);

    // The 7 allocator sub-arrays live contiguously inside a single
    // RWByteAddressBuffer, addressed by these per-region byte offsets.  The
    // uint16 freeGapsSize region is 4-byte aligned; its entries stay 2-byte.
    BufferRanges ranges;
    m_shaderData.freeGapsPosByteOffset       = uint32_t(ranges.Append(sizeof(uint32_t) * memory32s, 4));
    m_shaderData.freeGapsSizeByteOffset      = uint32_t(ranges.Append(sizeof(uint16_t) * memory32s, 4));
    m_shaderData.freeGapsPosBinnedByteOffset = uint32_t(ranges.Append(sizeof(uint32_t) * memory32s, 4));
    m_shaderData.freeSizeRangesByteOffset    = uint32_t(ranges.Append(sizeof(shaderio::AllocatorRange) * m_shaderData.maxAllocationSize, 8));
    m_shaderData.usedSectorBitsByteOffset    = uint32_t(ranges.Append(sizeof(uint32_t) * ((sectorCount + 31) / 32), 4));
    m_shaderData.usedBitsByteOffset          = uint32_t(ranges.Append(sizeof(uint32_t) * memory32s, 4));
    m_shaderData.statsByteOffset             = uint32_t(ranges.Append(sizeof(shaderio::AllocatorStats), 8));

    nvrhi::BufferDesc mgmtDesc;
    mgmtDesc.byteSize         = ranges.GetSize();
    mgmtDesc.format           = nvrhi::Format::UNKNOWN;
    mgmtDesc.canHaveUAVs      = true;
    mgmtDesc.canHaveRawViews  = true;
    mgmtDesc.structStride     = 0;
    mgmtDesc.initialState     = nvrhi::ResourceStates::Common;
    mgmtDesc.keepInitialState = true;
    mgmtDesc.debugName        = "rtxmg::StreamingAllocator::managementBuffer";
    m_managementBuffer.Create(mgmtDesc, device);

    m_shaderData.dispatchFreeGapsInsert.groupsX = 1;
    m_shaderData.dispatchFreeGapsInsert.groupsY = 1;
    m_shaderData.dispatchFreeGapsInsert.groupsZ = 1;

    // Zero blob for the per-frame freeSizeRanges clear (see ClearFreeSizeRanges),
    // allocated once here rather than per frame.
    m_freeSizeRangesZeroBlob.assign(
        sizeof(shaderio::AllocatorRange) * size_t(m_shaderData.maxAllocationSize),
        uint8_t(0));

    shaderData = m_shaderData;
}

void StreamingAllocator::Deinit()
{
    m_managementBuffer.Release();
    m_freeSizeRangesZeroBlob.clear();
    m_freeSizeRangesZeroBlob.shrink_to_fit();
    m_shaderData = {};
}

size_t StreamingAllocator::GetOperationsSize() const
{
    return m_managementBuffer.GetVramBytes();
}

uint32_t StreamingAllocator::GetMaxSized() const
{
    return m_shaderData.sectorMaxAllocationSized * m_shaderData.sectorCount;
}

void StreamingAllocator::ClearManagementBuffer(nvrhi::ICommandList* cmd)
{
    assert(cmd != nullptr);
    cmd->clearBufferUInt(m_managementBuffer.GetBuffer(), 0u);
}

void StreamingAllocator::ClearFreeSizeRanges(nvrhi::ICommandList* cmd)
{
    assert(cmd != nullptr);

    // Clear the per-class freeSizeRanges bins that build_freegaps refills each
    // frame; nvrhi's clearBufferUInt is whole-buffer only, hence the zero blob.
    // The caller's wholesale shader-data upload covers the other per-frame
    // allocator fields.
    if (!m_freeSizeRangesZeroBlob.empty())
    {
        cmd->writeBuffer(m_managementBuffer.GetBuffer(),
                         m_freeSizeRangesZeroBlob.data(),
                         m_freeSizeRangesZeroBlob.size(),
                         uint64_t(m_shaderData.freeSizeRangesByteOffset));
    }
}

//////////////////////////////////////////////////////////////////////////
//
// StreamingUpdates

void StreamingUpdates::Init(nvrhi::IDevice*        device,
                            const StreamingConfig& config,
                            uint32_t               geometryCount,
                            uint32_t               groupCountAlignment,
                            uint32_t               clusterCountAlignment)
{
    m_device                = device;
    m_useBlasCaching        = config.allowBlasCaching;
    m_clusterCountAlignment = clusterCountAlignment;
    m_scheduleIndex         = 0;
    m_pendingNew            = {};

    memset(m_scheduledNew, 0, sizeof(m_scheduledNew));
    memset(m_scheduledNewFrame, 0, sizeof(m_scheduledNewFrame));

    // some values are aligned up for easier gpu kernel access

    uint32_t loadRequests   = rtxmg::align_up(config.maxPerFrameLoadRequests, groupCountAlignment);
    uint32_t unloadRequests = rtxmg::align_up(config.maxPerFrameUnloadRequests, groupCountAlignment);

    static_assert(sizeof(shaderio::StreamingGeometryPatch) <= sizeof(shaderio::StreamingPatch));

    m_shaderData                        = {};
    m_shaderData.patchGroupsCount       = loadRequests + unloadRequests;
    m_shaderData.patchUnloadGroupsCount = unloadRequests;
    if (config.allowBlasCaching)
    {
        // caching up to only 1 blas per geometry
        uint32_t blasCount = std::min(geometryCount, config.maxPerFrameLoadRequests + config.maxPerFrameUnloadRequests);

        m_shaderData.patchCachedBlasCount = rtxmg::align_up(blasCount, groupCountAlignment);
    }

    // Per-task slot stride.  ApplyTask() writes its per-frame counts into its
    // output parameter, not into m_shaderData, so this sum stays constant and
    // GetPatchesByteOffsetForTask / UploadTaskPatches recompute it inline.
    uint32_t framePatchCount = m_shaderData.patchGroupsCount + m_shaderData.patchCachedBlasCount;

    m_unloadHandles = {};
    m_unloadHandles.resize(unloadRequests * kStreamingMaxActiveTasks);

    // Device-side patches ring (one slot per active task).
    {
        auto desc = nvrhi::BufferDesc()
                        .setByteSize(sizeof(shaderio::StreamingPatch) * framePatchCount * kStreamingMaxActiveTasks)
                        .setFormat(nvrhi::Format::UNKNOWN)
                        .setStructStride(sizeof(shaderio::StreamingPatch))
                        .setCanHaveUAVs(true)
                        .setInitialState(nvrhi::ResourceStates::Common)
                        .setKeepInitialState(true)
                        .setDebugName("rtxmg::StreamingUpdates::patchesBuffer");
        m_patchesBuffer.Create(desc, device);
    }

    // Host-mapped patches buffer (CPU writes per-task, GPU reads via copyBuffer).
    {
        auto desc = nvrhi::BufferDesc()
                        .setByteSize(sizeof(shaderio::StreamingPatch) * framePatchCount * kStreamingMaxActiveTasks)
                        .setCpuAccess(nvrhi::CpuAccessMode::Write)
                        .setStructStride(0)
                        .setDebugName("rtxmg::StreamingUpdates::patchesHostBuffer");
        m_patchesHostBuffer.Create(desc, device);
    }

    m_patchesHostMapping = device->mapBuffer(m_patchesHostBuffer.GetBuffer(), nvrhi::CpuAccessMode::Write);
    assert(m_patchesHostMapping && "failed to persistently-map StreamingUpdates host patches buffer");

    // BLAS caching: 16B geometry-patch ring, which can't alias the 32B one.
    m_geometryPatchesPerTask     = config.allowBlasCaching ? m_shaderData.patchCachedBlasCount : 0u;
    m_geometryPatchesHostMapping = nullptr;
    if (m_geometryPatchesPerTask > 0)
    {
        const uint64_t geomElems = uint64_t(m_geometryPatchesPerTask) * kStreamingMaxActiveTasks;
        {
            auto desc = nvrhi::BufferDesc()
                            .setByteSize(sizeof(shaderio::StreamingGeometryPatch) * geomElems)
                            .setFormat(nvrhi::Format::UNKNOWN)
                            .setStructStride(sizeof(shaderio::StreamingGeometryPatch))
                            .setCanHaveUAVs(true)
                            .setInitialState(nvrhi::ResourceStates::Common)
                            .setKeepInitialState(true)
                            .setDebugName("rtxmg::StreamingUpdates::geometryPatchesBuffer");
            m_geometryPatchesBuffer.Create(desc, device);
        }
        {
            auto desc = nvrhi::BufferDesc()
                            .setByteSize(sizeof(shaderio::StreamingGeometryPatch) * geomElems)
                            .setCpuAccess(nvrhi::CpuAccessMode::Write)
                            .setStructStride(0)
                            .setDebugName("rtxmg::StreamingUpdates::geometryPatchesHostBuffer");
            m_geometryPatchesHostBuffer.Create(desc, device);
        }
        m_geometryPatchesHostMapping =
            device->mapBuffer(m_geometryPatchesHostBuffer.GetBuffer(), nvrhi::CpuAccessMode::Write);
        assert(m_geometryPatchesHostMapping && "failed to map StreamingUpdates geometry-patches host buffer");
    }

    shaderio::StreamingPatch* hostBase = static_cast<shaderio::StreamingPatch*>(m_patchesHostMapping);
    shaderio::StreamingGeometryPatch* geomHostBase =
        static_cast<shaderio::StreamingGeometryPatch*>(m_geometryPatchesHostMapping);
    for (uint32_t c = 0; c < kStreamingMaxActiveTasks; c++)
    {
        StreamingUpdates::TaskInfo& task = m_taskInfos[c];
        task.unloadPatches               = hostBase + framePatchCount * c;
        task.loadPatches                 = task.unloadPatches + unloadRequests;
        task.geometryPatches             = geomHostBase
                                               ? geomHostBase + m_geometryPatchesPerTask * c
                                               : nullptr;
        task.unloadHandles               = m_unloadHandles.data() + unloadRequests * c;
    }
}

size_t StreamingUpdates::GetOperationsSize() const
{
    return m_patchesBuffer.GetVramBytes() + m_geometryPatchesBuffer.GetVramBytes();
}

void StreamingUpdates::InitClas(nvrhi::IDevice*        device,
                                const StreamingConfig& config,
                                const BakerConfig&     bakerConfig,
                                uint32_t               maxClusterTriangles)
{
    // some values are aligned up for easier gpu kernel access

    uint32_t maxLoadClusters =
        rtxmg::align_up(config.maxPerFrameLoadRequests * bakerConfig.clusterGroupSize, m_clusterCountAlignment);
    uint32_t maxClusters = rtxmg::align_up(config.maxClusters, m_clusterCountAlignment);

    uint32_t maxMovedClusters = config.usePersistentClasAllocator ? maxLoadClusters : maxClusters;

    // asBuildInput: only a cluster op's srcInfosArray / srcInfosCount need
    // Vulkan's ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY usage; the output
    // arrays get by with STORAGE_BUFFER.  No-op on D3D12 (runtime state only).
    auto makeStructuredBuffer = [&](auto& buf, uint64_t elementCount, uint32_t elementStride, const char* debugName,
                                    bool asBuildInput = false) {
        auto desc = nvrhi::BufferDesc()
                        .setByteSize(elementCount * elementStride)
                        .setStructStride(elementStride)
                        .setCanHaveUAVs(true)
                        .setIsAccelStructBuildInput(asBuildInput)
                        .setInitialState(nvrhi::ResourceStates::Common)
                        .setKeepInitialState(true)
                        .setDebugName(debugName);
        buf.Create(desc, device);
    };

    makeStructuredBuffer(m_newClasBuildsBuffer,          maxLoadClusters, sizeof(shaderio::ClasBuildInfo), "rtxmg::StreamingUpdates::newClasBuilds");
    makeStructuredBuffer(m_newClasAddressesBuffer,       maxLoadClusters, sizeof(uint64_t),                 "rtxmg::StreamingUpdates::newClasAddresses");
    makeStructuredBuffer(m_newClasSizesBuffer,           maxLoadClusters, GetClusterOperationSizeBytes(device), "rtxmg::StreamingUpdates::newClasSizes");
    makeStructuredBuffer(m_newClasResidentIDsBuffer,     maxLoadClusters, sizeof(uint32_t),                 "rtxmg::StreamingUpdates::newClasResidentIDs");

    // The mixed-cluster geometry-indices buffer is written as a UAV by
    // stream_fill_clas_geometry_indices, then read by the CLAS-build hardware
    // through arg.geometryIndexAndFlagsBuffer.  ApplyResidencyUpdate transitions it
    // explicitly — nvrhi can't track the VA reference inside CLAS args.
    {
        // One uint slot per triangle, strided by the scene-observed
        // maxClusterTriangles (not the 256 spec ceiling) and published as
        // sceneMaxClusterTriangles so the shaders agree on the stride.
        // stream_update_scene stashes a ClusterAddress in a slot's first two
        // entries, which stream_fill_clas_geometry_indices then overwrites
        // with the packed per-triangle entries the CLAS builder consumes.
        constexpr size_t kEntryBytes = sizeof(uint32_t);
        const uint64_t bytes = uint64_t(maxLoadClusters)
                             * uint64_t(maxClusterTriangles) * kEntryBytes;
        nvrhi::BufferDesc d;
        d.byteSize                = bytes;
        d.structStride            = uint32_t(kEntryBytes);
        d.canHaveUAVs             = true;
        d.canHaveRawViews         = true;
        d.initialState            = nvrhi::ResourceStates::Common;
        d.keepInitialState        = true;
        d.debugName               = "rtxmg::StreamingUpdates::newClasGeometryIndices";
        m_newClasGeometryIndicesBuffer.Create(d, device);

        // Stashed in our header so ApplyTask's copy propagates it every frame;
        // stream_update_scene stamps it into each mixed cluster's
        // IndirectTriangleClasArgs::geometryIndexAndFlagsBuffer.
        m_shaderData.newClasGeometryIndicesBufferVA =
            m_newClasGeometryIndicesBuffer.GetGpuVirtualAddress();

        m_shaderData.sceneMaxClusterTriangles = maxClusterTriangles;
    }

    // dst = the Move op's dstAddressesArray (an output) — no build-input usage.
    makeStructuredBuffer(m_moveClasDstAddressesBuffer,   maxMovedClusters, sizeof(uint64_t), "rtxmg::StreamingUpdates::moveClasDstAddresses");
    makeStructuredBuffer(m_moveClasSrcAddressesBuffer,   maxMovedClusters, sizeof(uint64_t), "rtxmg::StreamingUpdates::moveClasSrcAddresses", /*asBuildInput=*/true);
}

size_t StreamingUpdates::GetClasOperationsSize() const
{
    return m_newClasBuildsBuffer.GetVramBytes()
         + m_newClasAddressesBuffer.GetVramBytes()
         + m_newClasSizesBuffer.GetVramBytes()
         + m_newClasResidentIDsBuffer.GetVramBytes()
         + m_newClasGeometryIndicesBuffer.GetVramBytes()
         + m_moveClasDstAddressesBuffer.GetVramBytes()
         + m_moveClasSrcAddressesBuffer.GetVramBytes();
}

uint32_t StreamingUpdates::GetMaxCachedBlasBuilds() const
{
    return m_shaderData.patchCachedBlasCount;
}

void StreamingUpdates::DeinitClas()
{
    m_newClasBuildsBuffer.Release();
    m_newClasAddressesBuffer.Release();
    m_newClasSizesBuffer.Release();
    m_newClasResidentIDsBuffer.Release();
    m_newClasGeometryIndicesBuffer.Release();
    m_moveClasDstAddressesBuffer.Release();
    m_moveClasSrcAddressesBuffer.Release();
}

void StreamingUpdates::Deinit()
{
    DeinitClas();
    if (m_patchesHostMapping && m_device)
    {
        m_device->unmapBuffer(m_patchesHostBuffer.GetBuffer());
    }
    m_patchesHostMapping = nullptr;
    m_patchesHostBuffer.Release();
    m_patchesBuffer.Release();

    if (m_geometryPatchesHostMapping && m_device)
    {
        m_device->unmapBuffer(m_geometryPatchesHostBuffer.GetBuffer());
    }
    m_geometryPatchesHostMapping = nullptr;
    m_geometryPatchesHostBuffer.Release();
    m_geometryPatchesBuffer.Release();
    m_geometryPatchesPerTask = 0;

    for (uint32_t i = 0; i < kStreamingMaxActiveTasks; i++)
    {
        m_taskInfos[i] = {};
    }
    m_unloadHandles.clear();
    m_device = nullptr;
}

void StreamingUpdates::Reset()
{
    m_pendingNew = {};
    memset(m_scheduledNew, 0, sizeof(m_scheduledNew));
    memset(m_scheduledNewFrame, 0, sizeof(m_scheduledNewFrame));
    m_scheduleIndex = 0;
}

StreamingUpdates::NewInfo StreamingUpdates::GetFutureNew(uint64_t frameIndex) const
{
    // first get all pending counts that we don't know in which frame they end up yet,
    // but by design are guaranteed in the future of frameIndex
    NewInfo info = m_pendingNew;

    // then all scheduled work after this frame
    for (uint32_t i = 0; i < kStreamingMaxActiveTasks; i++)
    {
        if (m_scheduledNewFrame[i] > frameIndex)
        {
            info.groups += m_scheduledNew[i].groups;
            info.clusters += m_scheduledNew[i].clusters;
        }
    }
    return info;
}

StreamingUpdates::TaskInfo& StreamingUpdates::GetNewTask(uint32_t taskIndex)
{
    TaskInfo& task                   = m_taskInfos[taskIndex];
    task.loadCount                   = 0;
    task.unloadCount                 = 0;
    task.geometryCachedCount         = 0;
    task.geometryCachedClustersCount = 0;
    task.newClusterCount             = 0;
    task.loadActiveGroupsOffset      = ~0u;
    task.loadActiveClustersOffset    = ~0u;

    return task;
}

uint64_t StreamingUpdates::GetPatchesByteOffsetForTask(uint32_t taskIndex) const
{
    // kInvalidTaskIndex means no work this frame — bind at offset 0; callers
    // gate the dispatch behind patchGroupsCount > 0 anyway.
    if (taskIndex == kInvalidTaskIndex)
        return 0;
    const uint32_t framePatchCount = m_shaderData.patchGroupsCount + m_shaderData.patchCachedBlasCount;
    return uint64_t(sizeof(shaderio::StreamingPatch)) * framePatchCount * taskIndex;
}

uint64_t StreamingUpdates::GetGeometryPatchesByteOffsetForTask(uint32_t taskIndex) const
{
    // Per-task slot in the dedicated 16B geometry-patches ring.
    if (taskIndex == kInvalidTaskIndex)
        return 0;
    return uint64_t(sizeof(shaderio::StreamingGeometryPatch)) * m_geometryPatchesPerTask * taskIndex;
}

size_t StreamingUpdates::UploadTaskPatches(nvrhi::ICommandList* cmd, uint32_t taskIndex)
{
    const TaskInfo& task = m_taskInfos[taskIndex];

    assert(task.loadActiveGroupsOffset != ~0u);
    assert(task.loadActiveClustersOffset != ~0u);

    size_t transferSize = 0;

    const uint32_t framePatchCount  = m_shaderData.patchGroupsCount + m_shaderData.patchCachedBlasCount;
    const uint64_t slotBaseHost     = uint64_t(sizeof(shaderio::StreamingPatch)) * framePatchCount * taskIndex;
    const uint64_t slotBaseDevice   = slotBaseHost;
    uint64_t       dstCursor        = slotBaseDevice;

    // Three potential regions per task slot — unload (head), load (after
    // unload), geometry (after load) — one copyBuffer call each.

    if (task.unloadCount)
    {
        const uint64_t size      = uint64_t(sizeof(shaderio::StreamingPatch)) * task.unloadCount;
        const uint64_t srcOffset = slotBaseHost;  // unload at start on host
        cmd->copyBuffer(m_patchesBuffer.GetBuffer(), dstCursor,
                        m_patchesHostBuffer.GetBuffer(), srcOffset,
                        size);
        dstCursor += size;
        transferSize += size;
    }

    if (task.loadCount)
    {
        const uint64_t size      = uint64_t(sizeof(shaderio::StreamingPatch)) * task.loadCount;
        const uint64_t srcOffset = slotBaseHost
                                 + uint64_t(sizeof(shaderio::StreamingPatch)) * m_shaderData.patchUnloadGroupsCount;
        cmd->copyBuffer(m_patchesBuffer.GetBuffer(), dstCursor,
                        m_patchesHostBuffer.GetBuffer(), srcOffset,
                        size);
        dstCursor += size;
        transferSize += size;
    }

    if (task.geometryCachedCount && m_geometryPatchesPerTask > 0)
    {
        // Geometry patches live in their own 16B-stride ring (see init()).
        const uint64_t size      = uint64_t(sizeof(shaderio::StreamingGeometryPatch)) * task.geometryCachedCount;
        const uint64_t geomBase  = GetGeometryPatchesByteOffsetForTask(taskIndex);
        cmd->copyBuffer(m_geometryPatchesBuffer.GetBuffer(), geomBase,
                        m_geometryPatchesHostBuffer.GetBuffer(), geomBase,
                        size);
        transferSize += size;
    }

    // we know this task will get scheduled eventually
    m_pendingNew.clusters += task.newClusterCount;
    m_pendingNew.groups += task.loadCount;

    return transferSize;
}

void StreamingUpdates::ApplyTask(shaderio::StreamingUpdate& shaderData, uint32_t taskIndex, uint32_t frameIndex)
{
    const TaskInfo& task = m_taskInfos[taskIndex];
    // keep basics
    shaderData = m_shaderData;
    // override counts
    shaderData.patchGroupsCount         = task.loadCount + task.unloadCount;
    shaderData.patchUnloadGroupsCount   = task.unloadCount;
    shaderData.patchCachedBlasCount     = task.geometryCachedCount;
    shaderData.patchCachedClustersCount = task.geometryCachedClustersCount;
    shaderData.newClasCount             = task.newClusterCount;
    shaderData.moveClasCounter          = task.newClusterCount;

    shaderData.taskIndex                = taskIndex;
    shaderData.frameIndex               = frameIndex;
    shaderData.loadActiveGroupsOffset   = task.loadActiveGroupsOffset;
    shaderData.loadActiveClustersOffset = task.loadActiveClustersOffset;

    // Zero the mixed-cluster geometry-indices task counter + indirect dispatch
    // grid each task: stream_update_scene refills the counter, and stream_dispatch_setup
    // turns it into the dispatch grid for stream_fill_clas_geometry_indices.
    shaderData.newClasGeometryIndicesTaskCounter = 0;
    shaderData.dispatchClasGeometryIndicesX      = 0;
    shaderData.dispatchClasGeometryIndicesY      = 0;
    shaderData.dispatchClasGeometryIndicesZ      = 0;

    // we also want to keep track of the total amount of "future" cluster builds.
    // This is relevant to ray tracing, as the GPU's allocator need to provide enough
    // space for this number of "worst case" cluster or group sizes.
    assert(m_pendingNew.clusters >= task.newClusterCount);
    assert(m_pendingNew.groups >= task.loadCount);

    m_pendingNew.clusters -= task.newClusterCount;
    m_pendingNew.groups -= task.loadCount;
    m_scheduledNewFrame[m_scheduleIndex % kStreamingMaxActiveTasks]     = frameIndex;
    m_scheduledNew[m_scheduleIndex % kStreamingMaxActiveTasks].clusters = task.newClusterCount;
    m_scheduledNew[m_scheduleIndex % kStreamingMaxActiveTasks].groups   = task.loadCount;
    m_scheduleIndex++;
}

//////////////////////////////////////////////////////////////////////////
//
// StreamingStorage

void StreamingStorage::Init(nvrhi::IDevice*                        device,
                            donut::engine::DescriptorTableManager* descriptorTable,
                            const StreamingConfig&                 config)
{
    m_device          = device;
    m_descriptorTable = descriptorTable;

    m_maxTransferBytes = config.maxTransferMegaBytes * 1024 * 1024;
    // The pool is always at least one block, see
    // StreamingConfig::geometryBlockMegaBytes.
    m_blockBytes    = std::max<size_t>(1, config.geometryBlockMegaBytes) * 1024 * 1024;
    m_maxSceneBytes = std::max<size_t>(config.maxGeometryMegaBytes * 1024 * 1024, m_blockBytes);

    // CPU-mapped upload ring: m_maxTransferBytes per task slot.
    {
        const uint64_t bytes = uint64_t(m_maxTransferBytes) * kStreamingMaxActiveTasks;
        auto desc = nvrhi::BufferDesc()
                        .setByteSize(bytes)
                        .setCpuAccess(nvrhi::CpuAccessMode::Write)
                        .setDebugName("ClusterLodStreamingTransferHost")
                        .setStructStride(0);
        m_transferHostBuffer.Create(desc, device);
    }

    m_transferHostMapping = device->mapBuffer(m_transferHostBuffer.GetBuffer(), nvrhi::CpuAccessMode::Write);
    assert(m_transferHostMapping && "failed to persistently-map StreamingStorage transfer host buffer");

    // BufferSubAllocator owns the per-block buffers + descriptor handles;
    // keepLastBlock seeds one, the rest are created on demand.
    rtxmg::BufferSubAllocator::InitInfo allocInfo;
    allocInfo.device              = device;
    allocInfo.descriptorTable     = descriptorTable;
    allocInfo.debugName           = "ClusterLodStreamingStorage";
    allocInfo.minAlignment        = rtxmg::BufferSubAllocator::kMinAlignment;
    allocInfo.perBlockAllocations = 128 * 1024;
    allocInfo.blockSize           = m_blockBytes;
    allocInfo.maxAllocatedSize    = m_maxSceneBytes;
    allocInfo.keepLastBlock       = true;
    const bool ok = m_dataAllocator.init(allocInfo);
    assert(ok && "BufferSubAllocator::init failed");
    (void)ok;

    m_copyRegions = {};
    m_copyInfos   = {};
    m_copyRegions.reserve(config.maxGroups);
    m_copyInfos.reserve(config.maxGroups);
}

void StreamingStorage::Deinit()
{
    if (m_transferHostMapping && m_device)
    {
        m_device->unmapBuffer(m_transferHostBuffer.GetBuffer());
    }
    m_transferHostMapping = nullptr;
    m_transferHostBuffer.Release();

    m_dataAllocator.deinit();

    m_copyInfos   = {};
    m_copyRegions = {};

    m_device          = nullptr;
    m_descriptorTable = nullptr;
}

size_t StreamingStorage::GetOperationsSize() const
{
    // the geometry storage is not tracked as fixed operations
    return 0;
}

size_t StreamingStorage::GetMaxDataSize() const
{
    return (m_maxSceneBytes / m_blockBytes) * m_blockBytes;
}

StreamingStorage::TaskInfo& StreamingStorage::GetNewTask(uint32_t taskIndex)
{
    TaskInfo& task  = m_taskOperations[taskIndex];
    task.baseOffset = m_maxTransferBytes * taskIndex;
    task.usedMemory = 0;

    m_copyInfos.clear();
    m_copyRegions.clear();

    return task;
}

bool StreamingStorage::CanTransfer(const TaskInfo& task, size_t size) const
{
    return task.usedMemory + size <= m_maxTransferBytes;
}

void* StreamingStorage::AppendTransfer(TaskInfo& task, const rtxmg::BufferSubAllocation& dstHandle, size_t bytes)
{
    assert(task.usedMemory + bytes <= m_maxTransferBytes);

    const size_t transferOffset = task.baseOffset;
    void* const transferPointer = reinterpret_cast<uint8_t*>(m_transferHostMapping) + task.baseOffset;

    task.usedMemory += bytes;
    task.baseOffset += bytes;

    const rtxmg::BufferRange dstRange = m_dataAllocator.subRange(dstHandle);
    nvrhi::IBuffer* const    dstBuffer = dstRange.buffer;
    const uint64_t           dstOffset = dstRange.offset;

    // Extend the last region only when dst AND src both stay contiguous: an
    // AppendHostRead between two transfers takes ring bytes that are not meant
    // for the pool, and extending on dst alone would fold them into the copy.
    if (!m_copyInfos.empty() && m_copyInfos.back().targetBuffer == dstBuffer)
    {
        CopyRegion& lastRegion = m_copyRegions.back();
        if (lastRegion.dstOffset + lastRegion.size == dstOffset
            && lastRegion.srcOffset + lastRegion.size == transferOffset)
        {
            lastRegion.size += bytes;
            return transferPointer;
        }
        // otherwise append new region below
    }
    else
    {
        // new target buffer
        CopyInfo info;
        info.targetBuffer = dstBuffer;
        info.regionOffset = m_copyRegions.size();
        info.regionCount  = 0;
        m_copyInfos.push_back(info);
    }

    {
        // append new region
        CopyRegion region;
        region.srcOffset = transferOffset;
        region.dstOffset = dstOffset;
        region.size      = bytes;

        m_copyInfos.back().regionCount++;
        m_copyRegions.push_back(region);
    }

    return transferPointer;
}

void* StreamingStorage::AppendHostRead(TaskInfo& task, size_t bytes, uint64_t& gpuVA)
{
    // Ring space WITHOUT a pool copy: the GPU consumes transient data straight
    // from the upload heap by VA.  The range lives until the task's fence,
    // which covers the same-frame Implicit CLAS build.
    assert(task.usedMemory + bytes <= m_maxTransferBytes);

    void* const transferPointer =
        reinterpret_cast<uint8_t*>(m_transferHostMapping) + task.baseOffset;
    gpuVA = m_transferHostBuffer.GetBuffer()->getGpuVirtualAddress() + task.baseOffset;

    task.usedMemory += bytes;
    task.baseOffset += bytes;

    return transferPointer;
}

uint32_t StreamingStorage::UploadPendingTransfers(nvrhi::ICommandList* commandList)
{
    // nvrhi has no multi-region copyBuffer — issue one call per region.
    for (const CopyInfo& info : m_copyInfos)
    {
        for (size_t i = 0; i < info.regionCount; ++i)
        {
            const CopyRegion& region = m_copyRegions[info.regionOffset + i];
            commandList->copyBuffer(info.targetBuffer, region.dstOffset,
                                    m_transferHostBuffer.GetBuffer(), region.srcOffset,
                                    region.size);
        }
    }

    // Transition the storage blocks back to a shader-readable state here: every
    // consumer reads these blobs bindlessly (descriptor heap or raw device
    // address), so nvrhi's state tracker never sees a read and would leave them
    // in CopyDest — on Vulkan the transfer writes are then never made available
    // to shaders reading a blob the same frame it was uploaded.
    for (const CopyInfo& info : m_copyInfos)
    {
        commandList->setBufferState(info.targetBuffer,
                                    nvrhi::ResourceStates::ShaderResource);
    }
    commandList->commitBarriers();

    return uint32_t(m_copyRegions.size());
}

void StreamingStorage::Reset()
{
    // Tear down + re-init the allocator (same parameters); resets free-list
    // state and per-block buffers.
    rtxmg::BufferSubAllocator::InitInfo allocInfo;
    allocInfo.device              = m_device;
    allocInfo.descriptorTable     = m_descriptorTable;
    allocInfo.debugName           = "ClusterLodStreamingStorage";
    allocInfo.minAlignment        = rtxmg::BufferSubAllocator::kMinAlignment;
    allocInfo.perBlockAllocations = 128 * 1024;
    allocInfo.blockSize           = m_blockBytes;
    allocInfo.maxAllocatedSize    = m_maxSceneBytes;
    allocInfo.keepLastBlock       = true;
    m_dataAllocator.deinit();
    const bool ok = m_dataAllocator.init(allocInfo);
    assert(ok && "BufferSubAllocator::init failed during reset()");
    (void)ok;
}

bool StreamingStorage::Allocate(rtxmg::BufferSubAllocation& handle, GeometryGroup /*group*/, size_t sz, uint64_t& deviceAddress)
{
    if (!m_dataAllocator.subAllocate(handle, sz))
    {
        deviceAddress = 0;
        return false;
    }
    deviceAddress = m_dataAllocator.subRange(handle).address;
    return true;
}

void StreamingStorage::Free(rtxmg::BufferSubAllocation& handle)
{
    assert(handle);
    m_dataAllocator.subFree(handle);
}

StreamingStorage::PoolStats StreamingStorage::GetStats() const
{
    const rtxmg::BufferSubAllocator::Report report = m_dataAllocator.getReport();
    PoolStats stats;
    stats.reservedDataBytes  = report.reservedSize;
    stats.usedDataBytes      = report.requestedSize;
    stats.allocatedDataBytes = report.allocatedSize;
    return stats;
}

}  // namespace rtxmg
