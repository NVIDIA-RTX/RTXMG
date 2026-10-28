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

#include "rtxmg/cluster_lod/preloaded.h"

#include "rtxmg/cluster_lod/group_codec.h"  // DecompressGroup
#include "rtxmg/utils/cluster_operation.h"

#include <donut/core/log.h>
#include <donut/core/math/math.h>
#include <donut/engine/DescriptorTableManager.h>
#include <nvrhi/nvrhi.h>
#include <nvrhi/nvrhiHLSL.h>

#include <cassert>
#include <cstring>

using namespace donut::log;
using namespace donut::math;
using namespace nvrhi::rt;

// ---------------------------------------------------------------------------
// Local helper — expand a baked group blob into its device layout and patch the
// scene-global resident IDs into the group header.
// ---------------------------------------------------------------------------

static void FillGroupRuntimeData(
    const GroupInfo&  srcGroupInfo,
    const GroupView&  srcGroupView,
    uint32_t          clusterResidentID,     // scene-global cluster base id
    uint32_t          groupResidentID,       // scene-global resident group id
    void*             dst,
    size_t            dstSize)
{
    // Compressed groups expand into the uncompressed device layout; uncompressed
    // groups are a straight copy.
    GroupInfo dstGroupInfo = srcGroupInfo;
    if (srcGroupInfo.uncompressedSizeBytes != 0u)
    {
        DecompressGroup(srcGroupInfo, srcGroupView, dst, dstSize);
        dstGroupInfo.sizeBytes       = srcGroupInfo.uncompressedSizeBytes;
        dstGroupInfo.vertexDataCount = srcGroupInfo.uncompressedVertexDataCount;
    }
    else
    {
        assert(srcGroupView.rawSize <= dstSize);
        std::memcpy(dst, srcGroupView.raw, srcGroupView.rawSize);
    }

    GroupStorage gs(dst, dstGroupInfo);
    gs.group->clusterResidentID  = clusterResidentID;
    gs.group->groupResidentID    = groupResidentID;
}

// ---------------------------------------------------------------------------
// ClusterLodPreloaded::Init
// ---------------------------------------------------------------------------

void ClusterLodPreloaded::Init(
    const std::vector<GeometryView>&       geometries,
    const std::vector<ClusterLodInstance>& instances,
    const BakerConfig&                     bakerConfig,
    uint32_t                               clasPositionTruncateBits,
    uint32_t                               clusterLodMaterialBaseID,
    bool                                   hasAlphaMask,
    donut::engine::DescriptorTableManager* descriptorTable,
    nvrhi::IDevice*                        device,
    nvrhi::ICommandList*                   commandList)
{
    m_clusterLodMaterialBaseID = clusterLodMaterialBaseID;
    m_hasAlphaMaskScene        = hasAlphaMask;
    m_clasPositionTruncateBits =
        ResolveClasPositionTruncateBits(clasPositionTruncateBits, bakerConfig);
    const size_t numGeom   = geometries.size();

    if (numGeom == 0)
        return;

    m_shaderGeometries.resize(numGeom);
    m_geometries.resize(numGeom);

    // Bump-allocate scene-global cluster + group resident IDs.
    // Preload mode never evicts, so this is a simple prefix sum.
    m_geomFirstClusterResidentID.assign(numGeom + 1u, 0u);
    m_geomFirstGroupResidentID  .assign(numGeom + 1u, 0u);
    {
        uint32_t clusterRunning = 0;
        uint32_t groupRunning   = 0;
        for (size_t g = 0; g < numGeom; ++g)
        {
            m_geomFirstClusterResidentID[g] = clusterRunning;
            m_geomFirstGroupResidentID  [g] = groupRunning;
            clusterRunning += geometries[g].totalClustersCount;
            groupRunning   += uint32_t(geometries[g].groupInfos.size());
        }
        m_geomFirstClusterResidentID[numGeom] = clusterRunning;
        m_geomFirstGroupResidentID  [numGeom] = groupRunning;
        m_sceneTotalClusters = clusterRunning;
        m_sceneTotalGroups   = groupRunning;
    }

    // Scene-global per-resident-cluster tables.  m_residentClasAddresses is
    // filled in InitClas; m_residentClusters in the per-geom loop below.
    {
        const uint64_t numElems = std::max<uint32_t>(m_sceneTotalClusters, 1u);
        {
            nvrhi::BufferDesc d;
            d.byteSize         = numElems * sizeof(nvrhi::GpuVirtualAddress);
            d.structStride     = sizeof(nvrhi::GpuVirtualAddress);
            d.canHaveUAVs      = true;
            d.canHaveRawViews  = true;
            d.initialState     = nvrhi::ResourceStates::Common;
            d.keepInitialState = true;
            d.debugName        = "ClusterLodResidentClasAddresses";
            m_residentClasAddresses.Create(d, device);
        }
        {
            nvrhi::BufferDesc d;
            d.byteSize         = numElems * sizeof(shaderio::ClusterAddress);
            d.structStride     = sizeof(shaderio::ClusterAddress);
            d.canHaveUAVs      = true;
            d.canHaveRawViews  = true;
            d.initialState     = nvrhi::ResourceStates::Common;
            d.keepInitialState = true;
            d.debugName        = "ClusterLodResidentClusters";
            m_residentClusters.Create(d, device);
        }

        // Preload never runs the age filter, so the contents are ignored — but
        // traversal_run_groups writes age=0 per visited group, so the table must
        // still cover every Group.groupResidentID.
        {
            uint32_t totalGroups = m_geomFirstGroupResidentID.empty()
                                 ? 0u
                                 : m_geomFirstGroupResidentID.back();
            nvrhi::BufferDesc d;
            d.byteSize         = uint64_t(std::max(totalGroups, 1u)) * sizeof(shaderio::StreamingGroup);
            d.structStride     = sizeof(shaderio::StreamingGroup);
            d.canHaveUAVs      = true;
            d.canHaveRawViews  = true;
            d.initialState     = nvrhi::ResourceStates::Common;
            d.keepInitialState = true;
            d.debugName        = "ClusterLodPreloadResidentGroups";
            m_residentGroupsBufferBase.Create(d, device);
        }
    }

    // Traversal iterates the flat render-instance table per geometry, so each
    // shaderio::Geometry needs an instancesOffset/instancesCount slice.
    std::vector<uint32_t> instanceCount(numGeom, 0u);
    for (const ClusterLodInstance& inst : instances)
        if (inst.geometryID < numGeom)
            ++instanceCount[inst.geometryID];

    uint32_t instancesOffset = 0;

    for (size_t g = 0; g < numGeom; ++g)
    {
        const GeometryView& geom        = geometries[g];
        shaderio::Geometry& shaderGeom  = m_shaderGeometries[g];
        PreloadGeometry&    preloadGeom = m_geometries[g];

        // LoD tree, base shaderio::Geometry fields and base SRVs — shared with
        // the streaming path.
        UploadGeometryMetadata(g, geom, preloadGeom, shaderGeom,
                               instancesOffset, instanceCount[g]);

        // ---- Group-data buffer (cluster blobs) --------------------------------
        {
            // Blobs are expanded into the uncompressed device layout as they are
            // filled, so a compressed bake needs more room than the stored span.
            size_t groupDataSize = 0;
            for (const GroupInfo& info : geom.groupInfos)
                groupDataSize += info.GetDeviceSize();

            nvrhi::BufferDesc groupDataDesc = nvrhi::BufferDesc()
                .setByteSize(std::max(groupDataSize, size_t(1)))
                .setDebugName("ClusterLodGroupData")
                .setCanHaveUAVs(true)
                .setCanHaveRawViews(true)
                .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
                .setKeepInitialState(true);
            preloadGeom.groupData.Create(groupDataDesc, device);

            // Each blob must carry scene-global resident IDs (same convention as
            // streaming) so traversal_run_groups' `clusterResidentID + c` indexes
            // the global resident tables.
            std::vector<uint8_t> stagingGroupData(groupDataSize);
            size_t   dstOffset    = 0;
            uint32_t clusterOffset = 0;
            const uint32_t geomClusterBase = m_geomFirstClusterResidentID[g];

            const uint32_t geomGroupBase   = m_geomFirstGroupResidentID  [g];
            for (uint32_t gi = 0; gi < static_cast<uint32_t>(geom.groupInfos.size()); ++gi)
            {
                const GroupInfo groupInfo = geom.groupInfos[gi];
                const GroupView groupView(geom.groupData, groupInfo);
                const size_t    devSize   = groupInfo.GetDeviceSize();

                FillGroupRuntimeData(
                    groupInfo, groupView,
                    geomClusterBase + clusterOffset,       // clusterResidentID (scene-global)
                    geomGroupBase + gi,                    // groupResidentID (scene-global)
                    stagingGroupData.data() + dstOffset,
                    devSize);

                dstOffset     += devSize;
                clusterOffset += groupInfo.clusterCount;
            }
            if (groupDataSize > 0)
                commandList->writeBuffer(preloadGeom.groupData, stagingGroupData.data(), groupDataSize);
        }

        // ---- Per-cluster addresses into groupData for the hit shader ---------
        // The hit shader reads vertices/indices out of the same groupData blob
        // the CLAS-build hardware reads, so publish the bindless SRV slot plus
        // each Cluster header's byte offset.
        {
            const size_t numClusters = geom.totalClustersCount;

            // In preload mode the whole geometry shares one groupData buffer.
            preloadGeom.groupDataSRVHandle = descriptorTable->CreateDescriptorHandle(
                nvrhi::BindingSetItem::RawBuffer_SRV(0, preloadGeom.groupData));
            const uint32_t groupDataBlockIdx = static_cast<uint32_t>(
                preloadGeom.groupDataSRVHandle.GetIndexInHeap());

            std::vector<shaderio::ClusterAddress> clusterFlat(numClusters);
            std::vector<shaderio::GroupAddress> groupAddresses(geom.groupInfos.size());
            size_t   groupDataOffset = 0;
            uint32_t clusterOffset   = 0;
            for (uint32_t gi = 0; gi < static_cast<uint32_t>(geom.groupInfos.size()); ++gi)
            {
                const GroupInfo groupInfo = geom.groupInfos[gi];
                assert(groupDataOffset <= 0xFFFFFFFFull
                       && "GroupAddress uses uint32 byte offsets into groupData");
                groupAddresses[gi] = {
                    groupDataBlockIdx,
                    uint32_t(groupDataOffset),
                };

                for (uint32_t c = 0; c < groupInfo.clusterCount; ++c)
                {
                    const uint32_t clusterHeaderOffset =
                        uint32_t(groupDataOffset)
                        + uint32_t(sizeof(shaderio::Group))
                        + uint32_t(sizeof(shaderio::Cluster)) * c;
                    clusterFlat[clusterOffset + c] = {
                        groupDataBlockIdx,
                        clusterHeaderOffset,
                    };
                }

                clusterOffset   += groupInfo.clusterCount;
                groupDataOffset += groupInfo.GetDeviceSize();
            }
            assert(groupDataOffset <= 0xFFFFFFFFull
                   && "ClusterAddress uses uint32 byte offsets into groupData");

            // The global table is indexed by clusterResidentID, so write at the
            // geometry's bump-allocated base.
            if (numClusters > 0 && m_residentClusters.GetBuffer())
            {
                const uint32_t geomClusterBase = m_geomFirstClusterResidentID[g];
                commandList->writeBuffer(
                    m_residentClusters.GetBuffer(),
                    clusterFlat.data(),
                    numClusters * sizeof(shaderio::ClusterAddress),
                    uint64_t(geomClusterBase) * sizeof(shaderio::ClusterAddress));
            }

            if (!groupAddresses.empty())
                commandList->writeBuffer(preloadGeom.streamingGroupAddresses.buffer,
                                         groupAddresses.data(),
                                         groupAddresses.size() * sizeof(shaderio::GroupAddress),
                                         preloadGeom.streamingGroupAddresses.ByteOffset());
        }

        // UploadGeometryMetadata leaves lowDetailClusterID as a per-geom local
        // offset; re-base it onto the scene-global ID space every consumer uses.
        if (geom.lodLevelsCount > 0)
        {
            const shaderio::LodLevel& lastLevel = geom.lodLevels[geom.lodLevelsCount - 1];
            shaderGeom.lowDetailClusterID =
                m_geomFirstClusterResidentID[g] + lastLevel.clusterOffset;
        }

        instancesOffset += shaderGeom.instancesCount;
    }

    // ---- Build and upload render-instance table ------------------------------
    {
        const size_t numInst = instances.size();
        m_renderInstances.resize(numInst);

        for (size_t i = 0; i < numInst; ++i)
        {
            const ClusterLodInstance& src = instances[i];
            shaderio::RenderInstance& dst = m_renderInstances[i];

            const affine3 xf = homogeneousToAffine(src.transform);
            affineToColumnMajor(xf, dst.worldMatrix.m_data);
            // Affine inverse for the USE_BLAS_SHARING object-space camera transform.
            affineToColumnMajor(inverse(xf), dst.worldMatrixI.m_data);
            dst.geometryID = src.geometryID;
            // materialID is the fallback "instance material" for single-material
            // geometries; multi-material ones go through resolveMaterialID()'s
            // materialBaseID + localID indirection instead.
            const GeometryView& geo = geometries[src.geometryID];
            // Without materials the scene pushed only a default-gray slot at
            // m_clusterLodMaterialBaseID, so the per-instance offset is dropped.
            dst.materialID                = m_enableMaterials
                                                ? (m_clusterLodMaterialBaseID + src.materialID)
                                                : m_clusterLodMaterialBaseID;
            if (m_enableMaterials)
            {
                dst.multiMaterial             = (geo.localMaterialIDs.size() > 1) ? 1u : 0u;
                dst.lowDetailClusterStateBits = geo.lowDetailClusterStateBits;
                // opaqueStatus: aggregate the geometry's per-material alpha bits.
                bool anyAlpha = false;
                bool allAlpha = !geo.localMaterialStateBits.empty();
                for (uint8_t mb : geo.localMaterialStateBits)
                {
                    const bool a = (mb & shaderio::ClusterState::AlphaMasked) != 0u;
                    anyAlpha = anyAlpha || a;
                    allAlpha = allAlpha && a;
                }
                dst.opaqueStatus = !anyAlpha ? uint32_t(shaderio::OpaqueStatus::Opaque)
                                              : (allAlpha ? uint32_t(shaderio::OpaqueStatus::AlphaMasked)
                                                          : uint32_t(shaderio::OpaqueStatus::Mixed));
            }
            else
            {
                dst.multiMaterial             = 0u;
                dst.lowDetailClusterStateBits = 0u;
                dst.opaqueStatus              = uint32_t(shaderio::OpaqueStatus::Opaque);
            }
            dst._stateReserved = 0;
            dst._pad2          = 0;
        }

        if (numInst > 0)
        {
            m_renderInstancesBuffer.Create(numInst, "ClusterLodRenderInstances", device);
            commandList->writeBuffer(m_renderInstancesBuffer,
                                     m_renderInstances.data(),
                                     numInst * sizeof(shaderio::RenderInstance));
        }
    }

    // ---- Upload shader-geometry table ----------------------------------------
    m_shaderGeometriesBuffer.Create(numGeom, "ClusterLodShaderGeometries", device);
    commandList->writeBuffer(m_shaderGeometriesBuffer,
                             m_shaderGeometries.data(),
                             numGeom * sizeof(shaderio::Geometry));

    // ---- Build CLASes and low-detail BLASes ----------------------------------
    InitClas(geometries, descriptorTable, device, commandList);
}

// ---------------------------------------------------------------------------
// ClusterLodPreloaded::LogGeometryData
// ---------------------------------------------------------------------------

void ClusterLodPreloaded::LogGeometryData() const
{
    info("--- ClusterLodPreloaded geometry data ---");
    info("  maxNodeTreeDepth=%u (drives multipass traversal_run dispatch count)",
         m_maxNodeTreeDepth);
    for (size_t g = 0; g < m_shaderGeometries.size(); ++g)
    {
        const shaderio::Geometry& geom = m_shaderGeometries[g];
        info("  geom[%zu]: lodLevelsCount=%u lowDetailClusterID=%u lowDetailBlasAddress=0x%llx",
             g, geom.lodLevelsCount, geom.lowDetailClusterID, (unsigned long long)geom.lowDetailBlasAddress);
        info("  geom[%zu]: nodesSRV=%u streamingGroupAddressesSRV=%u",
             g, geom.nodesSRV, geom.streamingGroupAddressesSRV);
        info("  geom[%zu]: instancesOffset=%u instancesCount=%u",
             g, geom.instancesOffset, geom.instancesCount);

        info("  geom[%zu]: lodNodes count=%u lodLevels count=%u groupAddresses count=%u",
             g,
             m_geometries[g].lodNodes.numElements,
             m_geometries[g].lodLevels.numElements,
             m_geometries[g].streamingGroupAddresses.numElements);
    }
    info("  numRenderInstances=%zu", m_renderInstances.size());
    for (size_t i = 0; i < m_renderInstances.size(); ++i)
        info("  inst[%zu]: geometryID=%u", i, m_renderInstances[i].geometryID);
}

// ---------------------------------------------------------------------------
// ClusterLodPreloaded::InitClas
// ---------------------------------------------------------------------------

void ClusterLodPreloaded::InitClas(
    const std::vector<GeometryView>&       geometries,
    donut::engine::DescriptorTableManager* descriptorTable,
    nvrhi::IDevice*                        device,
    nvrhi::ICommandList*                   commandList)
{
    const size_t numGeom = geometries.size();

    // BLAS build args (one per geometry); filled incrementally below.
    std::vector<cluster::IndirectArgs> blasArgs(numGeom);

    for (size_t g = 0; g < numGeom; ++g)
    {
        const GeometryView& geom       = geometries[g];
        shaderio::Geometry& shaderGeom = m_shaderGeometries[g];
        PreloadGeometry&    pg         = m_geometries[g];

        const uint32_t numClusters = geom.totalClustersCount;
        if (numClusters == 0)
            continue;

        // CLAS addresses land in the scene-global m_residentClasAddresses at
        // [geomClusterBase, +numClusters); sizes are only needed transiently.
        const uint32_t geomClusterBase = m_geomFirstClusterResidentID[g];
        ClusterOperationSizesBuffer tempSizes;
        tempSizes.Create(numClusters, "ClusterLodPreloadClasSizesTemp", device);

        // -- Build IndirectTriangleClasArgs[] on CPU --------------------------
        // GeometryView's own max/total fields can be 0 when it came from cache,
        // so recompute them from the cluster headers for the OperationParams.
        uint32_t maxTrianglesPerCluster = 0;
        uint32_t maxVerticesPerCluster  = 0;
        uint32_t totalTriangles         = 0;
        uint32_t totalVertices          = 0;

        std::vector<cluster::nv::IndirectTriangleClasArgs> clasArgs(numClusters);

        // Per-triangle geometryIndexAndFlags for every mixed cluster in this
        // geometry, concatenated; each cluster's arg points at its own slice,
        // patched in a second pass once the GPU VA exists.
        std::vector<nvrhi::rt::cluster::GeometryIndexAndFlags> clasGeometryIndicesHost;
        std::vector<uint32_t> mixedClusterOffsets(numClusters, ~0u);
        clasGeometryIndicesHost.reserve(64);  // grows as mixed clusters arrive

        {
            const nvrhi::GpuVirtualAddress groupDataBaseVA =
                pg.groupData.GetGpuVirtualAddress();

            size_t   groupDataOffset = 0;
            uint32_t clusterOffset   = 0;

            std::vector<uint8_t> devBlob;  // decompression scratch, reused per group

            for (uint32_t gi = 0; gi < static_cast<uint32_t>(geom.groupInfos.size()); ++gi)
            {
                const GroupInfo groupInfo = geom.groupInfos[gi];
                const GroupView srcGroupView(geom.groupData, groupInfo);

                // The CLAS args address the device blob, which always holds the
                // uncompressed layout. DecompressGroup re-points each cluster's
                // triangle offset, so a compressed group's stored headers do not
                // describe what the GPU reads.
                GroupView groupView = srcGroupView;
                if (groupInfo.uncompressedSizeBytes != 0u)
                {
                    GroupInfo devInfo       = groupInfo;
                    devInfo.offsetBytes     = 0;
                    devInfo.sizeBytes       = groupInfo.uncompressedSizeBytes;
                    devInfo.vertexDataCount = groupInfo.uncompressedVertexDataCount;

                    devBlob.assign(groupInfo.GetDeviceSize(), uint8_t(0));
                    DecompressGroup(groupInfo, srcGroupView, devBlob.data(), devBlob.size());
                    groupView = GroupView(devBlob, devInfo);
                }

                const nvrhi::GpuVirtualAddress groupVA =
                    groupDataBaseVA + groupDataOffset;

                for (uint32_t c = 0; c < groupInfo.clusterCount; ++c)
                {
                    const shaderio::Cluster& cl = groupView.clusters[c];
                    const uint32_t triCount = cl.triangleCountMinusOne + 1u;
                    const uint32_t vtxCount = cl.vertexCountMinusOne  + 1u;

                    maxTrianglesPerCluster = std::max(maxTrianglesPerCluster, triCount);
                    maxVerticesPerCluster  = std::max(maxVerticesPerCluster,  vtxCount);
                    totalTriangles        += triCount;
                    totalVertices         += vtxCount;

                    const nvrhi::GpuVirtualAddress clusterVA =
                        groupVA
                        + sizeof(shaderio::Group)
                        + sizeof(shaderio::Cluster) * c;

                    cluster::nv::IndirectTriangleClasArgs& arg = clasArgs[clusterOffset + c];
                    arg = {};
                    // Scene-global, so the hit shader's GetClusterID() indexes
                    // the global resident tables directly.
                    arg.clusterId                    = geomClusterBase + clusterOffset + c;
                    arg.clusterFlags                 = 0;
                    arg.triangleCount                = triCount;
                    arg.vertexCount                  = vtxCount;
                    arg.positionTruncateBitCount     = m_clasPositionTruncateBits;
                    arg.indexFormat                  = uint32_t(cluster::OperationIndexFormat::IndexFormat8bit);
                    arg.opacityMicromapIndexFormat   = 0;
                    arg.indexBufferStride            = 1;           // uint8 indices
                    arg.vertexBufferStride           = uint16_t(sizeof(float3));  // 12
                    arg.indexBuffer                  = clusterVA + cl.triangles;
                    arg.vertexBuffer                 = clusterVA + cl.vertices;
                    arg.opacityMicromapArray         = 0;
                    arg.opacityMicromapIndexBuffer   = 0;

                    // Without materials everything is opaque/single-sided/mat 0.
                    // The alpha-mask geometry slot only exists when the scene has
                    // alpha-masked materials (maxGeometryIndex 0 otherwise).
                    const uint8_t stateBitsMask =
                        m_hasAlphaMaskScene ? uint8_t(~0u) : uint8_t(~shaderio::ClusterState::AlphaMasked);
                    const uint8_t effStateBits = (m_enableMaterials ? cl.stateBits : uint8_t(0)) & stateBitsMask;
                    const uint8_t effLocalMat  = m_enableMaterials ? cl.localMaterialID : uint8_t(0);
                    const bool requiresMixedGeometryBuffer =
                        (effStateBits & shaderio::ClusterState::AlphaMaskedMixed) != 0u ||
                        (effStateBits & shaderio::ClusterState::TwoSidedMixed)    != 0u;
                    if (requiresMixedGeometryBuffer)
                    {
                        // Uniform "base" cleared; per-triangle entries take
                        // precedence via geometryIndexAndFlagsBuffer.
                        arg.baseGeometryIndexAndFlags         = {};
                        arg.geometryIndexAndFlagsBufferStride = uint16_t(sizeof(nvrhi::rt::cluster::GeometryIndexAndFlags));
                        mixedClusterOffsets[clusterOffset + c] =
                            static_cast<uint32_t>(clasGeometryIndicesHost.size());

                        // Per-triangle material bytes follow the index bytes
                        // in the cluster's triangle payload — same layout the
                        // baker emits.
                        const uint8_t* clusterMaterialBytes =
                            groupView.GetClusterIndices(c) + size_t(triCount) * 3u;
                        if (effLocalMat == shaderio::kPerTriangleMaterials)
                        {
                            for (uint32_t t = 0; t < triCount; ++t)
                                clasGeometryIndicesHost.push_back(
                                    ClasEncodePerTriangleGeometryIndexAndFlags(clusterMaterialBytes[t]));
                        }
                        else
                        {
                            // _MIXED state without per-triangle bytes would
                            // be a baker bug — fall back to uniform encoding
                            // for every triangle.
                            const auto uniformEncoded =
                                ClasEncodeBaseGeometryIndexAndFlagsFromState(effStateBits);
                            for (uint32_t t = 0; t < triCount; ++t)
                                clasGeometryIndicesHost.push_back(uniformEncoded);
                        }
                        // arg.geometryIndexAndFlagsBuffer patched below after
                        // pg.clasGeometryIndices is created.
                    }
                    else
                    {
                        arg.baseGeometryIndexAndFlags         =
                            ClasEncodeBaseGeometryIndexAndFlagsFromState(effStateBits);
                        arg.geometryIndexAndFlagsBufferStride = 0;
                        arg.geometryIndexAndFlagsBuffer       = 0;
                    }

                }

                clusterOffset   += groupInfo.clusterCount;
                groupDataOffset += groupInfo.GetDeviceSize();
            }
        }

        // Upload the mixed-cluster geometry-indices buffer, then patch the args
        // that point into it now that its GPU VA exists.
        if (!clasGeometryIndicesHost.empty())
        {
            pg.clasGeometryIndices.Create(uint32_t(clasGeometryIndicesHost.size()),
                                          "ClusterLodClasGeometryIndices", device);
            commandList->writeBuffer(pg.clasGeometryIndices,
                                     clasGeometryIndicesHost.data(),
                                     clasGeometryIndicesHost.size() * sizeof(nvrhi::rt::cluster::GeometryIndexAndFlags));
            const nvrhi::GpuVirtualAddress clasGeomVA =
                pg.clasGeometryIndices.GetGpuVirtualAddress();
            for (uint32_t c = 0; c < numClusters; ++c)
            {
                if (mixedClusterOffsets[c] != ~0u)
                    clasArgs[c].geometryIndexAndFlagsBuffer =
                        clasGeomVA + uint64_t(mixedClusterOffsets[c]) * sizeof(nvrhi::rt::cluster::GeometryIndexAndFlags);
            }
        }

        // Upload CLAS build args.
        nvrhi::BufferHandle clasArgsBuffer = CreateAndUploadClusterArgs(
            clasArgs,
            GetGenericDesc(numClusters, uint32_t(sizeof(cluster::nv::IndirectTriangleClasArgs)), "ClusterLodClasArgsBuffer")
                .setIsAccelStructBuildInput(true),
            commandList);

        // -- GetSizes pass: query per-CLAS sizes --------------------------------
        cluster::OperationParams clasParams = {};
        clasParams.maxArgCount = numClusters;
        clasParams.type        = cluster::OperationType::ClasBuild;
        clasParams.mode        = cluster::OperationMode::GetSizes;
        clasParams.flags       = cluster::OperationFlags::FastTrace;
        clasParams.clas.vertexFormat              = nvrhi::Format::RGB32_FLOAT;
        // Alpha-mask scenes use two geometry-index slots (0 = opaque,
        // 1 = alpha-mask), which the CLAS-build hardware must reserve room for.
        // maxUniqueGeometryCount bounds packed *elements*, not index values, and the
        // encoder varies CullDisable independently of the alpha slot -- a mixed
        // cluster can emit {0,Opaque}, {0,Opaque|CullDisable}, {1,None} and
        // {1,CullDisable}.  Sidedness has no scene-wide flag, so assume it varies.
        clasParams.clas.maxGeometryIndex          = m_hasAlphaMaskScene ? 1u : 0u;
        clasParams.clas.maxUniqueGeometryCount    = m_hasAlphaMaskScene ? 4u : 2u;
        clasParams.clas.maxTriangleCount          = maxTrianglesPerCluster;
        clasParams.clas.maxVertexCount            = maxVerticesPerCluster;
        clasParams.clas.maxTotalTriangleCount     = totalTriangles;
        clasParams.clas.maxTotalVertexCount       = totalVertices;
        clasParams.clas.minPositionTruncateBitCount = m_clasPositionTruncateBits;
        clasParams.clas.indexFormat                 = cluster::OperationIndexFormat::IndexFormat8bit;

        cluster::OperationSizeInfo sizeInfo = device->getClusterOperationSizeInfo(clasParams);

        {
            cluster::OperationDesc getSizesDesc = {};
            getSizesDesc.params           = clasParams;
            getSizesDesc.scratchSizeInBytes = sizeInfo.scratchSizeInBytes;
            getSizesDesc.inIndirectArgsBuffer = clasArgsBuffer;
            // Temp sizes buffer, discarded at end of InitClas.
            getSizesDesc.outSizesBuffer       = tempSizes;

            commandList->executeMultiIndirectClusterOperation(getSizesDesc);
        }

        // Download sizes (closes + submits + reopens commandList).
        const std::vector<uint64_t> clasSizes = tempSizes.Download(commandList);

        // -- Allocate clasData and compute per-cluster CLAS addresses ----------
        //    The build consumes addresses contiguously from offset 0, so it gets
        //    a per-geometry temp buffer; the same addresses are also written into
        //    the scene-global table at geomClusterBase.
        RTXMGBuffer<nvrhi::GpuVirtualAddress> tempAddrs;
        tempAddrs.Create(numClusters, "ClusterLodPreloadClasAddrsTemp", device);
        {
            uint64_t totalClasSize = 0;
            for (uint64_t size : clasSizes)
            {
                assert(size && "CLAS with invalid size");
                totalClasSize += size;
            }

            nvrhi::BufferDesc clasDataDesc = nvrhi::BufferDesc()
                .setByteSize(totalClasSize)
                .setDebugName("ClusterLodClasData")
                .setCanHaveUAVs(true)
                .setIsAccelStructStorage(true)
                .setInitialState(nvrhi::ResourceStates::AccelStructWrite)
                .setKeepInitialState(true);
            pg.clasData.Create(clasDataDesc, device);

            std::vector<nvrhi::GpuVirtualAddress> clasAddrCpu(numClusters);
            const nvrhi::GpuVirtualAddress clasDataBase = pg.clasData.GetGpuVirtualAddress();
            uint64_t clasOffset = 0;
            for (uint32_t c = 0; c < numClusters; ++c)
            {
                clasAddrCpu[c]  = clasDataBase + clasOffset;
                clasOffset     += clasSizes[c];
            }
            commandList->writeBuffer(tempAddrs,
                                     clasAddrCpu.data(),
                                     numClusters * sizeof(nvrhi::GpuVirtualAddress));
            if (m_residentClasAddresses.GetBuffer())
            {
                commandList->writeBuffer(
                    m_residentClasAddresses.GetBuffer(),
                    clasAddrCpu.data(),
                    numClusters * sizeof(nvrhi::GpuVirtualAddress),
                    uint64_t(geomClusterBase) * sizeof(nvrhi::GpuVirtualAddress));
            }
        }

        // -- ExplicitDestinations pass: build the CLASes -----------------------
        // outAccelerationStructuresBuffer is left null: ExplicitDestinations mode
        // writes each CLAS directly to the VA in inOutAddressesBuffer (pg.clasData).
        {
            clasParams.mode = cluster::OperationMode::ExplicitDestinations;

            cluster::OperationDesc buildDesc = {};
            buildDesc.params             = clasParams;
            buildDesc.scratchSizeInBytes = sizeInfo.scratchSizeInBytes;
            buildDesc.inIndirectArgsBuffer   = clasArgsBuffer;
            buildDesc.inOutAddressesBuffer   = tempAddrs;

            commandList->executeMultiIndirectClusterOperation(buildDesc);
        }

        // -- Set up low-detail BLAS arg ----------------------------------------
        //    Source address = &m_residentClasAddresses[lowDetailClusterID], which
        //    Init already re-based onto the scene-global ID space.
        //    A root group may hold more than one cluster; taking only the first
        //    drops the rest from every instance that falls back to this BLAS,
        //    which streaming does not do (it sizes from the group's clusterCount).
        {
            const uint32_t lowDetailClusterID = shaderGeom.lowDetailClusterID;
            uint32_t       lowDetailClusters  = 1;
            if (geom.lodLevelsCount > 0)
            {
                const shaderio::LodLevel& lastLevel = geom.lodLevels[geom.lodLevelsCount - 1];
                lowDetailClusters = uint32_t(geom.groupInfos[lastLevel.groupOffset].clusterCount);
            }
            blasArgs[g].clusterCount    = lowDetailClusters;
            blasArgs[g].clusterAddressStride = 8;
            blasArgs[g].clusterAddresses =
                (m_residentClasAddresses.GetBuffer() ? m_residentClasAddresses.GetGpuVirtualAddress() : 0ull)
                + static_cast<uint64_t>(lowDetailClusterID) * sizeof(nvrhi::GpuVirtualAddress);
        }
    }

    // ---- Build low-detail BLASes for all geometries -------------------------
    {
        // Upload blas args buffer.
        RTXMGBuffer<cluster::IndirectArgs> blasArgsBuffer;
        {
            auto desc = GetGenericDesc(numGeom,
                                       uint32_t(sizeof(cluster::IndirectArgs)),
                                       "ClusterLodBlasArgsBuffer")
                            .setIsAccelStructBuildInput(true);
            blasArgsBuffer.Create(desc, device);
            commandList->writeBuffer(blasArgsBuffer, blasArgs.data(),
                                     numGeom * sizeof(cluster::IndirectArgs));
        }

        cluster::OperationParams blasParams = {};
        blasParams.maxArgCount             = static_cast<uint32_t>(numGeom);
        blasParams.type                    = cluster::OperationType::BlasBuild;
        blasParams.mode                    = cluster::OperationMode::ImplicitDestinations;
        blasParams.flags                   = cluster::OperationFlags::FastTrace;
        blasParams.blas.maxClasPerBlasCount = 1;
        blasParams.blas.maxTotalClasCount   = static_cast<uint32_t>(numGeom);

        cluster::OperationSizeInfo blasSizeInfo =
            device->getClusterOperationSizeInfo(blasParams);

        // Create the bulk BLAS storage buffer.
        {
            nvrhi::BufferDesc blasDesc = nvrhi::BufferDesc()
                .setByteSize(blasSizeInfo.resultMaxSizeInBytes)
                .setDebugName("ClusterLodLowDetailBlas")
                .setCanHaveUAVs(true)
                .setIsAccelStructStorage(true)
                .setInitialState(nvrhi::ResourceStates::AccelStructWrite)
                .setKeepInitialState(true);
            m_clasLowDetailBlasBuffer.Create(blasDesc, device);
        }

        // Buffer to receive per-geometry BLAS addresses after ImplicitDestinations build.
        RTXMGBuffer<nvrhi::GpuVirtualAddress> blasAddressesBuffer;
        blasAddressesBuffer.Create(numGeom, "ClusterLodBlasAddresses", device);

        // Optional sizes buffer (required by some nvrhi backends for ImplicitDestinations).
        ClusterOperationSizesBuffer blasSizesBuffer;
        blasSizesBuffer.Create(numGeom, "ClusterLodBlasSizes", device);

        {
            cluster::OperationDesc blasBuildDesc = {};
            blasBuildDesc.params             = blasParams;
            blasBuildDesc.scratchSizeInBytes = blasSizeInfo.scratchSizeInBytes;
            blasBuildDesc.inIndirectArgsBuffer         = blasArgsBuffer;
            blasBuildDesc.inOutAddressesBuffer         = blasAddressesBuffer;
            blasBuildDesc.outSizesBuffer               = blasSizesBuffer;
            blasBuildDesc.outAccelerationStructuresBuffer = m_clasLowDetailBlasBuffer;

            commandList->executeMultiIndirectClusterOperation(blasBuildDesc);
        }

        // Read back BLAS addresses and store in shaderGeometries.
        const std::vector<nvrhi::GpuVirtualAddress> blasAddrs =
            blasAddressesBuffer.Download(commandList);

        for (size_t g = 0; g < numGeom; ++g)
            m_shaderGeometries[g].lowDetailBlasAddress = blasAddrs[g];

        // Re-upload the updated shaderGeometries (now with lowDetailBlasAddress filled).
        commandList->writeBuffer(m_shaderGeometriesBuffer,
                                 m_shaderGeometries.data(),
                                 numGeom * sizeof(shaderio::Geometry));
    }
}
