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

// ClusterLodSystem — the per-frame cluster-LoD sequence.  ClusterLodPass and
// ClusterLodBlasPass are the components; this is the one order their data
// dependencies allow, and the place the ordering contract is written down:
//
//   StageResidencyUpdate -> ApplyResidencyUpdate -> traversal ->
//   FinalizeResidency -> BLAS build -> CaptureFrameRequests -> early submit ->
//   SignalTasksSubmitted
//
// The application owns the residency backend (it is what picks streaming vs
// preload) and hands it over with SetResources; everything else lives here.

#pragma once

#include <cstdint>

#include <nvrhi/nvrhi.h>

#include "rtxmg/cluster_lod/blas_pass.h"
#include "rtxmg/cluster_lod/pass.h"
#include "rtxmg/cluster_lod/resources.h"
#include "rtxmg/cluster_lod/shaderio.h"
#include "rtxmg/cluster_lod/streaming_hooks.h"

class ZBuffer;

namespace rtxmg
{

class ClusterLodSystem
{
public:
    // Per-frame policy.  The caller has already resolved the feature toggles
    // against each other (caching requires sharing, merging requires streaming).
    struct FrameParams
    {
        shaderio::SceneBuildingConstants constants = {};

        // Previous-frame depth pyramid for the traversal cull; see
        // ClusterLodPassParams::zbuffer.
        const ZBuffer* zbuffer          = nullptr;
        bool           useHizOcclusion  = true;

        bool           useStreaming     = true;
        bool           useBlasSharing   = false;
        bool           useBlasCaching   = false;
        bool           useBlasMerging   = false;

        // Cache policy: coarse-tail level count, and the per-frame cluster
        // budget the cached builds are bounded by.
        uint32_t       blasCacheMinLevel    = 0;
        uint32_t       blasCacheMaxClusters = 0;

        bool           debugClusterLod  = false;
    };

    // Non-owning; null until a scene with cluster-LoD geometry is loaded, which
    // is what Update() skips on.
    void SetResources(ClusterLodResources* resources) { m_resources = resources; }
    ClusterLodResources* GetResources() const { return m_resources; }

    // Init the two passes against the current resources before the first
    // Update; ClusterLodBlasPass::Init needs the traversal pass.
    ClusterLodPass&     GetPass()     { return m_pass; }
    ClusterLodBlasPass& GetBlasPass() { return m_blasPass; }

    // Runs LOD traversal and per-instance cluster-LOD BLAS construction.  The
    // caller fills the instance descs and builds the TLAS afterwards.
    // commandList must be open; it is open again on return, but on the
    // streaming path it has been submitted and reopened in between.
    void Update(nvrhi::IDevice*      device,
                nvrhi::ICommandList* commandList,
                const FrameParams&   params);

    // Async-read one frame behind: valid after the second Update.
    const shaderio::SceneBuildingCounters& GetCounters() const { return m_counters; }
    uint64_t GetBlasActualBytes() const { return m_blasActualBytes; }
    uint64_t GetBlasBufferBytes() const { return m_blasPass.GetBlasBufferBytes(); }

    // Traversal + BLAS-build tables, excluding the BLAS storage itself.
    uint64_t GetMetadataBytes() const
    {
        return m_pass.GetMetadataBytes() + m_blasPass.GetMetadataBytes();
    }

    // Log-once state is per scene, not per process: after a scene switch the
    // same dump is worth having again.
    void ResetPerSceneDiagnostics() { m_loggedPreloadGeometry = false; }

private:
    ClusterLodResources* m_resources = nullptr;

    ClusterLodPass     m_pass;
    ClusterLodBlasPass m_blasPass;

    IClusterLodStreamingHooks::FrameSettings m_frameSettings;

    shaderio::SceneBuildingCounters m_counters         = {};
    uint64_t                        m_blasActualBytes  = 0;
    bool                            m_loggedPreloadGeometry = false;
};

}  // namespace rtxmg
