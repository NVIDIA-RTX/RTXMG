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
//

#pragma once

#include <donut/core/math/math.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/View.h>
#include <donut/render/ToneMappingPasses.h>
#include <nvrhi/nvrhi.h>

#include "render_params.h"
#include "blit_params.h"
#include "motion_vectors_params.h"
#include "render_targets.h"
#include "cluster_lod_budgets.h"

#include "rtxmg/cluster_lod/adaptive_lod_error.h"
#include "rtxmg/cluster_lod/blas_pass.h"
#include "rtxmg/cluster_lod/blas_build_params.h"
#include "rtxmg/cluster_lod/pass.h"
#include "rtxmg/cluster_lod/preloaded.h"
#include "rtxmg/cluster_lod/resources.h"
#include "rtxmg/cluster_lod/streaming.h"
#include "rtxmg/cluster_lod/system.h"
#include "rtxmg/cluster_tess/cluster_tessellator.h"
#include "envmap/preprocess_envmap.h"
#include "envmap/scan_system.h"
#include "rtxmg/hiz/zbuffer.h"
#include "zrenderer.h"
#include "rtxmg/scene/camera.h"
#include "rtxmg/utils/buffer.h"
#include "rtxmg/utils/pixel_pick.h"
#include "rtxmg/utils/shader_debug.h"

using namespace donut::engine;
using namespace donut::math;

typedef vector<unsigned short, 2> uint16_t2;
typedef vector<unsigned short, 4> uint16_t4;

class RTXMGScene;
class Camera;

#define cStablePlaneCount (3u)

class RTXMGRenderer
{
public:
    struct Options
    {
        RenderParams& params;
        nvrhi::IDevice* device;
        // Path trace with the DXR cluster intrinsics even when NVAPI builds the clusters.
        bool forceDxrClusterIntrinsics = false;
    };

    enum class OutputTexture : uint32_t
    {
        DlssOutputColor,
        Accumulation,
        Depth,
        Normals,
        Albedo,
        Specular,
        SpecularHitT,
        Roughness,
        MotionVectors,
#if ENABLE_DUMP_FLOAT
        Debug1,
        Debug2,
        Debug3,
        Debug4,
#endif
        Count
    };

    enum class Output : uint32_t
    {
        DlssOutputColor,
        Accumulation,
        Depth,
        Normals,
        Albedo,
        Specular,
        SpecularHitT,
        Roughness,
        MotionVectors,
#if ENABLE_DUMP_FLOAT
        Debug1,
        Debug2,
        Debug3,
        Debug4,
#endif
        OutputTextureCount,
        InstanceId = OutputTextureCount,
        SurfaceIndex,
        SurfaceUv,
        Texcoord,
        HiZ,
        Count
    };

    // Output is a superset of OutputTexture since we need to be able to visualize both textures/buffers
    static_assert(uint32_t(Output::OutputTextureCount) == uint32_t(OutputTexture::Count));

    RTXMGRenderer(Options const& opts);
    ~RTXMGRenderer();

    // Frame order: RetireAccelResources -> RenderHiZPrepass ->
    // UpdateAccelerationStructures.  Retiring must precede the prepass, which
    // traces the previous frame's TLAS; see the definitions.
    void RetireAccelResources(nvrhi::ICommandList* commandList,
                              const TessellatorConfig* tessConfig);
    void RenderHiZPrepass(Camera& camera, nvrhi::ICommandList* commandList);

    void UpdateAccelerationStructures(const TessellatorConfig& tessConfig,
        ClusterTessStatistics& buildStats,
        uint32_t frameIndex,
        nvrhi::ICommandList* commandList);
    void ReloadShaders();

    void CreateOutputs(nvrhi::ICommandList* commandList);
    void Launch(nvrhi::ICommandList* commandList, uint32_t frameIndex,
        std::shared_ptr<Light> light);
    void BlitFramebuffer(nvrhi::ICommandList* commandList, nvrhi::IFramebuffer* framebuffer);
    void DlssUpscale(nvrhi::ICommandList* commandList, uint32_t frameIndex);

    void ResetSubframes();
    void ForceResetSubframes();
    // Subframes accumulated since the last force-reset.  Note ResetSubframes is
    // a no-op while the denoiser is on, so this only restarts at a force-reset.
    uint32_t GetSubframeIndex() const { return m_params.subFrameIndex; }
    void ResetDenoiser() { m_resetDenoiser = true; }
    void SetRenderCamera(Camera& camera, bool isCameraCut);

    void SetShadingMode(ShadingMode shadingMode)
    {
        m_shadingMode = shadingMode;
        ResetSubframes();
        ResetDenoiser();
    }
    ShadingMode GetShadingMode() const { return m_shadingMode; }
    ShadingMode GetEffectiveShadingMode() const { return m_showMicroTriangles ? ShadingMode::PRIMARY_RAYS : m_shadingMode; }

    void SetColorMode(ColorMode colorMode)
    {
        m_colorMode = colorMode;
        ResetSubframes();
        ResetDenoiser();
    }
    ColorMode GetColorMode() const { return m_colorMode; }

    void SetOutputIndex(Output output)
    {
        m_outputIndex = output;
    }
    Output GetOutputIndex() const { return m_outputIndex; }

    const char* GetOutputLabel(Output output) const
    {
        uint32_t outputIndex = uint32_t(output);
        if (outputIndex < m_outputTextures.size())
        {
            auto& textureHandle = m_outputTextures[outputIndex];
            return textureHandle.Get() ? textureHandle->getDesc().debugName.c_str() : "";
        }
        else
        {
            const char* label = "";
            switch (output)
            {
            case Output::InstanceId:
                label = "Instance Id";
                break;
            case Output::SurfaceIndex:
                label = "Surface Index";
                break;
            case Output::SurfaceUv:
                label = "Surface UV";
                break;
            case Output::Texcoord:
                label = "Texcoord UV";
                break;
            case Output::HiZ:
                label = "HiZ Buffer";
                break;
            default:
                assert(false);
                label = "Unknown";
                break;
            }
            return label;
        }
    }

    nvrhi::TextureHandle GetOutputTexture(Output output) const
    {
        return m_outputTextures[uint32_t(output)];
    }

    void SetSPP(int spp)
    {
        m_params.spp = spp;
        ResetSubframes();
    }
    int GetSPP() const { return m_params.spp; }

    void SetMissColor(float3 missColor)
    {
        m_params.missColor = missColor;
        ResetSubframes();
        ResetDenoiser();
    }
    float3 GetMissColor() const { return m_params.missColor; }

    void SetWireframe(bool wireframe)
    {
        m_params.enableWireframe = wireframe;
        ResetSubframes();
        ResetDenoiser();
    }
    bool GetWireframe() const { return m_params.enableWireframe; }

    void SetShowMicroTriangles(bool showMicroTriangles) { m_showMicroTriangles = showMicroTriangles; }
    bool GetShowMicroTriangles() const { return m_showMicroTriangles; }

    void SetWireframeThickness(float thickness)
    {
        m_params.wireframeThickness = thickness;
        ResetSubframes();
        ResetDenoiser();
    }
    float GetWireframeThickness() const { return m_params.wireframeThickness; }

    int GetPTMaxBounces() const { return m_params.ptMaxBounces; }
    void SetPTMaxBounces(int maxBounces)
    {
        m_params.ptMaxBounces = maxBounces;
        ResetSubframes();
        ResetDenoiser();
    }

    float GetFireflyMaxIntensity() const { return m_params.fireflyMaxIntensity; }
    void SetFireflyMaxIntensity(float fireflyFilterMaxIntensity)
    {
        m_params.fireflyMaxIntensity = fireflyFilterMaxIntensity;
        ResetSubframes();
        ResetDenoiser();
    }

    float GetRoughnessOverride() const { return m_params.roughnessOverride; }
    void  SetRoughnessOverride(float roughness)
    {
        m_params.roughnessOverride = roughness;
        ResetSubframes();
        ResetDenoiser();
    }

    float GetExposure() const
    {
        return m_exposure;
    }
    void SetExposure(float exposure)
    {
        m_exposure = exposure;
        ResetSubframes();
    }

    // Drives exposure from average scene luminance; the Exposure slider stays
    // live as compensation on top.  Display-side only, so no accumulation reset.
    bool GetAutoExposure() const { return m_autoExposure; }
    void SetAutoExposure(bool autoExposure) { m_autoExposure = autoExposure; }

    // Wall-clock frame delta (seconds), forwarded from the app's Animate; drives
    // the auto-exposure temporal eye-adaptation.
    void SetFrameDeltaTime(float dt) { m_frameDeltaTime = dt; }

    float GetLodPixelError() const { return m_lodPixelError; }
    void  SetLodPixelError(float pixelError)
    {
        m_lodPixelError = pixelError;
        ResetSubframes();
        ResetDenoiser();
    }
    // Under streaming budget pressure the effective pixel error is raised above
    // m_lodPixelError (coarser LoDs → smaller resident set), recovering slowly
    // once pressure drops.  Streaming only; --no-adaptive-error opts out.
    void  SetAdaptiveLodError(bool b)       { m_adaptiveLodError = b; }
    bool  GetAdaptiveLodError() const       { return m_adaptiveLodError; }
    // Drop the raised error back to m_lodPixelError.  The controller only
    // recovers at 0.5%/frame and never below the base, so without this a cut to
    // a cheaper viewpoint or a finer -lpe inherits the old error indefinitely.
    void  ResetAdaptiveLodError()           { m_adaptiveLodErrorController.Reset(m_lodPixelError); }
    // The pixel error the last traversal actually used (== m_lodPixelError
    // unless the adaptive path raised it).
    float GetEffectiveLodPixelError() const { return m_adaptiveLodErrorController.GetEffectiveError(); }
    float GetAdaptiveLodLoad() const        { return m_adaptiveLodErrorController.GetSmoothedLoadFactor(); }

    TonemapOperator GetTonemapOperator() const { return m_tonemapOperator; }
    void SetTonemapOperator(TonemapOperator tonemapOperator)
    {
        m_tonemapOperator = tonemapOperator;
    }

    MvecDisplacement GetMVecDisplacement() const { return m_mvecDisplacement; }
    void SetMvecDisplacement(MvecDisplacement mvecDisplacement)
    {
        m_mvecDisplacement = mvecDisplacement;
    }

    float GetEnvMapAzimuth() const { return m_environmentMapAzimuth; }
    void  SetEnvMapAzimuth(float azimuth)
    {
        m_environmentMapAzimuth = azimuth;
        UpdateEnvMapTransform();
        ResetSubframes();
    }
    float GetEnvMapElevation() const { return m_environmentMapElevation; }
    void  SetEnvMapElevation(float elevation)
    {
        m_environmentMapElevation = elevation;
        UpdateEnvMapTransform();
        ResetSubframes();
    }
    float GetEnvMapIntensity() const { return m_params.envmapIntensity; }
    void  SetEnvMapIntensity(float intensity)
    {
        m_params.envmapIntensity = intensity;
        ResetSubframes();
    }

    float GetDenoiserSeparator() const { return m_denoiserSeparator; }
    void  SetDenoiserSeparator(float separator)
    {
        m_denoiserSeparator = separator;
    }

    bool GetEnableEnvmapHeatmap() const { return m_params.enableEnvmapHeatmap; }
    void  SetEnableEnvmapHeatmap(bool enableEnvmapHeatmap)
    {
        m_params.enableEnvmapHeatmap = enableEnvmapHeatmap;
        ResetSubframes();
        ResetDenoiser();
    }

    void SetTimeView(bool timeView);
    bool GetTimeView() const { return m_params.enableTimeView && m_hlslExtensionsSupported; }
    bool IsTimeViewSupported() const { return m_hlslExtensionsSupported; }

    // Warn/log-once state is per scene, not per process: after a scene switch
    // the same misconfiguration deserves the same warning.
    void ResetPerSceneDiagnostics()
    {
        m_warnedLinearAllocCaching = false;
        m_clusterLodSystem.ResetPerSceneDiagnostics();
    }

    void SetDisplayZBuffer(bool displayZBuffer)
    {
        m_displayZBuffer = displayZBuffer;
        m_outputIndex = displayZBuffer ? Output::HiZ : Output::Accumulation;
    }
    bool GetDisplayZBuffer() const { return m_displayZBuffer; }

    int2& GetDebugPixel() { return m_params.debugPixel; }
    
    void SetDebugSurfaceIndex(int surfaceIndex)
    {
        m_params.debugSurfaceIndex = surfaceIndex;
        ResetSubframes(); // Reset accumulation when debug surface changes
    }
    int GetDebugSurfaceIndex() const { return m_params.debugSurfaceIndex; }

    // Inspector-selected SubD mesh; the hit shader draws its wireframe in red.
    void SetSelectedSubdMesh(int meshID)
    {
        if (m_params.selectedSubdMesh != meshID)
        {
            m_params.selectedSubdMesh = meshID;
            ResetSubframes();
        }
    }

    // Inspector-selected cluster-LOD geometry; the hit shader draws its wireframe
    // in red.  instanceID/lodLevel narrow the highlight; -1 means "all".
    void SetSelectedClusterLodGeometry(int geometryID, int instanceID = -1, int lodLevel = -1)
    {
        if (m_params.selectedClusterLodGeometry != geometryID ||
            m_params.selectedClusterLodInstance != instanceID ||
            m_params.selectedClusterLodLevel != lodLevel)
        {
            m_params.selectedClusterLodGeometry = geometryID;
            m_params.selectedClusterLodInstance = instanceID;
            m_params.selectedClusterLodLevel      = lodLevel;
            ResetSubframes(); // restart accumulation so the highlight reads cleanly
        }
    }

    // Must be called while the command list is still open (after RTXMGScene::FinishedLoading).
    void SceneFinishedLoading(std::shared_ptr<RTXMGScene> scene, const ClusterLodBudgets& budgets,
                              nvrhi::ICommandList* commandList);

    // Rebuild the accel-structure resources for the currently loaded scene
    // without reloading it, re-streaming from scratch — how a streaming-budget
    // change is applied.  Needs an open command list and an idle GPU.
    void ReinitAccelStructs(const ClusterLodBudgets& budgets, nvrhi::ICommandList* commandList);

    // Must be set BEFORE SceneFinishedLoading; CreateAccelStructs branches on it.
    void SetUseStreaming(bool useStreaming) { m_useStreaming = useStreaming; }
    bool GetUseStreaming() const            { return m_useStreaming; }
    // Diagnostic: swap the persistent CLAS allocator for the compaction
    // allocator (--linearalloc).
    void SetUseLinearClasAllocator(bool b)    { m_useLinearClasAllocator = b; }
    void SetUseBlasSharing(bool b)            { m_useBlasSharing = b; }
    bool GetUseBlasSharing() const            { return m_useBlasSharing; }
    // Coarse tail LoD levels eligible for sharing (sharingEnabledLevels).
    void SetBlasSharingEnabledLevels(uint32_t n) { m_blasSharingEnabledLevels = n; }
    uint32_t GetBlasSharingEnabledLevels() const { return m_blasSharingEnabledLevels; }
    // BLAS caching: enable BLAS caching (requires sharing; the CLI parser turns
    // sharing on implicitly with --blascaching).
    void SetUseBlasCaching(bool b)            { m_useBlasCaching = b; }
    bool GetUseBlasCaching() const            { return m_useBlasCaching; }
    // Coarse tail LoD levels eligible for caching (blasCacheMinLevel).
    void SetBlasCachingEnabledLevels(uint32_t n) { m_blasCachingEnabledLevels = n; }
    uint32_t GetBlasCachingEnabledLevels() const { return m_blasCachingEnabledLevels; }
    // BLAS merging: enable BLAS merging (requires sharing + streaming; the CLI parser
    // turns sharing on implicitly with --blasmerging).
    void SetUseBlasMerging(bool b)            { m_useBlasMerging = b; }
    bool GetUseBlasMerging() const            { return m_useBlasMerging; }
    // Frustum culling.  Soft (useCulling): off-screen instances coarser
    // via culledErrorScale, kept in TLAS.  Hard (useHardCull): skip traversal →
    // low-detail BLAS.  hardCullForcesInvisible: remove them (null BLAS).
    void SetUseCulling(bool b)                { m_useCulling = b; }
    bool GetUseCulling() const                { return m_useCulling; }
    // Hard cull: off-screen instances skip traversal → low-detail BLAS.
    void SetUseHardCull(bool b)               { m_useHardCull = b; }
    bool GetUseHardCull() const               { return m_useHardCull; }
    // Hard cull removes the instance entirely (null BLAS → dropped from TLAS).
    void SetHardCullForcesInvisible(bool b)   { m_hardCullForcesInvisible = b; }
    bool GetHardCullForcesInvisible() const   { return m_hardCullForcesInvisible; }
    // HiZ occlusion test inside the cull (previous-frame depth pyramid);
    // independent of the frustum test, which keeps running when this is off.
    void SetUseHizOcclusion(bool b)           { m_useHizOcclusion = b; }
    bool GetUseHizOcclusion() const           { return m_useHizOcclusion; }

    // Soft-cull LoD bias (culledErrorScale; clamped >= 1 when applied each frame).
    void SetCulledErrorScale(float f)         { m_culledErrorScale = f; }
    float GetCulledErrorScale() const         { return m_culledErrorScale; }
    // Mantissa bits the CLAS builder drops from vertex positions; both cluster-LoD
    // paths raise it to the bake's compressionPosDropBits on a compressed scene.
    void     SetClasPositionTruncateBits(uint32_t b) { m_clasPositionTruncateBits = b; }
    uint32_t GetClasPositionTruncateBits() const     { return m_clasPositionTruncateBits; }
    // Render-cluster budget exponent the accel structs were built with.
    uint32_t GetRenderClusterBits() const     { return m_clodBudgets.renderClusterBits; }

    // BLAS-effectiveness stats for the Profiler "Streaming" tab, async-read each
    // frame after the BLAS build (1-frame lag).
    const shaderio::SceneBuildingCounters& GetClusterLodCounters() const { return m_clusterLodSystem.GetCounters(); }
    uint32_t GetClusterLodMaxRenderClusters() const { return 1u << m_clodBudgets.renderClusterBits; }

    // The unique/total triangle tallies atomically accumulate into the
    // descriptor-table-bound counters buffer, so they are only valid when the
    // device reports 64-bit atomics on descriptor-heap resources.
    void SetAtomicInt64OnHeapSupported(bool b) { m_atomicInt64OnHeapSupported = b; }
    bool GetAtomicInt64OnHeapSupported() const { return m_atomicInt64OnHeapSupported; }
    uint64_t GetClusterLodBlasActualBytes() const { return m_clusterLodSystem.GetBlasActualBytes(); }
    uint64_t GetClusterLodBlasBufferBytes() const { return m_clusterLodSystem.GetBlasBufferBytes(); }
    uint64_t GetClusterLodMetadataBytes() const { return m_clusterLodSystem.GetMetadataBytes(); }

    // Effective (post-init, defaults-resolved) streaming config for the sidebar
    // sliders to display. Returns false when not in streaming mode (--preload).
    bool GetClusterLodStreamingConfig(rtxmg::StreamingConfig& out) const
    {
        const IClusterLodStreamingHooks* hooks =
            (m_useStreaming && m_clusterLodResources) ? m_clusterLodResources->GetStreamingHooks() : nullptr;
        if (!hooks)
            return false;
        out = hooks->GetStreamingConfig();
        return true;
    }
    // Diagnostic: verbose Cluster-LoD streaming/traversal/BLAS readbacks.
    void SetDebugClusterLod(bool b)          { m_debugClusterLod = b; }
    // Diagnostic: upload resident group blobs verbatim (--nostrip) —
    // disables the position/normal strip rewrite at streaming init.
    void SetStripResidentData(bool b)        { m_stripResidentData = b; }
    // Diagnostic: keep positions resident (--nostrippos) while leaving the
    // normals strip to the Vertex Normals toggle — the channels are independent.
    void SetStripResidentPositions(bool b)   { m_stripResidentPositions = b; }
    // Cluster-LoD baked vertex-normals shading.  Streaming init also reads it to
    // decide whether resident group blobs keep their normal words, and init runs
    // before the first UpdateAccelerationStructures re-syncs the flag — so this
    // setter only matters for the scene-load / re-stream ordering.
    void SetEnableClusterLodVertexNormals(bool b)  { m_enableClusterLodVertexNormals = b; }
    // Cluster-LoD normal-map shading (--normalmaps).  Only reachable on top of
    // the vertex normals above; the permutation resolve below folds that in.
    void SetEnableClusterLodNormalMaps(bool b)     { m_enableClusterLodNormalMaps = b; }
    bool GetEnableClusterLodNormalMaps() const     { return m_enableClusterLodNormalMaps; }

    ClusterLodResources* GetClusterLodResources() const { return m_clusterLodResources.get(); }

    // Cluster-LOD streaming stats for the Profiler "Streaming" tab.  Returns
    // false, leaving `out` untouched, on the --preload path.
    bool GetClusterLodStreamingStats(rtxmg::StreamingStats& out) const
    {
        const IClusterLodStreamingHooks* hooks =
            (m_useStreaming && m_clusterLodResources) ? m_clusterLodResources->GetStreamingHooks() : nullptr;
        if (!hooks)
            return false;
        hooks->GetStats(out);
        return true;
    }

    std::shared_ptr<TextureCache> GetTextureCache() const
    {
        return m_textureCache;
    }

    std::shared_ptr<DescriptorTableManager> GetDescriptorTable() const
    {
        return m_descriptorTable;
    }

    std::shared_ptr<ShaderFactory> GetShaderFactory() const
    {
        return m_shaderFactory;
    }

    std::shared_ptr<CommonRenderPasses> GetCommonPasses() const
    {
        return m_commonPasses;
    }

    void DumpPixelDebugBuffers(nvrhi::ICommandList* commandList);

#if ENABLE_PIXEL_PICK
    void ReadPixelPick(nvrhi::ICommandList* commandList);

    struct PixelPick
    {
        uint32_t    instanceID  = ~0u;
        uint32_t    surfaceID   = ~0u;   // subd: surface index; cluster-LOD: geometryID
        uint32_t    materialID  = ~0u;
        uint32_t    lodLevel    = ~0u;   // cluster-LOD hits only
        std::string name;
        bool        isClusterLod = false;
        bool        valid       = false;
        // Bumps on every valid pick (even re-picking the same mesh) so the
        // Inspector can react once per right-click; 0 = never picked.
        uint32_t    sequence    = 0;
    };
    const PixelPick& GetPixelPick() const { return m_pixelPick; }
#endif

    void SetRenderSize(int2 renderSize, int2 displaySize);

    // Device bytes held by the per-view targets: the output/display textures, the
    // z-prepass depth and its HiZ chain, and the per-pixel readback buffers.  Only
    // what this renderer owns -- DLSS's own targets are internal to Streamline.
    uint64_t GetRenderTargetBytes() const;
    // Envmap texture plus its importance-sampling tables, split out because the
    // texture budget does not cover it.
    uint64_t GetEnvMapBytes() const;
    // TLAS storage and its instance-desc buffer.
    uint64_t GetTlasBytes() const;
    uint64_t GetClusterTessWorkingBytes() const;

    ZBuffer* GetZBuffer() { return m_zbuffer.get(); }
    const ZBuffer* GetZBuffer() const { return m_zbuffer.get(); }

    nvrhi::rt::AccelStructHandle GetTopLevelAS() const { return m_topLevelAS; }
    // True once the current TLAS handle has been built at least once — the
    // HiZ prepass traces the previous frame's TLAS at the top of the frame and
    // must skip unbuilt handles (first frame, or right after a pool reinit
    // recreated the TLAS).
    bool IsTopLevelASBuilt() const { return m_topLevelASBuilt; }

    std::unique_ptr<ClusterTessAccels>& GetSceneAccels() { return m_sceneAccels; }
    std::unique_ptr<ClusterTessellator>& GetAccelBuilder() { return m_clusterAccelBuilder; }

    void SetEnvMap(const std::string& filePath, nvrhi::ICommandList* commandList);
    std::shared_ptr<LoadedTexture> GetEnvMap() const { return m_envMap; }
    void ClearEnvMap() { m_envMap = nullptr; }
private:

    nvrhi::IDevice* GetDevice() const { return m_options.device; }

    void FillInstanceDescs(nvrhi::ICommandList* commandList, nvrhi::IBuffer* outInstanceDescs, nvrhi::IBuffer* blasAddresses, uint32_t numInstances, uint32_t instanceOffset = 0u);
    bool ShouldDebugClusterLodTlasFill();
    void LogClusterLodTlasFillInputs(nvrhi::IBuffer* blasAddresses,
                                     uint32_t numInstances,
                                     uint32_t instanceOffset) const;
    void DebugReadbackClusterLodTlasFill(nvrhi::ICommandList* commandList,
                                         uint32_t instanceOffset);

    void CreateAccelStructs(nvrhi::ICommandList* commandList);

    void UpdateEnvMapTransform();
    void UpdateEnvMapSampling(nvrhi::ICommandList* commandList);

    void ComputeMotionVectors(nvrhi::ICommandList* commandList);

    // Needs m_displayTexture, so the pass is created on the first blit.
    void EnsureToneMappingPass(nvrhi::ICommandList* commandList);
private:
    

    Options m_options;
    RenderParams& m_params;
    ShadingMode m_shadingMode = ShadingMode::PT;
    ColorMode m_colorMode = ColorMode::BASE_COLOR;
    TonemapOperator m_tonemapOperator = TonemapOperator::Aces;
    float m_exposure = 1.0f;
    // The exposure target comes from m_toneMappingParams.exposureBias (donut's
    // own calibration); m_frameDeltaTime feeds the eye-adaptation blend.
    bool m_autoExposure = true;
    float m_frameDeltaTime = 0.0f;
    float m_lodPixelError = 1.0f;
    // Adaptive LoD error state (see SetAdaptiveLodError).
    bool  m_adaptiveLodError = true;
    rtxmg::AdaptiveLodError m_adaptiveLodErrorController;

    PreprocessEnvMapShaders m_preprocessEnvMapShaders;
    PreprocessEnvMapResources m_preprocessEnvMapResources;
    std::shared_ptr<LoadedTexture> m_envMap;
    ScanSystem m_scanSystem;
    float m_environmentMapAzimuth = 0.f;
    float m_environmentMapElevation = 0.f;

    Camera m_camera;
    Camera m_previousCamera;

    int2 m_renderSize = int2(0, 0);
    int2 m_displaySize = int2(0, 0);

    // Ray tracing permutation support
    class RayTracingPermutation
    {
    public:
        // Cluster-LoD shading ladder, matching the CLUSTER_LOD_SHADING macro.
        // Normal maps are a step past vertex normals rather than an independent
        // bit, so "normal maps without vertex normals" cannot be expressed.
        enum ClusterLodShading : uint32_t
        {
            Flat = 0,
            VertexNormals = 1,
            NormalMapped = 2,
            ShadingCount
        };
        static constexpr size_t kCount = 2 * size_t(ShadingCount);
        uint32_t index() const { return (m_vertexNormals ? 1u : 0u) + 2u * m_clusterLodShading; }

        RayTracingPermutation(bool enableVertexNormals, bool enableClusterLodVertexNormals,
                              bool enableClusterLodNormalMaps)
            : m_vertexNormals(enableVertexNormals)
            , m_clusterLodShading(!enableClusterLodVertexNormals ? Flat
                                  : (enableClusterLodNormalMaps ? NormalMapped : VertexNormals))
        {
        }

        bool     isVertexNormalsEnabled() const { return m_vertexNormals; }
        uint32_t clusterLodShading() const { return m_clusterLodShading; }

    private:
        bool     m_vertexNormals = false;
        uint32_t m_clusterLodShading = Flat;
    };

    std::array<nvrhi::rt::PipelineHandle, RayTracingPermutation::kCount> m_rayPipelines = {};
    std::array<nvrhi::rt::ShaderTableHandle, RayTracingPermutation::kCount> m_shaderTables = {};
    nvrhi::BindingLayoutHandle m_bindingLayout;
    nvrhi::BindingSetHandle m_bindingSet;
    nvrhi::BindingLayoutHandle m_bindlessLayout;

    nvrhi::BufferHandle m_dummyBuffer;

    nvrhi::ComputePipelineHandle m_blitPipeline;
    nvrhi::BindingLayoutHandle m_blitBL;
    nvrhi::BufferHandle m_blitParamsBuffer;

    // The FramebufferFactory only exists to satisfy the tone-mapping pass ctor's
    // (unused) render PSO.
    std::shared_ptr<FramebufferFactory> m_toneMappingFbFactory;
    std::unique_ptr<donut::render::ToneMappingPass> m_toneMappingPass;
    donut::render::ToneMappingParameters m_toneMappingParams;

    std::unique_ptr<ClusterTessellator>   m_clusterAccelBuilder;
    // A ClusterLodStreaming (default) or a ClusterLodPreloaded (--preload); the
    // cluster-LoD passes only see the ClusterLodResources interface.
    std::unique_ptr<ClusterLodResources>  m_clusterLodResources;

    bool                                  m_useStreaming = true;
    bool                                  m_useLinearClasAllocator = false;
    bool                                  m_useBlasSharing = true;
    uint32_t                              m_blasSharingEnabledLevels = 8;
    bool                                  m_useBlasCaching = true;
    uint32_t                              m_blasCachingEnabledLevels = 8;
    // Last frame's effective useBlasCaching, to detect the toggle edge and
    // reset the cached-BLAS state (see RetireClusterLodAccelResources).
    bool                                  m_clusterLodPrevUseBlasCaching = false;
    bool                                  m_atomicInt64OnHeapSupported = false;
    bool                                  m_useBlasMerging = true;
    // Frustum culling.  Soft culling on by default; see args.h.
    bool                                  m_useCulling = true;
    bool                                  m_useHardCull = false;
    bool                                  m_hardCullForcesInvisible = false;
    // HiZ occlusion test inside the cull (z-prepass depth pyramid).
    bool                                  m_useHizOcclusion = true;
    float                                 m_culledErrorScale = 2.0f;
    // The budgets the accel structs were last built with; the app owns the requested ones.
    ClusterLodBudgets                     m_clodBudgets;
    uint32_t                              m_clasPositionTruncateBits = 0;
    bool                                  m_debugClusterLod      = false;
    bool                                  m_stripResidentData    = true;
    bool                                  m_stripResidentPositions = true;
    uint64_t                              m_debugClusterLodTlasFillFrame = 0;
    // Owns the two cluster-LoD passes and sequences them around the streaming
    // hooks; m_clusterLodResources is handed to it non-owning.
    rtxmg::ClusterLodSystem               m_clusterLodSystem;
    RTXMGBuffer<nvrhi::rt::InstanceDesc> m_instanceDescs;
    nvrhi::rt::AccelStructHandle         m_topLevelAS;
    bool                                 m_topLevelASBuilt = false;
    // Descriptor-table relocation tripwire (see ReserveCapacity at init).
    uint32_t                             m_reservedDescriptorCapacity = 0;
    bool                                 m_descriptorRelocationLogged = false;
    std::unique_ptr<ClusterTessAccels>   m_sceneAccels;

    nvrhi::BufferHandle m_fillInstanceDescsParams;
    nvrhi::BindingLayoutHandle m_fillInstanceDescsBL;
    nvrhi::ComputePipelineHandle m_fillInstanceDescsPSO;

    nvrhi::BufferHandle m_lightingConstantsBuffer;
    nvrhi::BufferHandle m_renderParamsBuffer;

    std::shared_ptr<DescriptorTableManager> m_descriptorTable;

    // Render Textures (RenderSize)
    std::array<nvrhi::TextureHandle, size_t(OutputTexture::Count)> m_outputTextures;
    Output m_outputIndex = Output::Accumulation;
    nvrhi::BufferHandle m_hitResultBuffer;

    // Display Textures (DisplayRes)
    nvrhi::TextureHandle m_dlssOutputColorTexture; // Upscaled color output.
    nvrhi::TextureHandle m_displayTexture; // Can be eliminated when blit is converted to PS

    float m_denoiserSeparator = 0.0f;
    bool m_resetDenoiser = false;
    bool m_showMicroTriangles = false;

    // Motion Vectors
    MvecDisplacement m_mvecDisplacement = MvecDisplacement::FromSubdEval;
    RTXMGBuffer<SubdInstance> m_subdInstancesBuffer;
    nvrhi::BindingLayoutHandle m_motionVectorsBL;
    nvrhi::ComputePipelineHandle m_motionVectorsPSO[size_t(MvecDisplacement::Count)];

    // Debug Buffers
#if ENABLE_SHADER_DEBUG
    RTXMGBuffer<ShaderDebugElement> m_pixelDebugBuffer;
#endif
#if ENABLE_PIXEL_PICK
    RTXMGBuffer<PixelPickResult>    m_pixelPickBuffer;
    PixelPick                       m_pixelPick;
    uint32_t                        m_pixelPickSeq = 0;
#endif
    nvrhi::BufferHandle m_timeViewBuffer;

    std::unique_ptr<BindingCache> m_bindingCache;
    std::shared_ptr<TextureCache> m_textureCache;
    std::shared_ptr<RTXMGScene> m_scene;



    std::shared_ptr<ShaderFactory> m_shaderFactory;
    std::shared_ptr<CommonRenderPasses> m_commonPasses;

    PlanarView m_view;
    PlanarView m_viewPrevious;

    bool m_needsRebind = true;
    bool m_needsEnvMapUpdate = false;
    bool m_displayZBuffer = false;
    bool m_warnedLinearAllocCaching = false;
    bool m_enableVertexNormals = false;
    bool m_enableClusterLodVertexNormals = false;
    bool m_enableClusterLodNormalMaps = false;
    // NVAPI HLSL extensions; time view reads the GPU timer through them.
    bool m_hlslExtensionsSupported = false;
    std::unique_ptr<ZBuffer> m_zbuffer;
    // Owned alongside the ZBuffer and the TLAS it traces, so the prepass cannot
    // outlive memory this renderer retires.
    std::unique_ptr<ZRenderer> m_zRenderer;

    void RetireClusterLodAccelResources(nvrhi::ICommandList* commandList);
    void RetireClusterTessAccelResources(const TessellatorConfig* tessConfig);

    bool EffectiveUseBlasCaching() const
    {
        return m_useBlasSharing && m_useBlasCaching && !m_useLinearClasAllocator;
    }
};
