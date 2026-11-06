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

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtxmg
{
    // Pure greedy-waterline mip-budget solver.
    //
    // perTextureLevelBytes[t] = the GPU byte size of each mip level of texture t,
    // index 0 = mip 0 (largest). budgetBytes = the total GPU memory target.
    //
    // Returns a baseMip per texture: keep levels [baseMip, end), drop the
    // [0, baseMip) highest-resolution levels. Repeatedly drops the finest resident
    // mip of the largest-footprint texture until the total fits, or every texture
    // is down to its last (smallest) mip. Returns all-zero if the full set already
    // fits the budget.
    std::vector<uint32_t> SolveMipBudgetWaterline(
        const std::vector<std::vector<uint64_t>>& perTextureLevelBytes,
        uint64_t budgetBytes);

    // What the budget pre-pass measured, for the Memory tab.  Covers only the
    // slots the renderer loads (base color / metallic-roughness / emissive, plus
    // normal maps unless --no-normalmaps), so it lines up with what reaches VRAM.
    // textureCount/diskBytes span every
    // format; the budgetable*/keptBytes figures describe the KTX2 subset, the
    // only one whose per-level header index allows dropping a mip undecoded.
    struct TextureBudgetStats
    {
        uint32_t textureCount    = 0;  // unique images the scene loads, any format
        uint32_t budgetableCount = 0;  // of those, how many are KTX2
        uint32_t droppedCount    = 0;  // of the KTX2 ones, how many lost high-res mips
        uint64_t budgetableFullBytes = 0;  // full-res GPU footprint of the KTX2 subset
        uint64_t keptBytes       = 0;  // GPU footprint the solver planned for that subset
        uint64_t diskBytes       = 0;  // file bytes as stored on disk, any format
        uint64_t budgetBytes     = 0;  // the budget applied (0 = unlimited)
    };

    // Load-time KTX2 mip-budget pre-pass: reads each KTX2 header (no pixel
    // data), solves the waterline against budgetBytes, and writes the chosen base
    // mip per texture into outBaseMips, keyed by the resolved path's
    // generic_string.  Hand that map to RTXMGScene::SetTextureBaseMips before the
    // scene loads its textures.  Accepts a .gltf or a .scene.json (whose "models"
    // entries are accounted together).  budgetBytes == 0 measures into outStats
    // without dropping any mip; outStats is zeroed up front, so an unreadable
    // scene reports zeroes rather than stale numbers.
    //
    // normalMaps must match MaterialLibrary::SetEnableNormalMaps, or the
    // waterline is solved against a different texture set than the loader reads.
    void ApplyKtxTextureBudget(
        std::unordered_map<std::string, uint32_t>& outBaseMips,
        const std::filesystem::path& sceneFile,
        uint64_t budgetBytes,
        bool normalMaps = false,
        // One mip-drop line per dropped texture; the summary prints regardless.
        bool verbose = false,
        TextureBudgetStats* outStats = nullptr);
}
