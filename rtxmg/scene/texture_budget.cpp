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

#include "rtxmg/scene/texture_budget.h"

#include "rtxmg/scene/json.h"
#include "rtxmg/subdivision/shape.h"

#include <donut/engine/TextureCache.h>
#include <donut/engine/DDSFile.h>
#include <donut/engine/KTX2File.h>
#include <donut/core/log.h>

#include <json/json.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <queue>
#include <unordered_set>
#include <utility>

#if DONUT_WITH_KTX
#include <cgltf.h>
#endif

namespace fs = std::filesystem;

namespace rtxmg
{
    std::vector<uint32_t> SolveMipBudgetWaterline(
        const std::vector<std::vector<uint64_t>>& perTex, uint64_t budgetBytes)
    {
        const size_t n = perTex.size();
        std::vector<uint32_t> baseMip(n, 0);
        std::vector<uint64_t> footprint(n, 0);

        uint64_t total = 0;
        for (size_t t = 0; t < n; ++t)
        {
            for (uint64_t b : perTex[t]) footprint[t] += b;
            total += footprint[t];
        }
        if (total <= budgetBytes)
            return baseMip;

        // Max-heap by current footprint; exactly one entry per still-droppable
        // texture (we pop then push back the updated entry), so the popped
        // footprint is never stale.
        std::priority_queue<std::pair<uint64_t, size_t>> heap;
        for (size_t t = 0; t < n; ++t)
            if (perTex[t].size() > 1)
                heap.push({ footprint[t], t });

        while (total > budgetBytes && !heap.empty())
        {
            const size_t t = heap.top().second;
            heap.pop();

            const uint64_t dropped = perTex[t][baseMip[t]]; // finest resident level
            baseMip[t] += 1;
            footprint[t] -= dropped;
            total -= dropped;

            if (baseMip[t] + 1 < perTex[t].size()) // keep at least the last mip
                heap.push({ footprint[t], t });
        }
        return baseMip;
    }

#if DONUT_WITH_KTX
    namespace
    {
        std::string LowerExt(const fs::path& p)
        {
            std::string e = p.extension().string();
            std::transform(e.begin(), e.end(), e.begin(),
                           [](unsigned char c) { return char(std::tolower(c)); });
            return e;
        }

        // Read just the leading bytes of a file (enough for a KTX2 header + level
        // index). Returns empty on failure.
        std::vector<char> ReadHead(const fs::path& path, size_t maxBytes)
        {
            std::ifstream f(path, std::ios::binary);
            if (!f) return {};
            std::vector<char> buf(maxBytes);
            f.read(buf.data(), std::streamsize(maxBytes));
            buf.resize(size_t(f.gcount()));
            return buf;
        }
    } // namespace

    void ApplyKtxTextureBudget(std::unordered_map<std::string, uint32_t>& outBaseMips,
                               const fs::path& sceneFile, uint64_t budgetBytes, bool normalMaps,
                               bool verbose, TextureBudgetStats* outStats)
    {
        using namespace donut;

        outBaseMips.clear();
        if (outStats)
            *outStats = { .budgetBytes = budgetBytes };

        // Resolve the model file(s) to account. A plain .gltf/.obj is taken as-is;
        // a .scene.json is parsed for its "models" array and every entry is
        // resolved against the json's directory (absolute entries pass through
        // operator/ unchanged) — the same resolution RTXMGScene::LoadSceneFile
        // applies.
        std::vector<fs::path> modelFiles;
        const std::string sceneExt = LowerExt(sceneFile);
        if (sceneExt == ".gltf" || sceneExt == ".glb" || sceneExt == ".obj")
        {
            modelFiles.push_back(sceneFile);
        }
        else if (sceneExt == ".json")
        {
            Json::Value jsonRoot;
            try
            {
                jsonRoot = readFile(sceneFile);
            }
            catch (const std::exception& e)
            {
                log::warning("Texture budget: failed to parse '%s': %s",
                             sceneFile.generic_string().c_str(), e.what());
                return;
            }
            if (const Json::Value& models = jsonRoot["models"]; models.isArray())
            {
                for (const Json::Value& model : models)
                {
                    if (!model.isString())
                        continue;
                    const fs::path modelPath = model.asString();
                    const std::string ext = LowerExt(modelPath);
                    if (ext != ".gltf" && ext != ".glb" && ext != ".obj")
                        continue;
                    modelFiles.push_back((sceneFile.parent_path() / modelPath).lexically_normal());
                }
            }
            if (modelFiles.empty())
            {
                log::info("Texture budget: no models in '%s'; nothing to budget.",
                          sceneFile.generic_string().c_str());
                return;
            }
        }
        else
        {
            log::info("Texture budget: unsupported scene type; skipping '%s'.",
                      sceneFile.generic_string().c_str());
            return;
        }

        std::vector<std::string>            keys;
        std::vector<std::vector<uint64_t>>  perTexBytes;
        std::vector<std::pair<uint32_t, uint32_t>> texDims; // mip-0 w,h (for drop logging)
        std::unordered_set<std::string>     seen;
        uint32_t                            totalCount = 0; // unique images, any format
        uint64_t                            diskBytes  = 0; // their file bytes as stored

        // Account one already-resolved image path.  Only the slots RTXMGScene
        // loads are passed here: budgeting images the loader skips would over-drop
        // mips on the ones that do reach VRAM.  The path must resolve exactly as
        // the loader's does, or the base-mip keys won't match its lookups.  Every
        // format is counted for the Memory tab, but only the container formats
        // that describe their mips up front can be budgeted.
        auto accountTexture = [&](const fs::path& resolved)
        {
            const std::string key = resolved.generic_string();
            if (!seen.insert(key).second)
                return; // already accounted for (shared across materials)

            // On-disk size of the file as stored, for the Memory tab's "on disk"
            // figure. std::error_code overload: an image the scene references but
            // that isn't there just contributes 0 (the loader warns separately).
            std::error_code ec;
            const uint64_t fileBytes = uint64_t(fs::file_size(resolved, ec));
            if (ec)
                return;  // missing file: not a texture this scene will load
            ++totalCount;
            diskBytes += fileBytes;

            const std::string ext = LowerExt(resolved);
            if (ext != ".ktx2" && ext != ".dds")
                return;  // counted above; not mip-budgetable

            // 4 KB covers a KTX2 header + level index and a DDS header + DX10 block.
            std::vector<char> head = ReadHead(resolved, 4096);
            if (head.empty())
                return;

            std::vector<uint64_t> levels;
            uint32_t width = 0, height = 0;

            if (ext == ".ktx2")
            {
                engine::KTX2HeaderInfo info;
                if (!ReadKTX2Header(head.data(), head.size(), key.c_str(), info) || !info.supported)
                    return;
                levels.resize(info.levelCount);
                for (uint32_t l = 0; l < info.levelCount; ++l)
                    levels[l] = info.levels[l].gpuBytes;
                width = info.width;
                height = info.height;
            }
            else
            {
                engine::DDSHeaderInfo info;
                if (!ReadDDSHeader(head.data(), head.size(), key.c_str(), info) || !info.supported)
                    return;
                levels.resize(info.levelCount);
                for (uint32_t l = 0; l < info.levelCount; ++l)
                    levels[l] = info.levels[l].gpuBytes * info.arraySize;
                width = info.width;
                height = info.height;
            }

            if (levels.empty())
                return;

            keys.push_back(key);
            perTexBytes.push_back(std::move(levels));
            texDims.emplace_back(width, height);
        };

        // glTF: base color, metallic-roughness, emissive and (unless --no-normalmaps)
        // normal, matching what RTXMGScene's importer loads.
        auto accountGltf = [&](const fs::path& gltfFile)
        {
            cgltf_options options = {};
            cgltf_data*   data    = nullptr;
            if (cgltf_parse_file(&options, gltfFile.string().c_str(), &data) != cgltf_result_success)
            {
                log::warning("Texture budget: cgltf failed to parse '%s'.",
                             gltfFile.generic_string().c_str());
                return;
            }
            const fs::path gltfDir = gltfFile.parent_path();

            auto accountView = [&](const cgltf_texture_view& tv)
            {
                if (!tv.texture || !tv.texture->image || !tv.texture->image->uri)
                    return;
                std::string uri = tv.texture->image->uri;
                cgltf_decode_uri(uri.data());
                uri = uri.c_str();  // truncate at the decoded (earlier) null terminator
                accountTexture((gltfDir / uri).lexically_normal());
            };

            for (size_t mi = 0; mi < data->materials_count; ++mi)
            {
                const cgltf_material& m = data->materials[mi];
                if (m.has_pbr_metallic_roughness)
                {
                    accountView(m.pbr_metallic_roughness.base_color_texture);
                    accountView(m.pbr_metallic_roughness.metallic_roughness_texture);
                }
                accountView(m.emissive_texture);
                if (normalMaps)
                    accountView(m.normal_texture);
            }

            cgltf_free(data);
        };

        // OBJ: the five .mtl maps RTXMGScene loads, resolved relative to the
        // mtllib the way its addTexture does.  Only the .mtl is parsed - the mesh
        // itself has nothing to contribute here.
        // A frame-sequence model ("foo.[100-387].obj") names a range, not a file.
        // Every frame shares one mtllib, so the first existing frame answers for
        // the whole sequence.  Mirrors ObjImporter's getSequenceFormat.
        auto resolveSequenceFrame = [](const fs::path& p) -> fs::path
        {
            const std::string s = p.generic_string();
            const size_t open  = s.find('[');
            if (open == std::string::npos)
                return p;
            const size_t dash  = s.find('-', open);
            const size_t close = s.find(']', open);
            if (dash == std::string::npos || close == std::string::npos || dash > close)
                return p;

            int first = 0;
            std::from_chars(s.data() + open + 1, s.data() + dash, first);
            const std::string prefix = s.substr(0, open);
            const std::string suffix = s.substr(close + 1);
            for (const char* format : { "%d", "%03d", "%04d" })
            {
                char buf[16];
                std::snprintf(buf, std::size(buf), format, first);
                fs::path candidate = prefix + buf + suffix;
                if (fs::is_regular_file(candidate))
                    return candidate;
            }
            return p;
        };

        auto accountObj = [&](const fs::path& objFileIn)
        {
            const fs::path objFile = resolveSequenceFrame(objFileIn);
            const std::string mtllib = ReadObjMtllibName(objFile);
            if (mtllib.empty())
                return;

            const fs::path mtlPath = (objFile.parent_path() / mtllib).lexically_normal();
            if (!fs::is_regular_file(mtlPath))
            {
                log::warning("Texture budget: mtllib '%s' of '%s' not found.",
                             mtlPath.generic_string().c_str(), objFile.generic_string().c_str());
                return;
            }

            // Tiled maps ("foo.<UDIM>.dds") name no file of their own, so resolve
            // them to the tiles on disk -- that is what the loader ends up reading.
            const fs::path texDir = mtlPath.parent_path();
            for (const std::unique_ptr<Shape::material>& mtl : ParseMtllibResolved(mtlPath, objFile.parent_path()))
            {
                if (!mtl)
                    continue;
                for (const std::string* map : { &mtl->map_kd, &mtl->map_pm, &mtl->map_pr,
                                                &mtl->map_ks, &mtl->map_bump })
                {
                    if (!map->empty())
                        accountTexture((texDir / *map).lexically_normal());
                }
            }
        };

        for (const fs::path& modelFile : modelFiles)
        {
            if (LowerExt(modelFile) == ".obj")
                accountObj(modelFile);
            else
                accountGltf(modelFile);
        }

        // No KTX2 anywhere (e.g. a .jpg/.png glTF): nothing to budget, but the
        // scene was still measured — publish the totals so the Memory tab can
        // report the footprint. Their GPU size isn't knowable from the file
        // headers here; it is filled in from residency as they finalize.
        if (keys.empty())
        {
            if (outStats)
            {
                outStats->textureCount = totalCount;
                outStats->diskBytes    = diskBytes;
            }
            log::info("Texture budget: no mip-budgetable images among the %u textures of '%s' "
                      "(disk=%llu MB); nothing to budget.",
                      totalCount, sceneFile.generic_string().c_str(),
                      (unsigned long long)(diskBytes >> 20));
            return;
        }

        uint64_t fullBytes = 0;
        for (const auto& v : perTexBytes)
            for (uint64_t b : v) fullBytes += b;

        // budgetBytes == 0 is "measure only": keep every mip. (Handing 0 to the
        // solver would instead strip every texture down to its last mip.)
        const std::vector<uint32_t> baseMips =
            budgetBytes ? SolveMipBudgetWaterline(perTexBytes, budgetBytes)
                        : std::vector<uint32_t>(keys.size(), 0);

        uint64_t keptBytes = 0;
        uint32_t droppedCount = 0;
        for (size_t t = 0; t < keys.size(); ++t)
        {
            outBaseMips[keys[t]] = baseMips[t];

            uint64_t texFull = 0, texKept = 0;
            for (size_t l = 0; l < perTexBytes[t].size(); ++l)
            {
                texFull += perTexBytes[t][l];
                if (l >= baseMips[t])
                    texKept += perTexBytes[t][l];
            }
            keptBytes += texKept;

            if (baseMips[t] > 0)
            {
                ++droppedCount;
                if (verbose)
                {
                    // Per-texture drop: full mip-0 res -> kept (base-mip) res.
                    const uint32_t kw = std::max(1u, texDims[t].first  >> baseMips[t]);
                    const uint32_t kh = std::max(1u, texDims[t].second >> baseMips[t]);
                    log::info("Texture budget: dropped %u mip(s) from '%s'  %ux%u -> %ux%u  (%llu KB -> %llu KB)",
                              baseMips[t], keys[t].c_str(),
                              texDims[t].first, texDims[t].second, kw, kh,
                              (unsigned long long)(texFull >> 10),
                              (unsigned long long)(texKept >> 10));
                }
            }
        }

        if (outStats)
            *outStats = {
                .textureCount    = totalCount,
                .budgetableCount = uint32_t(keys.size()),
                .droppedCount    = droppedCount,
                .budgetableFullBytes = fullBytes,
                .keptBytes       = keptBytes,
                .diskBytes       = diskBytes,
                .budgetBytes     = budgetBytes,
            };

        {
            const std::string budgetStr = budgetBytes
                ? std::to_string(budgetBytes >> 20) + " MB"
                : std::string("unlimited");
            log::info("Texture budget: %u textures (base/MR/emissive%s), disk=%llu MB; "
                      "%zu are KTX2/DDS: full=%llu MB, budget=%s, kept=%llu MB; %u had high-res mips dropped.",
                      totalCount,
                      normalMaps ? "/normal" : "",
                      (unsigned long long)(diskBytes >> 20),
                      keys.size(),
                      (unsigned long long)(fullBytes >> 20),
                      budgetStr.c_str(),
                      (unsigned long long)(keptBytes >> 20),
                      droppedCount);
        }
    }
#else  // !DONUT_WITH_KTX
    void ApplyKtxTextureBudget(std::unordered_map<std::string, uint32_t>& outBaseMips,
                               const fs::path&, uint64_t, bool, bool,
                               TextureBudgetStats* outStats)
    {
        outBaseMips.clear();
        if (outStats)
            *outStats = {};
        donut::log::warning("Texture budget requested but build has no DONUT_WITH_KTX support.");
    }
#endif // DONUT_WITH_KTX
}
