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

// clang-format off

#include "shot_list.h"

#include "rtxmg/scene/camera.h"
#include "rtxmg/scene/json.h"

#include <donut/core/log.h>

#include <json/json.h>

#include <fstream>
#include <map>

// clang-format on

namespace fs = std::filesystem;
using donut::log::warning;

namespace
{

// Same spellings the -sm / -cm flags accept, so a shot list reads like the CLI.
const std::map<std::string, ShadingMode> kShadingModes{
    { "prim_rays", ShadingMode::PRIMARY_RAYS }, { "primary", ShadingMode::PRIMARY_RAYS },
    { "ao",        ShadingMode::AO },
    { "pt",        ShadingMode::PT },
};

// Same spellings --dlssMode accepts.
const std::map<std::string, donut::app::StreamlineInterface::DLSSMode> kDlssModes{
    { "off",         donut::app::StreamlineInterface::DLSSMode::eOff },
    { "ultra",       donut::app::StreamlineInterface::DLSSMode::eUltraPerformance },
    { "performance", donut::app::StreamlineInterface::DLSSMode::eMaxPerformance },
    { "balanced",    donut::app::StreamlineInterface::DLSSMode::eBalanced },
    { "quality",     donut::app::StreamlineInterface::DLSSMode::eMaxQuality },
    { "DLAA",        donut::app::StreamlineInterface::DLSSMode::eDLAA },
    { "dlaa",        donut::app::StreamlineInterface::DLSSMode::eDLAA },
};

const std::map<std::string, ColorMode> kColorModes{
    { "base",       ColorMode::BASE_COLOR },        { "basecolor", ColorMode::BASE_COLOR },
    { "n",          ColorMode::COLOR_BY_SHADING_NORMAL },
    { "normal",     ColorMode::COLOR_BY_SHADING_NORMAL },
    { "texcoord",   ColorMode::COLOR_BY_TEXCOORD },
    { "uv",         ColorMode::COLOR_BY_CLUSTER_UV },
    { "mat",        ColorMode::COLOR_BY_MATERIAL },
    { "cid",        ColorMode::COLOR_BY_CLUSTER_ID },
    { "tid",        ColorMode::COLOR_BY_MICROTRI_ID },
    { "tArea",      ColorMode::COLOR_BY_MICROTRI_AREA },
    { "lod",        ColorMode::COLOR_BY_LOD_LEVEL },
    { "group",      ColorMode::COLOR_BY_CLUSTER_GROUP },
    { "blas",       ColorMode::COLOR_BY_BLAS_SOURCE },
    { "blascached", ColorMode::COLOR_BY_BLAS_CACHED },
};

// Reverse lookups for the filename tag.  First spelling wins, so the tag is
// stable even though several aliases parse to the same enum.
std::string NameOf(ShadingMode m)
{
    switch (m)
    {
    case ShadingMode::PRIMARY_RAYS: return "primary";
    case ShadingMode::AO:           return "ao";
    default:                        return "pt";
    }
}

std::string NameOf(ColorMode m)
{
    // "n" would win the map order but reads as noise in a filename.
    if (m == ColorMode::COLOR_BY_SHADING_NORMAL)
        return "normal";
    for (const auto& [name, value] : kColorModes)
        if (value == m)
            return name;
    return "base";
}

}  // anonymous namespace

const char* DlssModeName(donut::app::StreamlineInterface::DLSSMode mode)
{
    using DLSSMode = donut::app::StreamlineInterface::DLSSMode;
    switch (mode)
    {
    case DLSSMode::eMaxPerformance:   return "performance";
    case DLSSMode::eBalanced:         return "balanced";
    case DLSSMode::eMaxQuality:       return "quality";
    case DLSSMode::eUltraPerformance: return "ultra";
    case DLSSMode::eUltraQuality:     return "ultraquality";
    case DLSSMode::eDLAA:             return "DLAA";
    default:                          return "off";
    }
}

bool Shot::ShowsColorMode() const
{
    // AO writes baseColor * visibility, so a colour mode survives to the image
    // there too -- but tagging plain AO "ao-base" would rename existing goldens.
    return shadingMode == ShadingMode::PRIMARY_RAYS ||
           (shadingMode == ShadingMode::AO && colorMode != ColorMode::BASE_COLOR);
}

std::string Shot::Tag() const
{
    std::string tag = NameOf(shadingMode);
    // PT still runs the colour mode through the BRDF rather than showing it flat
    // (see ShowsColorMode), but the file still needs a distinct name per mode.
    if (ShowsColorMode() || (shadingMode == ShadingMode::PT && colorMode != ColorMode::BASE_COLOR))
        tag += "-" + NameOf(colorMode);
    if (wireframe)
        tag += "-wf";
    return tag;
}

float ShotExposure(const Shot& shot, const ShotPosition& pos, const ShotConfig& config)
{
    if (shot.exposure > 0.f)
        return shot.exposure;
    if (shot.ShowsColorMode())
        return 1.f;
    return pos.exposure > 0.f ? pos.exposure : config.exposure;
}

namespace
{

// Optional, but > 0 when present; absent leaves `value` alone.
bool ReadExposure(const Json::Value& node, const std::string& owner, float& value)
{
    const Json::Value& exposure = node["exposure"];
    if (exposure.isNull())
        return true;
    if (!exposure.isNumeric() || exposure.asFloat() <= 0.f)
    {
        warning("--shot-list '%s': 'exposure' must be a number > 0", owner.c_str());
        return false;
    }
    value = exposure.asFloat();
    return true;
}

// Shared by LoadShotList and the two writers.
bool ReadShotListRoot(const fs::path& path, Json::Value& root)
{
    try
    {
        root = readFile(path);
    }
    catch (const std::exception& e)
    {
        warning("--shot-list: %s", e.what());
        return false;
    }
    return true;
}

bool WriteShotListRoot(const fs::path& path, const Json::Value& root)
{
    std::ofstream ofs(path);
    if (!ofs)
    {
        warning("--shot-list: cannot open '%s' for writing", path.generic_string().c_str());
        return false;
    }
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    ofs << Json::writeString(builder, root);
    if (!ofs)
    {
        warning("--shot-list: failed writing '%s'", path.generic_string().c_str());
        return false;
    }
    return true;
}

bool ParsePosition(const Json::Value& node, Json::ArrayIndex i, ShotPosition& position)
{
    position.label = read<std::string>(node["label"], "");
    if (position.label.empty())
    {
        warning("--shot-list position %u: missing 'label'", i);
        return false;
    }
    position.camera = read<std::string>(node["camera"], "");
    if (position.camera.empty())
    {
        warning("--shot-list '%s': missing 'camera'", position.label.c_str());
        return false;
    }
    if (Camera probe; !probe.Set(position.camera))
    {
        warning("--shot-list '%s': malformed camera '%s'; expected [eye][lookat][up]fovY",
                position.label.c_str(), position.camera.c_str());
        return false;
    }
    return ReadExposure(node, position.label, position.exposure);
}

bool ParseConfig(const Json::Value& node, Json::ArrayIndex i, ShotConfig& config)
{
    config.name = read<std::string>(node["name"], "");
    if (config.name.empty())
    {
        warning("--shot-list config %u: missing 'name'", i);
        return false;
    }

    Json::StreamWriterBuilder oneLine;
    oneLine["indentation"] = "";
    for (const std::string& key : node.getMemberNames())
        if (key != "name" && !key.starts_with('_'))
            config.settingsText += key + ": " + Json::writeString(oneLine, node[key]) + "\n";

    // Presence is checked against the positions in LoadShotList.
    if (!ReadExposure(node, config.name, config.exposure))
        return false;

    if (const Json::Value& settle = node["settle"]; settle.isNumeric())
        config.settleFrames = settle.asInt();
    else if (settle.isString() && settle.asString() != "auto")
    {
        warning("--shot-list '%s': 'settle' must be a frame count or \"auto\"",
                config.name.c_str());
        return false;
    }
    config.settleCap = read<uint32_t>(node["settleCap"], config.settleCap);

    // Absent = inherit the CLI value.  Present but malformed is an error rather
    // than a silent no-op: a sweep whose settings never applied produces N
    // identical configs that look like a real result.
    if (const Json::Value& dlss = node["dlssMode"]; dlss.isString())
    {
        const auto it = kDlssModes.find(dlss.asString());
        if (it == kDlssModes.end())
        {
            warning("--shot-list '%s': unknown dlssMode '%s'",
                    config.name.c_str(), dlss.asString().c_str());
            return false;
        }
        config.dlssMode    = it->second;
        config.dlssModeSet = true;
    }
    else if (!dlss.isNull())
    {
        warning("--shot-list '%s': 'dlssMode' must be a string", config.name.c_str());
        return false;
    }

    if (const Json::Value& lpe = node["lodPixelError"]; lpe.isNumeric())
    {
        config.lodPixelError = lpe.asFloat();
        if (config.lodPixelError <= 0.f)
        {
            warning("--shot-list '%s': 'lodPixelError' must be > 0", config.name.c_str());
            return false;
        }
    }
    else if (!lpe.isNull())
    {
        warning("--shot-list '%s': 'lodPixelError' must be a number", config.name.c_str());
        return false;
    }

    if (const Json::Value& res = node["resolution"]; res.isArray())
    {
        if (res.size() != 2 || !res[0].isIntegral() || !res[1].isIntegral() ||
            res[0].asInt() <= 0 || res[1].asInt() <= 0)
        {
            warning("--shot-list '%s': 'resolution' must be [width, height]",
                    config.name.c_str());
            return false;
        }
        config.outputWidth  = res[0].asInt();
        config.outputHeight = res[1].asInt();
    }
    else if (!res.isNull())
    {
        warning("--shot-list '%s': 'resolution' must be [width, height]", config.name.c_str());
        return false;
    }

    if (const Json::Value& adaptive = node["adaptiveLodError"]; adaptive.isBool())
        config.adaptiveLodError = adaptive.asBool() ? 1 : 0;
    else if (!adaptive.isNull())
    {
        warning("--shot-list '%s': 'adaptiveLodError' must be true or false",
                config.name.c_str());
        return false;
    }

    if (const Json::Value& nms = node["normalMapShading"]; nms.isBool())
        config.normalMapShading = nms.asBool() ? 1 : 0;
    else if (!nms.isNull())
    {
        warning("--shot-list '%s': 'normalMapShading' must be true or false",
                config.name.c_str());
        return false;
    }

    const Json::Value& shots = node["shots"];
    if (!shots.isNull() && !shots.isArray())
    {
        warning("--shot-list '%s': 'shots' must be an array", config.name.c_str());
        return false;
    }

    for (const Json::Value& shotNode : shots)
    {
        Shot              shot;
        const std::string mode = read<std::string>(shotNode["mode"], "pt");
        const auto        modeIt = kShadingModes.find(mode);
        if (modeIt == kShadingModes.end())
        {
            warning("--shot-list '%s': unknown mode '%s'", config.name.c_str(), mode.c_str());
            return false;
        }
        shot.shadingMode = modeIt->second;

        if (const std::string cm = read<std::string>(shotNode["colorMode"], ""); !cm.empty())
        {
            const auto cmIt = kColorModes.find(cm);
            if (cmIt == kColorModes.end())
            {
                warning("--shot-list '%s': unknown colorMode '%s'",
                        config.name.c_str(), cm.c_str());
                return false;
            }
            shot.colorMode = cmIt->second;
        }

        if (const Json::Value& wf = shotNode["wireframe"]; wf.isBool())
            shot.wireframe = wf.asBool();
        else if (!wf.isNull())
        {
            warning("--shot-list '%s': 'wireframe' must be true or false", config.name.c_str());
            return false;
        }

        // A primary-ray debug mode has no stochastic sampling to converge; AO
        // takes one visibility sample a frame and the path tracer needs a long
        // accumulation.  None of the counts covers the denoiser's own
        // re-accumulation floor.
        const uint32_t defaultFrames = shot.shadingMode == ShadingMode::PT ? 128u
                                     : shot.shadingMode == ShadingMode::AO ? 64u
                                                                           : 8u;
        shot.frames = read<uint32_t>(shotNode["frames"], defaultFrames);
        if (!ReadExposure(shotNode, config.name, shot.exposure))
            return false;
        config.shots.push_back(shot);
    }

    return true;
}

}  // anonymous namespace

bool LoadShotList(const fs::path& path, std::vector<ShotPosition>& positions,
                   std::vector<ShotConfig>& configs)
{
    Json::Value root;
    if (!ReadShotListRoot(path, root))
        return false;

    if (!root.isObject())
    {
        warning("--shot-list '%s': expected an object with 'positions'/'shotConfigs'",
                path.generic_string().c_str());
        return false;
    }

    const Json::Value& positionsNode = root["positions"];
    if (!positionsNode.isArray() || positionsNode.empty())
    {
        warning("--shot-list '%s': 'positions' must be a non-empty array",
                path.generic_string().c_str());
        return false;
    }
    for (Json::ArrayIndex i = 0; i < positionsNode.size(); ++i)
    {
        ShotPosition position;
        if (!ParsePosition(positionsNode[i], i, position))
            return false;
        positions.push_back(std::move(position));
    }

    const Json::Value& configsNode = root["shotConfigs"];
    if (!configsNode.isArray() || configsNode.empty())
    {
        warning("--shot-list '%s': 'shotConfigs' must be a non-empty array",
                path.generic_string().c_str());
        return false;
    }
    for (Json::ArrayIndex i = 0; i < configsNode.size(); ++i)
    {
        ShotConfig config;
        if (!ParseConfig(configsNode[i], i, config))
            return false;
        configs.push_back(std::move(config));
    }

    // Deliberately no default: auto-exposure drifts with scene luminance, so a
    // shot that inherited it would produce a golden that moves on its own.
    for (const ShotConfig& config : configs)
    {
        if (config.exposure > 0.f)
            continue;
        for (const ShotPosition& position : positions)
        {
            if (position.exposure <= 0.f)
            {
                warning("--shot-list '%s': config '%s' has no 'exposure' and position '%s' "
                        "does not pin one either.  Auto-exposure is temporal and would "
                        "drift the golden.",
                        path.generic_string().c_str(), config.name.c_str(),
                        position.label.c_str());
                return false;
            }
        }
    }

    return true;
}

bool SaveShotPosition(const fs::path& path, size_t index, const std::string& cameraString,
                      float exposure)
{
    Json::Value root;
    if (!ReadShotListRoot(path, root))
        return false;

    Json::Value& positionsNode = root["positions"];
    if (!positionsNode.isArray() || index >= positionsNode.size())
    {
        warning("--shot-list '%s': position index %zu out of range",
                path.generic_string().c_str(), index);
        return false;
    }
    Json::Value& position = positionsNode[static_cast<Json::ArrayIndex>(index)];
    position["camera"]   = cameraString;
    position["exposure"] = exposure;

    return WriteShotListRoot(path, root);
}

bool AppendShotPosition(const fs::path& path, const std::string& label,
                         const std::string& cameraString, float exposure)
{
    Json::Value root;
    if (!fs::exists(path))
    {
        // A new list starts with one path-traced config; positions pin exposure.
        Json::Value shot;
        shot["mode"] = "pt";
        Json::Value config;
        config["name"]   = "beauty";
        config["settle"] = "auto";
        config["shots"].append(shot);
        root["positions"] = Json::Value(Json::arrayValue);
        root["shotConfigs"].append(config);
    }
    else if (!ReadShotListRoot(path, root))
        return false;

    Json::Value& positionsNode = root["positions"];
    if (!positionsNode.isArray())
    {
        warning("--shot-list '%s': 'positions' must be an array",
                path.generic_string().c_str());
        return false;
    }
    Json::Value newPosition;
    newPosition["label"]    = label;
    newPosition["camera"]   = cameraString;
    newPosition["exposure"] = exposure;
    positionsNode.append(std::move(newPosition));

    return WriteShotListRoot(path, root);
}

bool DeleteShotPosition(const fs::path& path, size_t index)
{
    Json::Value root;
    if (!ReadShotListRoot(path, root))
        return false;

    Json::Value& positionsNode = root["positions"];
    if (!positionsNode.isArray() || index >= positionsNode.size())
    {
        warning("--shot-list '%s': position index %zu out of range",
                path.generic_string().c_str(), index);
        return false;
    }
    if (positionsNode.size() == 1)
    {
        warning("--shot-list '%s': cannot delete the only position",
                path.generic_string().c_str());
        return false;
    }
    Json::Value removed;
    positionsNode.removeIndex(static_cast<Json::ArrayIndex>(index), &removed);

    return WriteShotListRoot(path, root);
}
