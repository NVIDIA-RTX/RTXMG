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

#pragma once

#include "rtxmg_demo.h"

#include <donut/app/StreamlineInterface.h>

#include <filesystem>
#include <string>
#include <vector>

// clang-format on

// --shot-list <file.json>: camera positions and capture configs walked in one
// process (every config against every position), so a scene load is amortized
// across every golden image it produces rather than paid once per screenshot.
//
// {
//   "positions": [
//     { "label": "board-closeup", "camera": "[1.2 0.8 -0.4][0 0.1 0][0 1 0]45" },
//     { "label": "dome", "camera": "[53 12 2.8][149 105 -43][0 1 0]35", "exposure": 32 }
//   ],
//   "shotConfigs": [
//     {
//       "name":     "beauty",
//       "exposure": 1.5,
//       "settle":   "auto",
//       "dlssMode": "quality",
//       "lodPixelError": 0.5,
//       "normalMapShading": true,
//       "shots":    [ { "mode": "pt", "frames": 128 },
//                     { "mode": "primary", "colorMode": "base" },
//                     { "mode": "ao", "colorMode": "normal", "wireframe": true } ]
//     },
//     { "name": "stats", "shots": [] }
//   ]
// }
//
// Positions carry a pose plus, optionally, an exposure -- the one setting that
// depends on where the camera stands rather than on what is captured.  A
// position's exposure overrides its config's; a config may omit exposure only
// if every position pins one.  A run walks every config against every position (config outer,
// so resolution/DLSS/LPE -- which reallocate -- change once per config rather
// than once per position).  A config's "shots" may be empty: it still jumps
// and settles at every position (so e.g. --dump-stats gets a number there)
// but captures nothing.
//
// The setting keys besides "name"/"shots" are optional and inherit the run's
// CLI values when absent, so a settings sweep is N configs that differ only in
// those.  A shot may carry its own "exposure"; primary-ray modes default to
// 1.0 rather than the position's or config's.

struct Shot
{
    ShadingMode shadingMode = ShadingMode::PT;
    ColorMode   colorMode   = ColorMode::BASE_COLOR;

    // Absolute, not inherited from -wf: a shot list states its shots in full,
    // the same way it does mode and colour mode.
    bool        wireframe   = false;

    // Frames to accumulate after applying the mode, before the capture.  A
    // path-traced shot needs many; a primary-ray debug mode converges at once.
    uint32_t    frames      = 0;  // 0 = per-mode default

    float       exposure    = 0.f;  // 0 = inherit, see ShotExposure

    // True when the colour mode reaches the image unshaded, which is what makes
    // it worth naming in the tag and worth capturing at exposure 1.  The path
    // tracer runs it through the BRDF, so there it is not a distinct look.
    bool ShowsColorMode() const;

    // "<label>.<tag>", the screenshot stem.
    std::string Tag() const;
};

struct ShotPosition
{
    std::string label;
    std::string camera;   // same syntax as -p / --cameraPos
    float       exposure = 0.f;  // 0 = use the config's
};

struct ShotConfig
{
    std::string       name;
    std::string       settingsText;    // "key: value" per line as written in the file, for the UI
    float             exposure = 0.f;  // 0 = every position pins its own
    std::vector<Shot> shots;   // may be empty: settle (+ stats dump) only, no capture

    // Settings applied at the camera cut, before the settle, so residency
    // converges under the settings the shot is captured with.  eOff is itself a
    // mode, so the flag carries "unset" rather than a sentinel value.
    bool                                      dlssModeSet = false;
    donut::app::StreamlineInterface::DLSSMode dlssMode =
        donut::app::StreamlineInterface::DLSSMode::eOff;
    float   lodPixelError    = 0.f;  // 0 = inherit -lpe
    int32_t adaptiveLodError = -1;   // -1 = inherit --adaptive-error, else 0/1
    // Shading only; loading normal maps is a scene-reload setting, so a list
    // that toggles this has to run with --normalmaps for either state to mean
    // anything.  -1 = inherit --normalmapshading, else 0/1.
    int32_t normalMapShading = -1;

    // Output resolution, i.e. the window's client size.  0 = leave it alone.
    // Ignored in fullscreen, where the display drives it -- so a matrix that
    // spans resolutions runs windowed for the small ones and -fs for native.
    int32_t outputWidth  = 0;
    int32_t outputHeight = 0;

    // Frames to wait for streaming residency after the camera cut.  -1 means
    // watch the residency counters go flat instead of using a fixed count.
    int32_t settleFrames = -1;
    // Upper bound on the "auto" wait, so a scene that never converges still
    // finishes rather than hanging the run.
    uint32_t settleCap = 1500;
};

// The shot's own exposure, else 1.0 for a mode that decodes flat values (a
// beauty exposure just blows it out), else the position's, else the config's.
float ShotExposure(const Shot& shot, const ShotPosition& pos, const ShotConfig& config);

// Parses `path`.  Returns false and logs on a malformed file, an unknown mode,
// a missing camera, or a position left with no exposure -- auto-exposure is a temporal feedback loop, so
// a config that inherits it would drift for reasons unrelated to any change.
bool LoadShotList(const std::filesystem::path& path,
                   std::vector<ShotPosition>& positions,
                   std::vector<ShotConfig>& configs);

// Rewrites positions[index]'s "camera" and "exposure" in the shot-list file at
// `path`; every other field/array is left exactly as parsed.  False (and logs)
// on a read/parse/write failure or an out-of-range index.
bool SaveShotPosition(const std::filesystem::path& path, size_t index,
                      const std::string& cameraString, float exposure);

// Appends {label, camera, exposure} to the file's "positions" array, creating
// the file with a single path-traced "beauty" config if it doesn't exist.
bool AppendShotPosition(const std::filesystem::path& path, const std::string& label,
                         const std::string& cameraString, float exposure);

// Removes positions[index]; refuses to remove the last one, which would leave
// a file LoadShotList rejects.
bool DeleteShotPosition(const std::filesystem::path& path, size_t index);

// The --dlssMode spelling of `mode`, for the stats dump.
const char* DlssModeName(donut::app::StreamlineInterface::DLSSMode mode);
