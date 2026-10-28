/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef WAVE_SIZE_HLSLI // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define WAVE_SIZE_HLSLI

// Pins a compute shader written for 32-lane waves. SPIR-V cannot express a wave size, so on
// Vulkan the startup check requires the device to run 32 lanes instead.
#if defined(TARGET_D3D12)
#define RTXMG_WAVE_SIZE_32 [WaveSize(32)]
#else
#define RTXMG_WAVE_SIZE_32
#endif

#endif // WAVE_SIZE_HLSLI
