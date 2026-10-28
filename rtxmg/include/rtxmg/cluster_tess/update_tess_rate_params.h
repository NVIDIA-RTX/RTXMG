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
//

#pragma once

struct UpdateTessRateParams
{
    // Tessellation counter slots of this build and the two before it
    uint32_t slot;
    uint32_t prevSlot;
    uint32_t prev2Slot;
    uint32_t numPrevBuilds;  // 0..2 valid previous slots

    uint32_t maxClusters;
    uint32_t maxVertices;
    uint32_t maxClasBlocks;
    uint32_t enable;
};
