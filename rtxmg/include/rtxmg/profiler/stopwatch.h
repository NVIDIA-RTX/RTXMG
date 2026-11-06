/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// clang-format off

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <optional>

#include <nvrhi/nvrhi.h>

// clang-format on

class StopwatchCPU
{
public:
    void Start();
    void Stop();

    std::optional<float> Elapsed();  // returns dt = stop - start
    std::optional<float> Before(std::chrono::steady_clock::time_point t);
    std::optional<float> After(std::chrono::steady_clock::time_point t);

private:
    using steady_clock = std::chrono::steady_clock;
    using duration = std::chrono::duration<double, std::milli>;

    steady_clock::time_point m_startTime;
    steady_clock::time_point m_stopTime;
};

class StopwatchGPU
{
public:
    void Start(nvrhi::ICommandList* commandList);
    void Stop();

    std::optional<float> Elapsed();      // returns dt = stop - start

    // Calls sink(ms) once per query resolved since the last call, oldest first.
    template <typename Sink>
    void ElapsedAsync(Sink&& sink);
private:

    void ProcessUnresolvedQueries();

    // MUST exceed DeviceCreationParameters::maxFramesInFlight + 1: one slot for
    // the frame being recorded plus one per frame the GPU may still be running,
    // or Start reuses a slot before it resolves and the timer wedges on its last
    // value.  At 3 (== maxFramesInFlight + 1) there was no slack and one heavy
    // frame was enough to trip it.
    static constexpr uint32_t kMaxInFlightQueries = 8;

    nvrhi::DeviceHandle m_device;
    std::array<nvrhi::TimerQueryHandle, kMaxInFlightQueries> m_timerQueries;
    nvrhi::CommandListHandle m_commandList;
    int32_t m_queryIndex = -1;
    int32_t m_unresolvedQueryIndex = -1;
    float m_lastDuration = 0.f;
    std::array<float, kMaxInFlightQueries> m_resolved = {};
    uint32_t m_resolvedCount = 0;

    enum class State : uint8_t
    {
        uninitialized = 0,
        reset,
        ticking,
        stopped
    } state = State::uninitialized;
};

template <typename Sink>
void StopwatchGPU::ElapsedAsync(Sink&& sink)
{
    if (state == State::reset || state == State::uninitialized)
        return;

    // user is responsible for device sync, so we can't track it
    assert(state != State::ticking);

    state = State::reset;

    ProcessUnresolvedQueries();
    for (uint32_t i = 0; i < m_resolvedCount; ++i)
        sink(m_resolved[i] * 1000.0f);
    m_resolvedCount = 0;
}
