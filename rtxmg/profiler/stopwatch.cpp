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

// clang-format off

#include "rtxmg/profiler/stopwatch.h"

#include <cassert>
#include <donut/core/log.h>

// clang-format on

//
// StopwatchCPU
//

void StopwatchCPU::Start()
{
    m_startTime = steady_clock::now();
    assert( m_stopTime < m_startTime );
}

void StopwatchCPU::Stop()
{
    assert( m_startTime.time_since_epoch().count() > 0 );
    m_stopTime = steady_clock::now();
}

std::optional<float> StopwatchCPU::Elapsed()
{
    assert( m_stopTime >= m_startTime );

    if( m_startTime == steady_clock::time_point{} )
        return {};

    float elapsed = static_cast<float>( duration( m_stopTime - m_startTime ).count() );
    m_startTime = m_stopTime = {};
    return elapsed;
}

std::optional<float> StopwatchCPU::Before( steady_clock::time_point t )
{
    assert( m_startTime >= t && m_startTime > steady_clock::time_point{});
    return (float)duration( m_startTime - t).count();
}

std::optional<float> StopwatchCPU::After( steady_clock::time_point t )
{
    assert( m_stopTime <= t && m_stopTime > steady_clock::time_point{});
    return (float)duration( t - m_stopTime ).count();
}

//
// StopwatchGPU
//
void StopwatchGPU::ProcessUnresolvedQueries()
{
    if (state == State::uninitialized)
        return;

    // New frame started
    // Check our previous queries
    uint32_t unresolvedQueryIndex = m_unresolvedQueryIndex;
    while(unresolvedQueryIndex != m_queryIndex)
    {
        unresolvedQueryIndex = (unresolvedQueryIndex + 1) % kMaxInFlightQueries;
        if (!m_device->pollTimerQuery(m_timerQueries[unresolvedQueryIndex]))
        {
            break;
        }
        // save the last one
        m_unresolvedQueryIndex = unresolvedQueryIndex;
        m_lastDuration = m_device->getTimerQueryTime(m_timerQueries[m_unresolvedQueryIndex]);
        if (m_resolvedCount < kMaxInFlightQueries)
            m_resolved[m_resolvedCount++] = m_lastDuration;
        m_device->resetTimerQuery(m_timerQueries[m_unresolvedQueryIndex]);
    }
}

void StopwatchGPU::Start(nvrhi::ICommandList* commandList)
{
    if (state == State::uninitialized)
    {
        m_device = commandList->getDevice();
        for (auto& query : m_timerQueries)
        {
            query = m_device->createTimerQuery();
        }
        m_queryIndex = -1;
        m_unresolvedQueryIndex = -1;
        m_resolvedCount = 0;
        state = State::reset;
    }

    ProcessUnresolvedQueries();
    
    // Start a new query. Assumption is one star/stop pair per frame
    m_queryIndex = (m_queryIndex + 1) % kMaxInFlightQueries;

    // all but 'stopped' states are valid, so can advance up to
    // kMaxInFlightQueries times
    assert(state != State::ticking);

    // When the ring wraps onto a slot whose result was never polled, the slot is
    // still marked started; nvrhi's Vulkan beginTimerQuery asserts on that (D3D12
    // tolerates it), so clear the stale state rather than crash on the overflow.
    m_device->resetTimerQuery(m_timerQueries[m_queryIndex]);

    commandList->beginTimerQuery(m_timerQueries[m_queryIndex]);
    m_commandList = commandList;
    m_device = commandList->getDevice();
    state = State::ticking;
}

void StopwatchGPU::Stop()
{
    assert(state == State::ticking);
    m_commandList->endTimerQuery(m_timerQueries[m_queryIndex]);
    state = State::stopped;
}

std::optional<float> StopwatchGPU::Elapsed()
{
    if( state == State::reset || state == State::uninitialized)
        return {};

    assert(state == State::stopped);

    state = State::reset;

    ProcessUnresolvedQueries();

    return m_lastDuration * 1000.0f;
}
