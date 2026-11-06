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


#include "rtxmg/profiler/profiler.h"
#include "rtxmg/profiler/stopwatch.h"


#include <cassert>
#include <type_traits>
#include <variant>

// clang-format on
static Profiler *s_profiler = nullptr;

Profiler& Profiler::Get()
{
    if (!s_profiler)
        s_profiler = new Profiler();
    return *s_profiler;
}

void Profiler::Terminate()
{
    delete s_profiler;
}

using CPUTimer = Profiler::Timer<StopwatchCPU>;
using GPUTimer = Profiler::Timer<StopwatchGPU>;

template <>
CPUTimer& Profiler::Timer<StopwatchCPU>::Resolve()
{
    if( auto e = Elapsed() )
        PushBack( *e );
    return *this;
}
template <>
GPUTimer& Profiler::Timer<StopwatchGPU>::Resolve()
{
    ElapsedAsync( [this]( float e ) { PushBack( e ); } );
    return *this;
}

template <>
CPUTimer& Profiler::Timer<StopwatchCPU>::Profile()
{
    if (Profiler::Get().IsRecording())
    {
        auto e = Elapsed();
        PushBack( e ? *e : 0.f );
    }
    return *this;
}
template <>
GPUTimer& Profiler::Timer<StopwatchGPU>::Profile()
{
    const bool recording = Profiler::Get().IsRecording();
    ElapsedAsync([this, recording](float e) { if (recording) PushBack(e); });
    return *this;
}

void Profiler::FrameStart( std::chrono::steady_clock::time_point time )
{
    int frequency = recordingFrequency;
    if( frequency > 0 )
    {
        double period = 1000. / double( frequency );

        m_isRecording = std::chrono::duration<double, std::milli>( time - m_prevTime ).count() >= period
                        || m_prevTime == std::chrono::steady_clock::time_point{};
    }
    else if( frequency < 0 )
        m_isRecording = true;
    else
        m_isRecording = false;

    if (m_isRecording)
        m_prevTime = time;
}

void Profiler::FrameEnd()
{

}

void Profiler::FrameResolve()
{
    auto resolveTimers = []( auto& timers ) {
        for( auto& timer : timers )
            timer->Resolve();
    };

    resolveTimers( m_cpuTimers );

    if( !m_gpuTimers.empty() )
    {
        resolveTimers( m_gpuTimers );
    }
}
