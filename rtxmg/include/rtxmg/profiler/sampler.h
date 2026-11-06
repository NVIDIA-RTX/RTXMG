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

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <string>
#include <vector>

// clang-format on

// A generic data sampler with basic statistics functionality

template <typename T, size_t _size = 60>
struct Sampler : public std::array<T, _size>
{
    std::string name;

    // values tracked since most recent reset
    size_t samples_count = 0;
    T      latest        = {};
    T      total         = T( 0 );
    T      min           = std::numeric_limits<T>::max();
    T      max           = std::numeric_limits<T>::lowest();

    std::array<std::chrono::steady_clock::time_point, _size> push_times = {};

    void PushBack( T sample );

    void Reset();

    // Over the retained window, not the whole history: only the last _size
    // samples are still around to sort.
    T Median() const { return Percentile( 0.5 ); }
    T Percentile( double p ) const
    {
        const size_t n = std::min( _size, samples_count );
        if( n == 0 )
            return T( 0 );
        std::vector<T> sorted( this->begin(), this->begin() + n );
        const size_t   k = std::min( n - 1, size_t( p * double( n ) ) );
        std::nth_element( sorted.begin(), sorted.begin() + k, sorted.end() );
        return sorted[k];
    }
    T Average() const { return static_cast<T>( double( total ) / double( samples_count ) ); }

    // Mean of the samples pushed in the last second (at least the latest one), so
    // the UI reads smoothly at any frame rate yet recovers from a spike quickly.
    T RunningAverage() const
    {
        const size_t n      = std::min( _size, samples_count );
        const auto   cutoff = std::chrono::steady_clock::now() - std::chrono::seconds( 1 );
        T            sum    = T( 0 );
        size_t       count  = 0;
        for( size_t i = 1; i <= n; ++i )
        {
            const size_t idx = ( samples_count - i ) % _size;
            if( count > 0 && push_times[idx] < cutoff )
                break;
            sum += ( *this )[idx];
            ++count;
        }
        return static_cast<T>( sum / double( count ) );
    }

    // current position in circular buffer
    uint32_t Offset() const { return static_cast<uint32_t>( samples_count % _size ); }

    void Print()
    {
        size_t n = std::min( _size, samples_count );
        for(size_t i = 0; i < n; i++ )
        {
            std::printf( "Benchmark: frame %d time %.4f ms\n", (int)i, (*this)[i]);
        }
    }
};

template <typename T, size_t _size>
inline void Sampler<T, _size>::PushBack( T sample )
{
    latest = sample;
    total += latest;
    min = std::min( latest, min );
    max = std::max( latest, max );

    push_times[samples_count % _size]  = std::chrono::steady_clock::now();
    ( *this )[samples_count++ % _size] = sample;
}

template <typename T, size_t _size>
inline void Sampler<T, _size>::Reset()
{
    samples_count = 0;
    latest        = {};
    total         = T( 0 );
    min           = std::numeric_limits<T>::max();
    max           = std::numeric_limits<T>::lowest();
#if !defined( NDEBUG )
    this->fill( T( 0 ) );
#endif
}