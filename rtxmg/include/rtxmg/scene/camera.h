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

#pragma once

#include <donut/core/math/math.h>

using namespace donut::math;

#include <array>
#include <string>

class Camera
{

public:
    bool HasChanged() const { return m_changed; }

    float3 GetDirection() const
    {
        // eye == lookat happens on a scene switch with a degenerate bbox, and
        // normalize(0) would NaN the whole camera.
        const float3 d = m_lookat - m_eye;
        const float  len = length(d);
        return (len > 1e-6f) ? (d * (1.f / len)) : float3(0.f, 0.f, -1.f);
    }
    void SetDirection(const float3& dir)
    {
        m_lookat = m_eye + length(m_lookat - m_eye) * dir;
    }

    void Translate(float3 const& v);
    void Rotate(float yaw, float pitch, float roll);
    void Roll(float speed);

    void Dolly(float factor);
    void Pan(float2 speed);
    void Zoom(const float factor);

    void Frame(box3 const& aabb);

    // UVW forms an orthogonal, but not orthonormal basis!
    std::array<float3, 3> const& GetBasis();

    void Print() const;
    // Same syntax Set() parses: "[eye][lookat][up]fovY".
    std::string ToString() const;

    float3 GetEye() const { return m_eye; }
    float3 GetLookat() const { return m_lookat; }
    float3 GetUp() const { return m_up; }

    float GetFovY() const { return m_fovY; }
    float GetAspectRatio() const { return m_aspectRatio; }
    float GetZNear() const { return m_zNear; }
    float GetZFar() const { return m_zFar; }

    // These return column vectors but are stored in row_major memory wise
    // Translation is in m[3][j]  
    float4x4 GetViewMatrix() const;
    float4x4 GetProjectionMatrix() const;
    float4x4 GetViewProjectionMatrix() const;

    void SetEye(float3 eye);
    void SetLookat(float3 lookat);
    void SetUp(float3 up);

    void SetFovY(float fovy);
    void SetAspectRatio(float ar);
    void SetNear(float near);
    void SetFar(float far);

    // Parses "[eye][lookat][up]fovY"; leaves the camera untouched and returns
    // false on anything malformed or degenerate.
    bool Set(std::string const& camc_string);

private:
    void ComputeBasis(float3& u, float3& v, float3& w) const;

    std::array<float3, 3> m_basis = {};

    float3 m_eye = float3(1.f);
    float3 m_lookat = float3(0.f);
    float3 m_up = float3(0.f, 1.f, 0.f);

    float m_fovY = 35.f;
    float m_aspectRatio = 1.f;
    float m_zNear = 0.1f;
    float m_zFar = 100.f;

    bool m_changed = true;
};