//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

#include "rtxmg/scene/box_extent.h"
#include "rtxmg/scene/camera.h"

#include <donut/core/log.h>
#include "rtxmg/utils/buffer.h"

void Camera::Translate(float3 const& v)
{
    m_eye += v;
    m_lookat += v;
    m_changed = true;
}

void Camera::Rotate(float yaw, float pitch, float roll)
{
    float3 w = m_lookat - m_eye;
    float wlen = length(w);
    float3 u = normalize(cross(w, m_up));

    affine3 rotation = yawPitchRoll(yaw, pitch, roll);

    float3 dir = rotation.transformVector(float3(w.x, w.y, w.z));
    float3 m_up = rotation.transformVector(m_up);

    m_lookat = m_eye + dir * wlen;

    m_changed = true;
}

void Camera::Roll(float speed)
{
    auto const& basis = GetBasis();
    float3 u = normalize(basis[0]);
    float3 v = normalize(basis[1]);
    m_up = u * cosf(radians(90.0f + speed)) + v * sinf(radians(90.0f + speed));
    m_changed = true;
}

void Camera::Pan(float2 speed)
{
    auto const& basis = GetBasis();
    float3 u = basis[0] * (-2.f * speed.x);
    float3 v = basis[1] * (-2.f * speed.y);
    Translate(u + v);
}

void Camera::Dolly(float factor)
{
    // move closer by factor
    float3 oldEyeOffset = m_eye - m_lookat;
    m_eye = m_lookat + (oldEyeOffset * factor);
    m_changed = true;
}

void Camera::Zoom(const float factor)
{
    // increase/decrease field-of-view angle by factor
    m_fovY = fminf(150.f, m_fovY * factor);
    m_changed = true;
}

void Camera::Frame(box3 const& aabb)
{
    SetFovY(35.0f);
    SetLookat(aabb.center());
    SetEye(aabb.center() + 1.2f * MaxBoxExtent(aabb));
    SetUp({ 0.f, 1.f, 0.f });
    m_changed = true;
}

std::array<float3, 3> const& Camera::GetBasis()
{
    if (HasChanged())
    {
        ComputeBasis(m_basis[0], m_basis[1], m_basis[2]);
        m_changed = false;
    }
    return m_basis;
}

void Camera::ComputeBasis(float3& U, float3& V, float3& W) const
{
    float wlen = 0.f;

    W = m_lookat - m_eye; // Do not normalize W -- it implies focal length
    wlen = length(W);
    U = normalize(cross(W, m_up));
    V = normalize(cross(U, W));

    float vlen = wlen * tanf(0.5f * m_fovY * PI_f / 180.0f);
    V *= vlen;
    float ulen = vlen * m_aspectRatio;
    U *= ulen;
}

void Camera::SetEye(float3 eye)
{
    if (!all(isfinite(eye)))
    {
        donut::log::error("setEye is NaN!: %f %f %f", eye.x, eye.y, eye.z);
        return;
    }

    m_eye = eye;
    m_changed = true;
}
void Camera::SetLookat(float3 lookat)
{
    if (!all(isfinite(lookat)))
    {
        donut::log::error("lookat is NaN!: %f %f %f", lookat.x, lookat.y, lookat.z);
        return;
    }
    m_lookat = lookat;
    m_changed = true;
}
void Camera::SetUp(float3 up)
{
    m_up = up;
    m_changed = true;
}
void Camera::SetFovY(float fovy)
{
    m_fovY = fovy;
    m_changed = true;
}
void Camera::SetAspectRatio(float ar)
{
    m_aspectRatio = ar;
    m_changed = true;
}
void Camera::SetNear(float near)
{
    m_zNear = near;
    m_changed = true;
}
void Camera::SetFar(float far)
{
    m_zFar = far;
    m_changed = true;
}

static inline std::istream& operator>>(std::istream& is, float3& v)
{
    char st;
    is >> st >> v.x >> st >> v.y >> st >> v.z >> st;
    return is;
}

void Camera::Set(std::string const& camc_string)
{
    std::istringstream istr(camc_string);
    istr >> m_eye;
    istr >> m_lookat;
    istr >> m_up;
    istr >> m_fovY;
    m_changed = true;
}

void Camera::Print() const
{
    printf("[%g,%g,%g][%g,%g,%g][%g,%g,%g]%g\n", m_eye.x, m_eye.y, m_eye.z,
        m_lookat.x, m_lookat.y, m_lookat.z, m_up.x, m_up.y, m_up.z, m_fovY);
}

float4x4 Camera::GetViewMatrix() const
{
    float3 W = normalize(m_lookat - m_eye);
    float3 U = normalize(cross(W, m_up));
    float3 V = normalize(cross(U, W));

    float m[16];
    m[0] = U.x;
    m[1] = U.y;
    m[2] = U.z;
    m[3] = -dot(U, m_eye);

    m[4] = V.x;
    m[5] = V.y;
    m[6] = V.z;
    m[7] = -dot(V, m_eye);

    m[8] = -W.x;
    m[9] = -W.y;
    m[10] = -W.z;
    m[11] = dot(W, m_eye);

    m[12] = 0.f;
    m[13] = 0.f;
    m[14] = 0.f;
    m[15] = 1.0f;

    return float4x4(m);
}

float4x4 Camera::GetProjectionMatrix() const
{
    const float fovRad = m_fovY * PI_f / 180.f;
    const float tanFov = tan(fovRad / 2.f);

    float m[16];
    m[0] = 1.f / (tanFov * m_aspectRatio);
    m[1] = 0.f;
    m[2] = 0.f;
    m[3] = 0.f;

    m[4] = 0.f;
    m[5] = 1.f / tanFov;
    m[6] = 0.f;
    m[7] = 0.f;

    m[8] = 0.f;
    m[9] = 0.f;
    m[10] = (m_zFar + m_zNear) / (m_zNear - m_zFar);
    m[11] = 2.f * m_zFar * m_zNear / (m_zNear - m_zFar);

    m[12] = 0.f;
    m[13] = 0.f;
    m[14] = -1.f;
    m[15] = 0.f;

    return float4x4(m);
}

float4x4 Camera::GetViewProjectionMatrix() const
{
    return GetProjectionMatrix() * GetViewMatrix();
}
