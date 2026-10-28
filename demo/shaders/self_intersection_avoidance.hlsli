//
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 

// Compute the object and world space position and normal corresponding to a triangle hit point.
// Compute a safe spawn point offset along the normal in world space to prevent self intersection of secondary rays.
void SafeSpawnPoint(
    out float3     outObjPosition, // position in object space
    out float3     outWldPosition, // position in world space    
    out float3     outObjNormal,   // unit length surface normal in object space
    out float3     outWldNormal,   // unit length surface normal in world space
    out float      outWldOffset,   // safe offset for spawn position in world space
    const float3   v0,             // spawn triangle vertex 0 in object space
    const float3   v1,             // spawn triangle vertex 1 in object space
    const float3   v2,             // spawn triangle vertex 2 in object space
    const float2   bary,           // spawn barycentrics
    const float3x4 o2w,            // spawn instance object-to-world transformation
    const float3x4 w2o )           // spawn instance world-to-object transformation
{
    precise float3 edge1 = v1 - v0;
    precise float3 edge2 = v2 - v0;

    // interpolate triangle using barycentrics.
    // add in base vertex last to reduce object space error.
    precise float3 objPosition = v0 + mad( bary.x, edge1, mul( bary.y, edge2 ) );
    float3 objNormal = cross( edge1, edge2 );

    // transform object space position.
    // add in translation last to reduce world space error.
    // Built from scalars: SM 6.9+ DXC cannot compile component writes into a precise vector.
    precise float wldX = o2w._m03 +
        mad( o2w._m00, objPosition.x,
            mad( o2w._m01, objPosition.y,
                mul( o2w._m02, objPosition.z ) ) );
    precise float wldY = o2w._m13 +
        mad( o2w._m10, objPosition.x,
            mad( o2w._m11, objPosition.y,
                mul( o2w._m12, objPosition.z ) ) );
    precise float wldZ = o2w._m23 +
        mad( o2w._m20, objPosition.x,
            mad( o2w._m21, objPosition.y,
                mul( o2w._m22, objPosition.z ) ) );
    precise float3 wldPosition = float3( wldX, wldY, wldZ );

    // transform normal to world - space using
    // inverse transpose matrix
    float3 wldNormal = mul( transpose( ( float3x3 )w2o ), objNormal );

    // normalize world space normal
    const float wldScale = rsqrt( dot( wldNormal, wldNormal ) );
    wldNormal = mul( wldScale, wldNormal );

    const float c0 = 5.9604644775390625E-8f;
    const float c1 = 1.788139769587360206060111522674560546875E-7f;

    const float3 extent3 = abs( edge1 ) + abs( edge2 ) + abs( edge1 - edge2 );
    const float  extent = max( max( extent3.x, extent3.y ), extent3.z );

    // bound object space error due to reconstruction and intersection
    float3 objErr = mad( c0, abs( v0 ), mul( c1, extent ) );

    // bound world space error due to object to world transform
    const float c2 = 1.19209317972490680404007434844970703125E-7f;
    float3 wldErr = mad( c1, mul( abs( ( float3x3 )o2w ), abs( objPosition ) ), mul( c2, abs( transpose( o2w )[3] ) ) );

    // bound object space error due to world to object transform
    objErr = mad( c2, mul( abs( w2o ), float4( abs( wldPosition ), 1 ) ), objErr );

    // compute world space self intersection avoidance offset
    float wldOffset = dot( wldErr, abs( wldNormal ) );
    float objOffset = dot( objErr, abs( objNormal ) );

    wldOffset = mad( wldScale, objOffset, wldOffset );

    // output safe front and back spawn points
    outObjPosition = objPosition;
    outWldPosition = wldPosition;
    outObjNormal = normalize( objNormal );
    outWldNormal = wldNormal;
    outWldOffset = wldOffset;
}

// Offset the world-space position along the world-space normal by the safe offset to obtain the safe spawn point.
float3 SafeSpawnPoint(
    const float3 position,
    const float3 normal,
    const float  offset )
{
    precise float3 p = mad( offset, normal, position );
    return p;
}
