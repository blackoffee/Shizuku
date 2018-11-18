#pragma once
#include "cuda_runtime.h"

__device__ float DotProduct(const float3 &u, const float3 &v);

__device__ float3 CrossProduct(const float3 &u, const float3 &v);

__device__ float CrossProductArea(const float2 &u, const float2 &v);

__device__ void Normalize(float3 &u);

__device__ float Distance(const float3 &u, const float3 &v);

__device__ bool IsPointsOnSameSide(const float2 &p1, const float2 &p2,
    const float2 &a, const float2 &b);

__device__ bool IsPointInsideTriangle(const float2 &p, const float2 &a,
    const float2 &b, const float2 &c);

__device__ bool IsPointInsideTriangle(const float3 &p1, const float3 &p2,
    const float3 &p3, const float3 &q);

__device__ float GetDistanceBetweenPointAndLineSegment(const float3 &p1, const float3 &q1, const float3 &q2);

__device__ float GetDistanceBetweenTwoLines(const float3 &p1, const float3 &p2, const float3 &q1, const float3 &q2);

// ! geomalgorithms.com/a07-_distance.html
__device__ float GetDistanceBetweenTwoLineSegments(const float3 &p1, const float3 &p2, const float3 &q1, const float3 &q2);

// Gets intersection of line with plane created by triangle
//p1, p2, p3 should be in clockwise order
__device__ float3 GetIntersectionOfLineWithTriangle(const float3 &lineOrigin,
    float3 &lineDir, const float3 &p1, const float3 &p2, const float3 &p3);

// Gets intersection of line segment with plane created by triangle
//p1, p2, p3 should be in clockwise order
__device__ bool GetIntersectionOfLineSegmentWithTriangle(float3 &intersect, const float3 &lineOrigin,
    float3 &lineDest, const float3 &p1, const float3 &p2, const float3 &p3);

// Only update intersect reference if intersect is inside the rectangle, and is closer to lineOrigin than previous value
__device__ bool IntersectLineSegmentWithRect(float3 &intersect, float3 lineOrigin, float3 lineDest,
    float3 topLeft, float3 topRight, float3 bottomRight, float3 bottomLeft);

__device__ float3 operator+(const float3 &u, const float3 &v);

__device__ float2 operator+(const float2 &u, const float2 &v);

__device__ float3 operator-(const float3 &u, const float3 &v);

__device__ float2 operator-(const float2 &u, const float2 &v);

__device__ float3 operator*(const float3 &u, const float3 &v);

__device__ float3 operator/(const float3 &u, const float3 &v);

__device__ float3 operator*(const float a, const float3 &u);

__device__ float3 operator/(const float3 &u, const float a);

