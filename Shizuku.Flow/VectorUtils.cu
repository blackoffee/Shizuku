#include "VectorUtils.h"
#include "Domain.h"

__device__ float DotProduct(const float3 &u, const float3 &v)
{
    return u.x*v.x + u.y*v.y + u.z*v.z;
}

__device__ float3 CrossProduct(const float3 &u, const float3 &v)
{
    return make_float3(u.y*v.z-u.z*v.y, -(u.x*v.z-u.z*v.x), u.x*v.y-u.y*v.x);
}

__device__ float CrossProductArea(const float2 &u, const float2 &v)
{
    return 0.5f*sqrt((u.x*v.y-u.y*v.x)*(u.x*v.y-u.y*v.x));
}

__device__ void Normalize(float3 &u)
{
    float mag = sqrt(DotProduct(u, u));
    u.x /= mag;
    u.y /= mag;
    u.z /= mag;
}

__device__ float Distance(const float3 &u, const float3 &v)
{
    return sqrt(DotProduct((u-v), (u-v)));
}

__device__ bool IsPointsOnSameSide(const float2 &p1, const float2 &p2,
    const float2 &a, const float2 &b)
{
    float cp1 = (b - a).x*(p1 - a).y - (b - a).y*(p1 - a).x;
    float cp2 = (b - a).x*(p2 - a).y - (b - a).y*(p2 - a).x;
    if (cp1*cp2 >= 0)
    {
        return true;
    }
    return false;
}

__device__ bool IsPointInsideTriangle(const float2 &p, const float2 &a,
    const float2 &b, const float2 &c)
{
    if (IsPointsOnSameSide(p, a, b, c) &&
        IsPointsOnSameSide(p, b, a, c) &&
        IsPointsOnSameSide(p, c, a, b))
    {
        return true;
    }
    return false;
}

__device__ bool IsPointInsideTriangle(const float3 &p1, const float3 &p2,
    const float3 &p3, const float3 &q)
{
    float3 n = CrossProduct((p2 - p1), (p3 - p1));

    if (DotProduct(CrossProduct(p2 - p1, q - p1), n) < 0) return false;
    if (DotProduct(CrossProduct(p3 - p2, q - p2), n) < 0) return false;
    if (DotProduct(CrossProduct(p1 - p3, q - p3), n) < 0) return false;

    return true;
}


__device__ float GetDistanceBetweenPointAndLineSegment(const float3 &p1, const float3 &q1, const float3 &q2)
{
    float3 q = q2 - q1;
    const float magQ = sqrt(DotProduct(q, q));
    float s = DotProduct(q2 - q1, p1 - q1) / magQ;

    Normalize(q);
    if (s > 0 && s < magQ)
    {
        return Distance(p1, q1 + s*q);
    }
    else
    {
        return dmin(Distance(p1, q1), Distance(p1, q2));
    }
}


__device__ float GetDistanceBetweenTwoLines(const float3 &p1, const float3 &p2, const float3 &q1, const float3 &q2)
{
    float3 n = CrossProduct(p2 - p1, q2 - q1);
    Normalize(n);
    return abs(DotProduct(p1 - q1, n));
}


// ! geomalgorithms.com/a07-_distance.html
__device__ float GetDistanceBetweenTwoLineSegments(const float3 &p1, const float3 &p2, const float3 &q1, const float3 &q2)
{
    float3 u = p2 - p1;
    Normalize(u);
    float3 v = q2 - q1;
    Normalize(v);
    float3 w = p1 - q1;

    float a = DotProduct(u, u);
    float b = DotProduct(u, v);
    float c = DotProduct(v, v);
    float d = DotProduct(u, w);
    float e = DotProduct(v, w);

    float sc = (b*e - c*d) / (a*c - b*b);

    if (sqrt(DotProduct(p2 - p1, p2 - p1)) > sc && sc > 0)
    {
        float3 n = CrossProduct(p2 - p1, q2 - q1);
        Normalize(n);
        return abs(DotProduct(p1 - q1, n));
    }

    float d1, d2, d3, d4;
    d1 = GetDistanceBetweenPointAndLineSegment(p1, q1, q2);
    d2 = GetDistanceBetweenPointAndLineSegment(p2, q1, q2);
    d3 = GetDistanceBetweenPointAndLineSegment(q1, p1, p2);
    d4 = GetDistanceBetweenPointAndLineSegment(q2, p1, p2);
    return dmin(dmin(dmin(d1, d2), d3), d4);
}

// Gets intersection of line with plane created by triangle
//p1, p2, p3 should be in clockwise order
__device__ float3 GetIntersectionOfLineWithTriangle(const float3 &lineOrigin,
    float3 &lineDir, const float3 &p1, const float3 &p2, const float3 &p3)
{
    //plane of triangle
    float3 n = CrossProduct((p2 - p1), (p3 - p1));
    Normalize(n);
    float d = DotProduct(n, p1); //coefficient "d" of plane equation (n.x = d)

    Normalize(lineDir);
    float t = (d-DotProduct(n,lineOrigin))/(DotProduct(n,lineDir));

    return lineOrigin + t*lineDir;
}

// Gets intersection of line segment with plane created by triangle
//p1, p2, p3 should be in clockwise order
__device__ bool GetIntersectionOfLineSegmentWithTriangle(float3 &intersect, const float3 &lineOrigin,
    float3 &lineDest, const float3 &p1, const float3 &p2, const float3 &p3)
{
    //plane of triangle
    float3 n = CrossProduct((p2 - p1), (p3 - p1));
    Normalize(n);
    float d = DotProduct(n, p1); //coefficient "d" of plane equation (n.x = d)

    float3 lineDir = lineDest - lineOrigin;
    const float length = sqrt(DotProduct(lineDir, lineDir));
    Normalize(lineDir);
    float t = (d-DotProduct(n,lineOrigin))/(DotProduct(n,lineDir));
    if (t > 0 && t < length)
    {
        intersect = lineOrigin + t*lineDir;
        return true;
    }
    return false;
}


// Only update intersect reference if intersect is inside the rectangle, and is closer to lineOrigin than previous value
__device__ bool IntersectLineSegmentWithRect(float3 &intersect, float3 lineOrigin, float3 lineDest, 
    float3 topLeft, float3 topRight, float3 bottomRight, float3 bottomLeft)
{
    float3 temp;
    if (GetIntersectionOfLineSegmentWithTriangle(temp, lineOrigin, lineDest, topLeft, topRight, bottomRight))
    {
        if (IsPointInsideTriangle(topLeft, topRight, bottomRight, temp))
        {
            if (Distance(temp, lineOrigin) < Distance(intersect, lineOrigin))
            {
                intersect = temp;
                return true;
            }
        }
    }
    if (GetIntersectionOfLineSegmentWithTriangle(temp, lineOrigin, lineDest, bottomRight, bottomLeft, topLeft))
    {
        if (IsPointInsideTriangle(bottomRight, bottomLeft, topLeft, temp))
        {
            if (Distance(temp, lineOrigin) < Distance(intersect, lineOrigin))
            {
                intersect = temp;
                return true;
            }
        }
    }
    return false;
}


__device__ float3 operator+(const float3 &u, const float3 &v)
{
    return make_float3(u.x + v.x, u.y + v.y, u.z + v.z);
}

__device__ float2 operator+(const float2 &u, const float2 &v)
{
    return make_float2(u.x + v.x, u.y + v.y);
}

__device__ float3 operator-(const float3 &u, const float3 &v)
{
    return make_float3(u.x - v.x, u.y - v.y, u.z - v.z);
}

__device__ float2 operator-(const float2 &u, const float2 &v)

{
    return make_float2(u.x - v.x, u.y - v.y);
}

__device__ float3 operator*(const float3 &u, const float3 &v)
{
    return make_float3(u.x * v.x, u.y * v.y, u.z * v.z);
}

__device__ float3 operator/(const float3 &u, const float3 &v)
{
    return make_float3(u.x / v.x, u.y / v.y, u.z / v.z);
}

__device__ float3 operator*(const float a, const float3 &u)
{
    return make_float3(a*u.x, a*u.y, a*u.z);
}

__device__ float3 operator/(const float3 &u, const float a)
{
    return make_float3(u.x / a, u.y / a, u.z / a);
}

