#include "Domain.h"
#include "common.h"
#include <math.h>

Domain::Domain()
{
    m_xDim = BLOCKSIZEX * 2;
    m_yDim = BLOCKSIZEX;
    m_xDimVisible = m_xDim;
    m_yDimVisible = m_yDim;
}

__host__ __device__ int Domain::GetXDim()
{
    return m_xDim;
}

__host__ __device__ int Domain::GetYDim()
{
    return m_yDim;
}

__host__ __device__ int Domain::GetXDimVisible()
{
    return m_xDimVisible;
}

__host__ __device__ int Domain::GetYDimVisible()
{
    return m_yDimVisible;
}

__host__ void Domain::SetXDim(const int xDim)
{
    //x dimension must be multiple of BLOCKSIZEX
    int xDimAsMultipleOfBlocksize = ceil(static_cast<float>(xDim)/BLOCKSIZEX)*BLOCKSIZEX;
    m_xDim = xDimAsMultipleOfBlocksize < MAX_XDIM ? xDimAsMultipleOfBlocksize : MAX_XDIM;
}

__host__ void Domain::SetYDim(const int yDim)
{
    //y dimension must be multiple of BLOCKSIZEY
    int yDimAsMultipleOfBlocksize = ceil(static_cast<float>(yDim)/BLOCKSIZEY)*BLOCKSIZEY;
    m_yDim = yDimAsMultipleOfBlocksize < MAX_YDIM ? yDimAsMultipleOfBlocksize : MAX_YDIM;
}

__host__ void Domain::SetXDimVisible(const int xDimVisible)
{
    m_xDimVisible = xDimVisible < MAX_XDIM ? xDimVisible : MAX_XDIM;
    SetXDim(xDimVisible);
}

__host__ void Domain::SetYDimVisible(const int yDimVisible)
{
    m_yDimVisible = yDimVisible < MAX_YDIM ? yDimVisible : MAX_YDIM;
    SetYDim(yDimVisible);
}

__device__ int dmin(const int a, const int b)
{
    if (a<b) return a;
    else return b - 1;
}
__device__ int dmax(const int a)
{
    if (a>-1) return a;
    else return 0;
}
__device__ int dmax(const int a, const int b)
{
    if (a>b) return a;
    else return b;
}
__device__ float dmin(const float a, const float b)
{
    if (a<b) return a;
    else return b;
}
__device__ float dmin(const float a, const float b, const float c, const float d)
{
    return dmin(dmin(a, b), dmin(c, d));
}
__device__ float dmax(const float a)
{
    if (a>0) return a;
    else return 0;
}
__device__ float dmax(const float a, const float b)
{
    if (a>b) return a;
    else return b;
}
__device__ float dmax(const float a, const float b, const float c, const float d)
{
    return dmax(dmax(a, b), dmax(c, d));
}

__device__ int f_mem(const int f_num, const int x, const int y,
    const size_t pitch, const int yDim)
{
    return (x + y*pitch) + f_num*pitch*yDim;
}

__device__ int f_mem(const int f_num, const int x, const int y)
{

    return (x + y*MAX_XDIM) + f_num*MAX_XDIM*MAX_YDIM;
}

__device__ void Swap(float &a, float &b)
{
    float c = a;
    a = b;
    b = c;
}

