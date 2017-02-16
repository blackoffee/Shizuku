#include "SimulationParameters.h"

SimulationParameters::SimulationParameters()
{
    m_xDim = BLOCKSIZEX * 2;
    m_yDim = BLOCKSIZEX;
    m_xDimVisible = m_xDim;
    m_yDimVisible = m_yDim;
}

__host__ __device__ int SimulationParameters::GetXDim()
{
    return m_xDim;
}

__host__ __device__ int SimulationParameters::GetYDim()
{
    return m_yDim;
}

__host__ __device__ int SimulationParameters::GetXDimVisible()
{
    return m_xDimVisible;
}

__host__ __device__ int SimulationParameters::GetYDimVisible()
{
    return m_yDimVisible;
}

__host__ void SimulationParameters::SetXDim(const int xDim)
{
    //x dimension must be multiple of BLOCKSIZEX
    int xDimAsMultipleOfBlocksize = ceil(static_cast<float>(xDim)/BLOCKSIZEX)*BLOCKSIZEX;
    m_xDim = xDimAsMultipleOfBlocksize < MAX_XDIM ? xDimAsMultipleOfBlocksize : MAX_XDIM;
}

__host__ void SimulationParameters::SetYDim(const int yDim)
{
    //y dimension must be multiple of BLOCKSIZEY
    int yDimAsMultipleOfBlocksize = ceil(static_cast<float>(yDim)/BLOCKSIZEY)*BLOCKSIZEY;
    m_yDim = yDimAsMultipleOfBlocksize < MAX_YDIM ? yDimAsMultipleOfBlocksize : MAX_YDIM;
}

__host__ void SimulationParameters::SetXDimVisible(const int xDimVisible)
{
    m_xDimVisible = xDimVisible < MAX_XDIM ? xDimVisible : MAX_XDIM;
    SetXDim(xDimVisible);
}

__host__ void SimulationParameters::SetYDimVisible(const int yDimVisible)
{
    m_yDimVisible = yDimVisible < MAX_YDIM ? yDimVisible : MAX_YDIM;
    SetYDim(yDimVisible);
}
