#include "SimulationParameters.h"

void SimulationParameters_init(SimulationParameters* p_this)
{
    p_this->m_xDim = BLOCKSIZEX * 2;
    p_this->m_yDim = BLOCKSIZEX;
    p_this->m_xDimVisible = p_this->m_xDim;
    p_this->m_yDimVisible = p_this->m_yDim;
    p_this->GetXDim = SimulationParameters_GetXDim;
    p_this->GetYDim = SimulationParameters_GetYDim;
    p_this->GetXDimVisible = SimulationParameters_GetXDimVisible;
    p_this->GetYDimVisible = SimulationParameters_GetYDimVisible;
    p_this->SetXDim = SimulationParameters_SetXDim;
    p_this->SetYDim = SimulationParameters_SetYDim;
    p_this->SetXDimVisible = SimulationParameters_SetXDimVisible;
    p_this->SetYDimVisible = SimulationParameters_SetYDimVisible;
}

int SimulationParameters_GetXDim(SimulationParameters* p_this)
{
    return p_this->m_xDim;
}

int SimulationParameters_GetYDim(SimulationParameters* p_this)
{
    return p_this->m_yDim;
}

int SimulationParameters_GetXDimVisible(SimulationParameters* p_this)
{
    return p_this->m_xDimVisible;
}

int SimulationParameters_GetYDimVisible(SimulationParameters* p_this)
{
    return p_this->m_yDimVisible;
}

void SimulationParameters_SetXDim(SimulationParameters* p_this, const int xDim)
{
    //x dimension must be multiple of BLOCKSIZEX
    int xDimAsMultipleOfBlocksize = ceil(static_cast<float>(xDim)/BLOCKSIZEX)*BLOCKSIZEX;
    p_this->m_xDim = xDimAsMultipleOfBlocksize;
}

void SimulationParameters_SetYDim(SimulationParameters* p_this, const int yDim)
{
    //y dimension must be multiple of BLOCKSIZEY
    int yDimAsMultipleOfBlocksize = ceil(static_cast<float>(yDim)/BLOCKSIZEY)*BLOCKSIZEY;
    p_this->m_yDim = yDimAsMultipleOfBlocksize;
}

void SimulationParameters_SetXDimVisible(SimulationParameters* p_this, const int xDimVisible)
{
    p_this->m_xDimVisible = xDimVisible;
    p_this->SetXDim(p_this,xDimVisible);
}

void SimulationParameters_SetYDimVisible(SimulationParameters* p_this, const int yDimVisible)
{
    p_this->m_yDimVisible = yDimVisible;
    p_this->SetYDim(p_this,yDimVisible);
}




//
//struct DevicePointers
//{

//};