#pragma once
#include <math.h>
#include "common.h"

extern "C"
{
    struct SimulationParameters
    {
        int m_xDim;
        int m_yDim;
        int m_xDimVisible;
        int m_yDimVisible;

        int(*GetXDim)(SimulationParameters* p_simulationParameters);
        int(*GetYDim)(SimulationParameters* p_simulationParameters);

        int(*GetXDimVisible)(SimulationParameters* p_simulationParameters);
        int(*GetYDimVisible)(SimulationParameters* p_simulationParameters);

        void(*SetXDim)(SimulationParameters* p_simulationParameters, const int xDim);
        void(*SetYDim)(SimulationParameters* p_simulationParameters, const int yDim);

        void(*SetXDimVisible)(SimulationParameters* p_simulationParameters, const int xDimVisible);
        void(*SetYDimVisible)(SimulationParameters* p_simulationParameters, const int yDimVisible);
    };


    void SimulationParameters_init(SimulationParameters* p_this);
    int SimulationParameters_GetXDim(SimulationParameters* p_this);
    int SimulationParameters_GetYDim(SimulationParameters* p_this);
    int SimulationParameters_GetXDimVisible(SimulationParameters* p_this);
    int SimulationParameters_GetYDimVisible(SimulationParameters* p_this);
    void SimulationParameters_SetXDim(SimulationParameters* p_this, const int xDim);
    void SimulationParameters_SetYDim(SimulationParameters* p_this, const int yDim);
    void SimulationParameters_SetXDimVisible(SimulationParameters* p_this, const int xDimVisible);
    void SimulationParameters_SetYDimVisible(SimulationParameters* p_this, const int yDimVisible);

}
