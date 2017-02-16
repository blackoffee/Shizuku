#pragma once
#include <math.h>
#include "cuda_runtime.h"
#include "common.h"

class SimulationParameters
{
public:
    int m_xDim;
    int m_yDim;
    int m_xDimVisible;
    int m_yDimVisible;

    SimulationParameters();

    int(*GetXDim)(SimulationParameters* p_simulationParameters);
    int(*GetYDim)(SimulationParameters* p_simulationParameters);

//    int GetXDim();
//    int GetYDim();
    __host__ __device__ int GetXDim_d();
    __host__ __device__ int GetYDim_d();

    int(*GetXDimVisible)(SimulationParameters* p_simulationParameters);
    int(*GetYDimVisible)(SimulationParameters* p_simulationParameters);

//    int GetXDimVisible();
//    int GetYDimVisible();
    __host__ __device__ int GetXDimVisible_d();
    __host__ __device__ int GetYDimVisible_d();


    void(*SetXDim)(SimulationParameters* p_simulationParameters, const int xDim);
    void(*SetYDim)(SimulationParameters* p_simulationParameters, const int yDim);

//    void SetXDim(const int xDim);
//    void SetYDim(const int yDim);

    void(*SetXDimVisible)(SimulationParameters* p_simulationParameters, const int xDimVisible);
    void(*SetYDimVisible)(SimulationParameters* p_simulationParameters, const int yDimVisible);

//    void SetXDimVisible(const int xDimVisible);
//    void SetYDimVisible(const int yDimVisible);
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


