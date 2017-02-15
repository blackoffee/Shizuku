#define SMAG_CONST 1.f

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstring>
#include "common.h"
#include "SimulationParameters.h"

void InitializeDomain(float4* vis, float* f_d, int* im_d, const float uMax,
    SimulationParameters* simParams);

void SetObstructionVelocitiesToZero(Obstruction* obst_h, Obstruction* obst_d);

void MarchSolution(float4* vis, float* fA_d, float* fB_d, int* im_d, Obstruction* obst_d, 
    const ContourVariable contVar, const float contMin, const float contMax,
    const ViewMode viewMode, const float uMax, const float omega, const int tStep,
    SimulationParameters* simParams, const bool paused);

void UpdateDeviceObstructions(Obstruction* obst_d, const int targetObstID,
    const Obstruction &newObst);

void CleanUpDeviceVBO(float4* vis, SimulationParameters* simParams);

void LightSurface(float4* vis, Obstruction* obst_d, const float3 cameraPosition, 
    SimulationParameters* simParams);

void InitializeFloor(float4* vis, float* floor_d, SimulationParameters* simParams);

void LightFloor(float4* vis, float* floor_d, Obstruction* obst_d,
    const float3 cameraPosition, SimulationParameters* simParams);

int RayCastMouseClick(float3 &selectedElementCoord, float4* vis, float4* rayCastIntersect_d, float3 rayOrigin,
    float3 rayDir, Obstruction* obst_d, SimulationParameters* simParams);
