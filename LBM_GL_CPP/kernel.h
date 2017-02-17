#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstring>
#include "common.h"
#include "Domain.h"
#include "LbmNode.h"

void InitializeDomain(float4* vis, float* f_d, int* im_d, const float uMax,
    Domain &simDomain);

void SetObstructionVelocitiesToZero(Obstruction* obst_h, Obstruction* obst_d);

void MarchSolution(float* fA_d, float* fB_d, int* im_d, Obstruction* obst_d,
    const float uMax, const float omega, const int tStep, Domain &simDomain,
    const bool paused);

void UpdateSolutionVbo(float4* vis, float* f_d, int* im_d, 
    const ContourVariable contVar, const float contMin, const float contMax,
    const ViewMode viewMode, const float uMax, Domain &simDomain);

void UpdateDeviceObstructions(Obstruction* obst_d, const int targetObstID,
    const Obstruction &newObst);

void CleanUpDeviceVBO(float4* vis, Domain &simDomain);

void LightSurface(float4* vis, Obstruction* obst_d, const float3 cameraPosition, 
    Domain &simDomain);

void InitializeFloor(float4* vis, float* floor_d, Domain &simDomain);

void LightFloor(float4* vis, float* floor_d, Obstruction* obst_d,
    const float3 cameraPosition, Domain &simDomain);

int RayCastMouseClick(float3 &selectedElementCoord, float4* vis,
    float4* rayCastIntersect_d, float3 rayOrigin, float3 rayDir,
    Obstruction* obst_d, Domain &simDomain);
