#pragma once
#include "Graphics/GraphicsManager.h"
#include "Domain.h"
#include "common.h"
#include "cuda_runtime.h"
#include "cuda.h"

class CudaLbm;

void InitializeDomain(float4* vis, float* f_d, int* im_d, const float uMax,
    Domain &simDomain);

void SetObstructionVelocitiesToZero(Obstruction* obst_h, Obstruction* obst_d, const float scaleFactor);

void MarchSolution(CudaLbm* cudaLbm);

void UpdateSolutionVbo(float4* vis, CudaLbm* cudaLbm, 
    const ContourVariable contVar, const float contMin, const float contMax,
    const ViewMode viewMode);

void UpdateDeviceObstructions(Obstruction* obst_d, const int targetObstID,
    const Obstruction &newObst, const float scaleFactor);

void CleanUpDeviceVBO(float4* vis, Domain &simDomain);

void LightSurface(float4* vis, Obstruction* obst_d, const float3 cameraPosition, 
    Domain &simDomain);

void InitializeFloor(float4* vis, float* floor_d, Domain &simDomain);

void LightFloor(float4* vis, float* floor_d, Obstruction* obst_d,
    const float3 cameraPosition, Domain &simDomain);

int RayCastMouseClick(float3 &selectedElementCoord, float4* vis,
    float4* rayCastIntersect_d, const float3 &rayOrigin, const float3 &rayDir,
    Obstruction* obst_d, Domain &simDomain);

void RefractSurface(float4* vis, cudaArray* floorTexture, cudaArray* envTexture, Obstruction* obst_d, const glm::vec4 cameraPos,
    Domain &simDomain);
