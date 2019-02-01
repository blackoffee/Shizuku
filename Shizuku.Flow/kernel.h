#pragma once
#include "Graphics/GraphicsManager.h"
#include "Domain.h"
#include "common.h"
#include "cuda_runtime.h"
#include "cuda.h"

class CudaLbm;

void InitializeDomain(float4* vis, float* f_d, int* im_d, const float uMax,
    Domain &simDomain);

void SetObstructionVelocitiesToZero(ObstDefinition* obst_h, ObstDefinition* obst_d, Domain &simDomain);

void MarchSolution(CudaLbm* cudaLbm);

void UpdateSolutionVbo(float4* vis, float4* p_normals, CudaLbm* cudaLbm, 
    const ContourVariable contVar, const float contMin, const float contMax,
    const ViewMode viewMode, const float waterDepth);

void UpdateDeviceObstructions(ObstDefinition* obst_d, const int targetObstID,
    const ObstDefinition &newObst, Domain &simDomain);

void SurfacePhongLighting(float4* vis, float4* p_normals, ObstDefinition* obst_d, const float3 cameraPosition, 
    Domain &simDomain);

void InitializeSurface(float4* vis, Domain &simDomain);

void InitializeFloor(float4* vis, Domain &simDomain);

void LightFloor(float4* vis, float4* p_normals, float* floor_d, ObstDefinition* obst_d, const int p_obstCount,
    const float3 cameraPosition, Domain &simDomain, CudaLbm& p_lbm, const float waterDepth, const float obstHeight);

void RefractSurface(float4* vis, float4* p_normals, cudaArray* floorTexture, cudaArray* envTexture, ObstDefinition* obst_d, const glm::vec4 cameraPos,
    Domain &simDomain, const float waterDepth, const float obstHeight, const bool simplified);
