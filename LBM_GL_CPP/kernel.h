#define SMAG_CONST 1.f

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstring>
#include "common.h"

//extern "C"
int runCUDA();

//extern "C"
void InitializeDomain(float4* vis, float* f_d, int* im_d, int xDim, int yDim, float uMax, int xDimVisible, int yDimVisible);

//extern "C"
void MarchSolution(float4* vis, float* fA_d, float* fB_d, int* im_d, Obstruction* obst_d, 
						ContourVariable contVar, float contMin, float contMax, int xDim, int yDim, float uMax, float omega, int tStep, int xDimVisible, int yDimVisible);

//extern "C"
void UpdateDeviceObstructions(Obstruction* obst_d, int targetObstID, Obstruction newObst);

