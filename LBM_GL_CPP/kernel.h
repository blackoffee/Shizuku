#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstring>
#include "common.h"

extern "C"
int runCUDA();

extern "C"
void InitializeDomain(float* f_d, int* im_d, int xDim, int yDim);

extern "C"
void MarchSolution(float4* vis, float* fA_d, float* fB_d, int* im_d, Obstruction* obst_d, 
						ContourVariable contVar, int xDim, int yDim, int tStep);

extern "C"
void UpdateObstructions(Obstruction* obst_d, int targetObstID);
