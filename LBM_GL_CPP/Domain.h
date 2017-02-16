#pragma once
#include <math.h>
#include "cuda_runtime.h"
#include "common.h"

class Domain
{
    int m_xDim;
    int m_yDim;
    int m_xDimVisible;
    int m_yDimVisible;

public:
    Domain();

    __host__ __device__ int GetXDim();
    __host__ __device__ int GetYDim();
    __host__ void SetXDim(const int xDim);
    __host__ void SetYDim(const int yDim);

    __host__ __device__ int GetXDimVisible();
    __host__ __device__ int GetYDimVisible();
    __host__ void SetXDimVisible(const int xDimVisible);
    __host__ void SetYDimVisible(const int yDimVisible);

};

