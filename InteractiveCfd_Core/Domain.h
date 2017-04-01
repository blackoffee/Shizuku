#pragma once
#include "cuda_runtime.h"

#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class FW_API Domain
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

__device__ int dmin(const int a, const int b);
__device__ int dmax(const int a);
__device__ int dmax(const int a, const int b);
__device__ float dmin(const float a, const float b);
__device__ float dmin(const float a, const float b, const float c, const float d);
__device__ float dmax(const float a);
__device__ float dmax(const float a, const float b);
__device__ float dmax(const float a, const float b, const float c, const float d);
__device__ int f_mem(const int f_num, const int x, const int y, const size_t pitch,
    const int yDim);
__device__ int f_mem(const int f_num, const int x, const int y);
__device__ void Swap(float &a, float &b);

