#pragma once
#include "cuda_runtime.h"

class LbmNode
{
    float m_f[9];
    int m_xDim, m_yDim;
public:
    __host__ __device__ LbmNode();
    __device__ int GetXDim();
    __device__ int GetYDim();
    __device__ void SetXDim(const int xDim);
    __device__ void SetYDim(const int yDim);
    __device__ float ComputeRho();
    __device__ float ComputeU();
    __device__ float ComputeV();
    __device__ void ReadIncomingDistributions(float* f, const int x, const int y);
    __device__ void ReadDistributions(float* f, const int x, const int y);
    __device__ void Initialize(float* f, const float rho, const float u, const float v);
    __device__ void ComputeFeqs(float* fOut, const float rho, const float u,
        const float v);
    __device__ void ComputeFeqs(float* fOut);
    __device__ float ComputeStrainRateMagnitude();
    __device__ void DirichletWest(const int y, const int xDim, const int yDim,
        const float uMax);
    __device__ void NeumannEast(const int y, const int xDim, const int yDim);
    __device__ void MovingWall(const float rho, const float u, const float v);
    __device__ void BounceBackWall();
    __device__ void ApplyBCs(const int y, const int im, const int xDim, const int yDim,
        const float uMax);
    __device__ void Collide(const float omega);
    __device__ void WriteDistributions(float* f, const int x, const int y);
};

