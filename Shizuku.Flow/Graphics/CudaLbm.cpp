#include "CudaLbm.h"
#include "Domain.h"
#include "ObstManager.h"

#include "Shizuku.Core/Types/Point.h"

#include <algorithm>

using namespace Shizuku::Core::Types;

namespace {
    Point<float> ModelSpacePosFromSimPos(const Point<int>& p_simPos, const int p_xDimVisible)
    {
        return Point<float>(static_cast<float>(p_simPos.X) / p_xDimVisible*2.f - 1.f, static_cast<float>(p_simPos.Y) / p_xDimVisible*2.f - 1.f);
    }
}

CudaLbm::CudaLbm()
{
    m_domain = new Domain;
    m_isPaused = false;
    m_timeStepsPerFrame = 15;
}

CudaLbm::CudaLbm(const int maxX, const int maxY)
{
    m_maxX = maxX;
    m_maxY = maxY;
}

Domain* CudaLbm::GetDomain()
{
    return m_domain;
}

Shizuku::Core::Rect<int> CudaLbm::GetDomainSize()
{
    return Shizuku::Core::Rect<int>(m_domain->GetXDimVisible(), m_domain->GetYDimVisible());
}

float* CudaLbm::GetFA()
{
    return m_fA_d;
}

float* CudaLbm::GetFB()
{
    return m_fB_d;
}

int* CudaLbm::GetImage()
{
    return m_Im_d;
}

float* CudaLbm::GetFloorTemp()
{
    return m_FloorTemp_d;
}

ObstDefinition* CudaLbm::GetDeviceObst()
{
    return m_obst_d;
}

ObstDefinition* CudaLbm::GetHostObst()
{
    return &m_obst_h[0];
}

float CudaLbm::GetInletVelocity()
{
    return m_inletVelocity;
}

float CudaLbm::GetOmega()
{
    return m_omega;
}

void CudaLbm::SetInletVelocity(const float velocity)
{
    m_inletVelocity = velocity;
}

void CudaLbm::SetOmega(const float omega)
{
    m_omega = omega;
}

void CudaLbm::TogglePausedState()
{
    m_isPaused = !m_isPaused;
}

void CudaLbm::SetPausedState(const bool isPaused)
{
    m_isPaused = isPaused;
}

bool CudaLbm::IsPaused()
{
    return m_isPaused;
}

int CudaLbm::GetTimeStepsPerFrame()
{
    return m_timeStepsPerFrame;
}

void CudaLbm::SetTimeStepsPerFrame(const int timeSteps)
{
    m_timeStepsPerFrame = timeSteps;
}



void CudaLbm::AllocateDeviceMemory()
{
    size_t memsize_lbm, memsize_int, memsize_float, memsize_inputs;

    int domainSize = ceil(MAX_XDIM / BLOCKSIZEX)*BLOCKSIZEX*ceil(MAX_YDIM / BLOCKSIZEY)*BLOCKSIZEY;
    memsize_lbm = domainSize*sizeof(float)*9;
    memsize_int = domainSize*sizeof(int);
    memsize_float = domainSize*sizeof(float);
    memsize_inputs = sizeof(m_obst_h);

    float* fA_h = new float[domainSize * 9];
    float* fB_h = new float[domainSize * 9];
    float* floor_h = new float[domainSize];
    m_Im_h = new int[domainSize];

    cudaMalloc((void **)&m_fA_d, memsize_lbm);
    cudaMalloc((void **)&m_fB_d, memsize_lbm);
    cudaMalloc((void **)&m_FloorTemp_d, memsize_float);
    cudaMalloc((void **)&m_Im_d, memsize_int);
    cudaMalloc((void **)&m_obst_d, memsize_inputs);
}

void CudaLbm::DeallocateDeviceMemory()
{
    cudaFree(m_fA_d);
    cudaFree(m_fB_d);
    cudaFree(m_Im_d);
    cudaFree(m_FloorTemp_d);
    cudaFree(m_obst_d);

    //TODO - separate method for host memory
    delete[] m_Im_h;
}

void CudaLbm::InitializeDeviceMemory()
{
    int domainSize = ceil(MAX_XDIM / BLOCKSIZEX)*BLOCKSIZEX*ceil(MAX_YDIM / BLOCKSIZEY)*BLOCKSIZEY;
    size_t memsize_lbm, memsize_float, memsize_inputs;
    memsize_lbm = domainSize*sizeof(float)*9;
    memsize_float = domainSize*sizeof(float);

    float* f_h = new float[domainSize*9];
    for (int i = 0; i < domainSize * 9; i++)
    {
        f_h[i] = 0;
    }
    cudaMemcpy(m_fA_d, f_h, memsize_lbm, cudaMemcpyHostToDevice);
    cudaMemcpy(m_fB_d, f_h, memsize_lbm, cudaMemcpyHostToDevice);
    delete[] f_h;
    float* floor_h = new float[domainSize];
    for (int i = 0; i < domainSize; i++)
    {
        floor_h[i] = 0;
    }
    cudaMemcpy(m_FloorTemp_d, floor_h, memsize_float, cudaMemcpyHostToDevice);
    delete[] floor_h;

    //UpdateDeviceImage();

    for (int i = 0; i < MAXOBSTS; i++)
    {
        m_obst_h[i].r1 = 0;
        m_obst_h[i].x = 0;
        m_obst_h[i].y = -1000;
        m_obst_h[i].state = State::SELECTED;
    }	

    memsize_inputs = sizeof(m_obst_h);
    cudaMemcpy(m_obst_d, m_obst_h, memsize_inputs, cudaMemcpyHostToDevice);
}

void CudaLbm::InitializeDeviceImage()
{
    int domainSize = ceil(MAX_XDIM / BLOCKSIZEX)*BLOCKSIZEX*ceil(MAX_YDIM / BLOCKSIZEY)*BLOCKSIZEY;
    for (int i = 0; i < domainSize; i++)
    {
        int x = i%MAX_XDIM;
        int y = i/MAX_XDIM;
        m_Im_h[i] = ImageFcn(x, y);
    }
    size_t memsize_int = domainSize*sizeof(int);
    cudaMemcpy(m_Im_d, m_Im_h, memsize_int, cudaMemcpyHostToDevice);
}

void CudaLbm::UpdateDeviceImage(ObstManager& p_obstMgr)
{
    int domainSize = ceil(MAX_XDIM / BLOCKSIZEX)*BLOCKSIZEX*ceil(MAX_YDIM / BLOCKSIZEY)*BLOCKSIZEY;
    for (int i = 0; i < domainSize; i++)
    {
        int x = i%MAX_XDIM;
        int y = i/MAX_XDIM;
        m_Im_h[i] = ImageFcn(x, y);
        const int xDimVisible = GetDomain()->GetXDimVisible();
        const Point<float> modelCoord = ModelSpacePosFromSimPos(Point<int>(x, y), xDimVisible);
		if (p_obstMgr.IsInsideObstruction(modelCoord))
            m_Im_h[i] = 1;
    }
    size_t memsize_int = domainSize*sizeof(int);
    cudaMemcpy(m_Im_d, m_Im_h, memsize_int, cudaMemcpyHostToDevice);
}

int CudaLbm::ImageFcn(const int x, const int y){
    int xDim = GetDomain()->GetXDim();
    int yDim = GetDomain()->GetYDim();
    if (x < 0.1f)
        return 3;//west
    else if ((xDim - x) < 1.1f)
        return 2;//east
    else if ((yDim - y) < 1.1f)
        return 11;//11;//xsymmetry top
    else if (y < 0.1f)
        return 12;//12;//xsymmetry bottom
    return 0;
}

