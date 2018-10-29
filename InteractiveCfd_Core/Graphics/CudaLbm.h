#pragma once
#include "common.h"
#include "Shizuku.Core/Rect.h"

#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class Domain;

class FW_API CudaLbm
{
private:
    int m_maxX;
    int m_maxY;
    Domain* m_domain;
    float* m_fA_d;
    float* m_fB_d;
    int* m_Im_d;
    float* m_FloorTemp_d;
    Obstruction* m_obst_d;
    Obstruction m_obst_h[MAXOBSTS];
    float m_inletVelocity;
    float m_omega;
    bool m_isPaused;
    int m_timeStepsPerFrame;
public:
    CudaLbm();
    CudaLbm(const int maxX, const int maxY);
    Domain* GetDomain();
    Shizuku::Core::Rect<int> GetDomainSize();
    float* GetFA();
    float* GetFB();
    int* GetImage();
    float* GetFloorTemp();
    Obstruction* GetDeviceObst();
    Obstruction* GetHostObst();
    float GetInletVelocity();
    float GetOmega();
    void SetInletVelocity(const float velocity);
    void SetOmega(const float omega);
    void TogglePausedState();
    void SetPausedState(const bool isPaused);
    bool IsPaused();
    int GetTimeStepsPerFrame();
    void SetTimeStepsPerFrame(const int timeSteps);

    void AllocateDeviceMemory();
    void InitializeDeviceMemory();
    void DeallocateDeviceMemory();
    void UpdateDeviceImage();
    int ImageFcn(const int x, const int y);

   
};

