#pragma once
#include "../common.h"
#include "Obstruction.h"
#include "Shizuku.Core/Rect.h"

using namespace Shizuku::Flow;

class Domain;

namespace Shizuku {
	namespace Flow {
		class ObstManager;
	}
}

class CudaLbm
{
private:
    int m_maxX;
    int m_maxY;
    Domain* m_domain;
    float* m_fA_d;
    float* m_fB_d;
    int* m_Im_d;
    int* m_Im_h;
    float* m_FloorTemp_d;
    ObstDefinition* m_obst_d;
    ObstDefinition m_obst_h[MAXOBSTS];
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
    ObstDefinition* GetDeviceObst();
    ObstDefinition* GetHostObst();
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
    void InitializeDeviceImage();
    void UpdateDeviceImage(ObstManager& p_obstMgr);
    int ImageFcn(const int x, const int y);

   
};

