#pragma once
#include <time.h>

#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class FW_API FpsTracker
{
private:
    clock_t m_before, m_diff;
    int m_frameCount;
    int m_frameLimit;
    float m_fps;
public:
    FpsTracker();
    void Tick();
    void Tock();
    float GetFps();
};

