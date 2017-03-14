#pragma once
#include <time.h>

#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class FpsTracker
{
private:
    clock_t m_before, m_diff;
    int m_frameCount;
    int m_frameLimit;
    float m_fps;
public:
    FW_API FpsTracker();
    FW_API void Tick();
    FW_API void Tock();
    FW_API float GetFps();
};

