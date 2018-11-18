#pragma once
#include <time.h>

#ifdef SHIZUKU_CORE_EXPORTS  
#define CORE_API __declspec(dllexport)   
#else  
#define CORE_API __declspec(dllimport)   
#endif  

namespace Shizuku{ namespace Core
{
    class CORE_API FpsTracker
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
}}
