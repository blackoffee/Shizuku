#pragma once
#include <time.h>
#include <queue>
#include <chrono>

#ifdef SHIZUKU_CORE_EXPORTS  
#define CORE_API __declspec(dllexport)   
#else  
#define CORE_API __declspec(dllimport)   
#endif  

namespace Shizuku{ namespace Core
{
    class StopwatchImpl
    {
    private:
        std::chrono::high_resolution_clock::time_point m_before;
        double m_total;
        unsigned int m_recCount;
        std::queue<double> m_recs;
    public:
        StopwatchImpl();
        StopwatchImpl(const int p_recCount);

        void Tick();

        // Return time from Tick
        double Tock();

        void Reset();

        // Return running average 
        double GetAverage();
    };
}}
