#pragma once

#ifdef SHIZUKU_CORE_EXPORTS  
#define CORE_API __declspec(dllexport)   
#else  
#define CORE_API __declspec(dllimport)   
#endif  


namespace Shizuku{ namespace Core
{
    class StopwatchImpl;

    class CORE_API Stopwatch
    {

    private:
        StopwatchImpl* m_impl;
    public:
        Stopwatch();
        Stopwatch(const int p_recCount);

        void Tick();

        // Return time from Tick
        double Tock();

        void Reset();

        // Return running average 
        double GetAverage();
    };
}}
