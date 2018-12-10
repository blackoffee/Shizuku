#pragma once

#ifdef SHIZUKU_FLOW_EXPORTS  
#define FLOW_API __declspec(dllexport)   
#else  
#define FLOW_API __declspec(dllimport)   
#endif  

namespace Shizuku { namespace Flow{
    enum FLOW_API TimerKey
    {
        SolveFluid,
        PrepareSurface,
        PrepareFloor,
        ProcessSurface,
        ProcessFloor
    };
} }