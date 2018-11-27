#pragma once
#include "Shizuku.Core/Types/MinMax.h"

#ifdef SHIZUKU_FLOW_EXPORTS  
#define FLOW_API __declspec(dllexport)   
#else  
#define FLOW_API __declspec(dllimport)   
#endif  

namespace Shizuku{ namespace Flow{ namespace Command{
    struct FLOW_API MinMaxParameter
    {
        Shizuku::Core::Types::MinMax<float> MinMax;

        MinMaxParameter()
        {}

        MinMaxParameter(const Shizuku::Core::Types::MinMax<float>& p_minMax) : MinMax(p_minMax)
        {}
    };
} } }

