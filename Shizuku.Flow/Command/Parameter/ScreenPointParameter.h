#pragma once
#include "Shizuku.Core/Types/Point.h"

#ifdef SHIZUKU_FLOW_EXPORTS  
#define FLOW_API __declspec(dllexport)   
#else  
#define FLOW_API __declspec(dllimport)   
#endif  

namespace Shizuku{ namespace Flow{ namespace Command{
    struct FLOW_API ScreenPointParameter
    {
        Shizuku::Core::Types::Point<int> Position;
        ScreenPointParameter();
        ScreenPointParameter(const Shizuku::Core::Types::Point<int>& p_pos);
    };
} } }

