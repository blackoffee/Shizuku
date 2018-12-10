#pragma once

#ifdef SHIZUKU_FLOW_EXPORTS  
#define FLOW_API __declspec(dllexport)   
#else  
#define FLOW_API __declspec(dllimport)   
#endif  

namespace Shizuku{ namespace Flow{ namespace Command{
    struct FLOW_API DepthParameter
    {
        float Depth;
        DepthParameter();
        DepthParameter(const float p_height);
    };
} } }

