#pragma once

#include "Shizuku.Core/Rect.h"

#ifdef SHIZUKU_FLOW_EXPORTS  
#define FLOW_API __declspec(dllexport)   
#else  
#define FLOW_API __declspec(dllimport)   
#endif  

using namespace Shizuku::Core;

namespace Shizuku { namespace Flow{
    class Flow;
    struct FLOW_API Diagnostics
    {
    private:
        Flow* m_flow;
    public:
        Diagnostics();
        Diagnostics(Flow& p_flow);
        Rect<int>& SimulationDomain();
    };
} }
