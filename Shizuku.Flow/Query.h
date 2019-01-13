#pragma once

#include "Shizuku.Core/Rect.h"
#include "Shizuku.Core/Types/Point.h"

#ifdef SHIZUKU_FLOW_EXPORTS  
#define FLOW_API __declspec(dllexport)   
#else  
#define FLOW_API __declspec(dllimport)   
#endif  

using namespace Shizuku::Core;

namespace Shizuku { namespace Flow{
    class Flow;
    enum TimerKey;
    struct FLOW_API Query
    {
    private:
        Flow* m_flow;
    public:
        Query();
        Query(Flow& p_flow);
        Rect<int> SimulationDomain();
        double GetTime(TimerKey p_key);
        Types::Point<float> ProbeModelSpaceCoord(const Types::Point<int>& p_screenPoint);
		int ObstructionCount();
		int SelectedObstructionCount();
		int PreSelectedObstructionCount();
    };
} }
