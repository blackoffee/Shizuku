#pragma once

#include "Diagnostics.h"
#include "Flow.h"
#include "TimerKey.h"
#include "Graphics/GraphicsManager.h"
#include "Graphics/CudaLbm.h"
#include "Shizuku.Core/Utilities/Stopwatch.h"

#ifdef SHIZUKU_FLOW_EXPORTS  
#define FLOW_API __declspec(dllexport)   
#else  
#define FLOW_API __declspec(dllimport)   
#endif  

using namespace Shizuku::Core;
using namespace Shizuku::Flow;

Diagnostics::Diagnostics()
{
}

Diagnostics::Diagnostics(Flow& p_flow)
{
    m_flow = &p_flow;
}

Rect<int> Diagnostics::SimulationDomain()
{
    return m_flow->Graphics()->GetCudaLbm()->GetDomainSize();
}

double Diagnostics::GetTime(TimerKey p_key)
{
    std::map<TimerKey, Stopwatch> timers = m_flow->Graphics()->GetTimers();
    return timers[p_key].GetAverage();
}