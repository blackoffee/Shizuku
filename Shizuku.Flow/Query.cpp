#pragma once

#include "Query.h"
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

Query::Query()
{
}

Query::Query(Flow& p_flow)
{
    m_flow = &p_flow;
}

Rect<int> Query::SimulationDomain()
{
    return m_flow->Graphics()->GetCudaLbm()->GetDomainSize();
}

double Query::GetTime(TimerKey p_key)
{
    std::map<TimerKey, Stopwatch> timers = m_flow->Graphics()->GetTimers();
    return timers[p_key].GetAverage();
}

Types::Point<float> Query::ProbeModelSpaceCoord(const Types::Point<int>& p_screenPoint)
{
    return m_flow->Graphics()->GetModelSpaceCoordFromScreenPos(p_screenPoint);
}

int Query::ObstructionCount()
{
	return m_flow->Graphics()->ObstCount();
}

int Query::SelectedObstructionCount()
{
	return m_flow->Graphics()->SelectedObstCount();
}

int Query::PreSelectedObstructionCount()
{
	return m_flow->Graphics()->PreSelectedObstCount();
}
