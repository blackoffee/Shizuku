#include "PauseRayTracing.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

PauseRayTracing::PauseRayTracing(Flow& p_flow) : Command(p_flow)
{
}

void PauseRayTracing::Start(boost::any const p_param)
{
    try
    {
        const bool& paused = boost::any_cast<bool>(p_param);
        m_flow->Graphics()->SetRayTracingPausedState(paused);
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}