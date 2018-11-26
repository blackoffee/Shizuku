#include "PauseSimulation.h"
#include "Graphics/GraphicsManager.h"
#include "Graphics/CudaLbm.h"
#include "Flow.h"
#include <boost/any.hpp>

using namespace Shizuku::Flow::Command;

PauseSimulation::PauseSimulation(Flow& p_flow) : Command(p_flow)
{
}

void PauseSimulation::Start(boost::any const p_param)
{
    try
    {
        const bool& paused = boost::any_cast<bool>(p_param);
        m_flow->Graphics()->GetCudaLbm()->SetPausedState(paused);
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}
