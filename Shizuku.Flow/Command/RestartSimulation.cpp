#include "RestartSimulation.h"
#include "Graphics/GraphicsManager.h"
#include "Graphics/CudaLbm.h"
#include "Flow.h"
#include <boost/any.hpp>

using namespace Shizuku::Flow::Command;

RestartSimulation::RestartSimulation(Flow& p_flow) : Command(p_flow)
{
}

void RestartSimulation::Start(boost::any const p_param)
{
    try
    {
        m_flow->Graphics()->InitializeFlow();
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}
