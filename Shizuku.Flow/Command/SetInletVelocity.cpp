#include "SetInletVelocity.h"
#include "Graphics/GraphicsManager.h"
#include "Parameter/VelocityParameter.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

SetInletVelocity::SetInletVelocity(Flow& p_flow) : Command(p_flow)
{
}

void SetInletVelocity::Start(boost::any const p_param)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    try
    {
        const VelocityParameter& vel = boost::any_cast<VelocityParameter>(p_param);
        graphicsManager->SetVelocity(vel.Velocity);
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}

