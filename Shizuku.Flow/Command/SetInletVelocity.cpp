#include "SetInletVelocity.h"
#include "Graphics/GraphicsManager.h"
#include "Parameter/Parameter.h"

using namespace Shizuku::Flow;

SetInletVelocity::SetInletVelocity(GraphicsManager &p_graphicsManager) : Command(p_graphicsManager)
{
}

void SetInletVelocity::Start(boost::any const p_param)
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    try
    {
        const VelocityParameter& vel = boost::any_cast<VelocityParameter>(p_param);
        graphicsManager->SetVelocity(vel.velocity);
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}

