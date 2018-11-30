#include "SetSimulationScale.h"
#include "Graphics/GraphicsManager.h"
#include "Parameter/ScaleParameter.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

SetSimulationScale::SetSimulationScale(Flow& p_flow) : Command(p_flow)
{
}

void SetSimulationScale::Start(boost::any const p_param)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    try
    {
        const ScaleParameter& scale = boost::any_cast<ScaleParameter>(p_param);
        graphicsManager->SetScaleFactor(scale.Scale);
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}

