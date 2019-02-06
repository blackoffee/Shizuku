#include "SetLightProbeVisibility.h"
#include "Graphics/GraphicsManager.h"
#include "Parameter/VisibilityParameter.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

SetLightProbeVisibility::SetLightProbeVisibility(Flow& p_flow) : Command(p_flow)
{
}

void SetLightProbeVisibility::Start(boost::any const p_param)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    try
    {
        const VisibilityParameter& vis = boost::any_cast<VisibilityParameter>(p_param);
        graphicsManager->EnableLightProbe(vis.Visible);
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}

