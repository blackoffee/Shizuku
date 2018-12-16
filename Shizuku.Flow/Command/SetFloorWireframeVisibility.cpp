#include "SetFloorWireframeVisibility.h"
#include "Graphics/GraphicsManager.h"
#include "Parameter/VisibilityParameter.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

SetFloorWireframeVisibility::SetFloorWireframeVisibility(Flow& p_flow) : Command(p_flow)
{
}

void SetFloorWireframeVisibility::Start(boost::any const p_param)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    try
    {
        const VisibilityParameter& vis = boost::any_cast<VisibilityParameter>(p_param);
        graphicsManager->SetFloorWireframeVisibility(vis.Visible);
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}

