#include "SetWaterDepth.h"
#include "Graphics/GraphicsManager.h"
#include "Parameter/DepthParameter.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

SetWaterDepth::SetWaterDepth(Flow& p_flow) : Command(p_flow)
{
}

void SetWaterDepth::Start(boost::any const p_param)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    try
    {
        const DepthParameter& heights = boost::any_cast<DepthParameter>(p_param);
        graphicsManager->SetWaterDepth(heights.Depth);
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}
