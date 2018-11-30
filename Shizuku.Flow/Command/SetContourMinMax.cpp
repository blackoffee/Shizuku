#include "SetContourMinMax.h"
#include "Graphics/GraphicsManager.h"
#include "Parameter/MinMaxParameter.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

SetContourMinMax::SetContourMinMax(Flow& p_flow) : Command(p_flow)
{
}

void SetContourMinMax::Start(boost::any const p_param)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    try
    {
        const MinMaxParameter& minMax = boost::any_cast<MinMaxParameter>(p_param);
        graphicsManager->SetContourMinMax(minMax.MinMax);
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}

