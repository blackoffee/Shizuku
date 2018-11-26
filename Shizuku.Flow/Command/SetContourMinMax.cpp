#include "SetContourMinMax.h"
#include "Graphics/GraphicsManager.h"
#include "common.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

SetContourMinMax::SetContourMinMax(Flow& p_flow) : Command(p_flow)
{
}

void SetContourMinMax::Start(const Shizuku::Core::MinMax<float>& p_minMax)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    graphicsManager->SetContourMinMax(p_minMax);
}

