#include "SetSimulationScale.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

SetSimulationScale::SetSimulationScale(Flow& p_flow) : Command(p_flow)
{
}

void SetSimulationScale::Start(const float p_scale)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    graphicsManager->SetScaleFactor(p_scale);
}

