#include "SetViscosity.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

SetViscosity::SetViscosity(Flow& p_flow) : Command(p_flow)
{
}

void SetViscosity::Start(const float p_viscosity)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    graphicsManager->SetViscosity(p_viscosity);
}

