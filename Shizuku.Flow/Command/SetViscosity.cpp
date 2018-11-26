#include "SetViscosity.h"
#include "Graphics/GraphicsManager.h"

using namespace Shizuku::Flow::Command;

SetViscosity::SetViscosity(GraphicsManager &p_graphicsManager) : Command(p_graphicsManager)
{
}

void SetViscosity::Start(const float p_viscosity)
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    graphicsManager->SetViscosity(p_viscosity);
}

