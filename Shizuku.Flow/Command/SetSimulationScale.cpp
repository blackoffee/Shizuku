#include "SetSimulationScale.h"
#include "Graphics/GraphicsManager.h"

using namespace Shizuku::Flow::Command;

SetSimulationScale::SetSimulationScale(GraphicsManager &p_graphicsManager) : Command(p_graphicsManager)
{
}

void SetSimulationScale::Start(const float p_scale)
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    graphicsManager->SetScaleFactor(p_scale);
}

