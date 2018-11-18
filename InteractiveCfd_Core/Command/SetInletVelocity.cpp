#include "SetInletVelocity.h"
#include "Graphics/GraphicsManager.h"

SetInletVelocity::SetInletVelocity(GraphicsManager &p_graphicsManager) : Command(p_graphicsManager)
{
}

void SetInletVelocity::Start(const float p_velocity)
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    graphicsManager->SetVelocity(p_velocity);
}

