#include "AddObstruction.h"
#include "Graphics/GraphicsManager.h"

using namespace Shizuku::Flow::Command;

AddObstruction::AddObstruction(GraphicsManager &graphicsManager) : Command(graphicsManager)
{
    m_state = INACTIVE;
}

void AddObstruction::Start(const float currentX, const float currentY)
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    int simX, simY;
    graphicsManager->GetSimCoordFromMouseRay(simX, simY, currentX, currentY, -0.5f);
    graphicsManager->AddObstruction(simX, simY);
}
