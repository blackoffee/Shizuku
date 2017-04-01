#include "AddObstruction.h"
#include "GraphicsManager.h"

AddObstruction::AddObstruction(Panel &rootPanel) : Command(rootPanel)
{
    m_rootPanel = &rootPanel;
    m_state = INACTIVE;
}

void AddObstruction::Start(const float currentX, const float currentY)
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    int simX, simY;
    graphicsManager->GetSimCoordFromMouseRay(simX, simY, currentX, currentY, -0.5f);
    graphicsManager->AddObstruction(simX, simY);
}
