#include "AddObstruction.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

AddObstruction::AddObstruction(Flow& p_flow) : Command(p_flow)
{
    m_state = INACTIVE;
}

void AddObstruction::Start(const float currentX, const float currentY)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    int simX, simY;
    graphicsManager->GetSimCoordFromMouseRay(simX, simY, currentX, currentY, -0.5f);
    graphicsManager->AddObstruction(simX, simY);
}
