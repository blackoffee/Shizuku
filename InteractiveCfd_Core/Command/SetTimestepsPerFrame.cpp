#include "SetTimestepsPerFrame.h"
#include "Graphics/GraphicsManager.h"

SetTimestepsPerFrame::SetTimestepsPerFrame(GraphicsManager &p_graphicsManager) : Command(p_graphicsManager)
{
}

void SetTimestepsPerFrame::Start(const int p_steps)
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    graphicsManager->SetTimestepsPerFrame(p_steps);
}

