#include "ClearSelection.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

ClearSelection::ClearSelection(Flow& p_flow) : Command(p_flow)
{
}

void ClearSelection::Start(boost::any const p_param)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    graphicsManager->ClearSelection();
}
