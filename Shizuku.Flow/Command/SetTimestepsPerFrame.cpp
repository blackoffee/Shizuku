#include "SetTimestepsPerFrame.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

SetTimestepsPerFrame::SetTimestepsPerFrame(Flow& p_flow) : Command(p_flow)
{
}

void SetTimestepsPerFrame::Start(const int p_steps)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    graphicsManager->SetTimestepsPerFrame(p_steps);
}

