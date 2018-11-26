#include "PauseRayTracing.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

PauseRayTracing::PauseRayTracing(Flow& p_flow) : Command(p_flow)
{
}

void PauseRayTracing::Start()
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    graphicsManager->SetRayTracingPausedState(true);
}

void PauseRayTracing::End()
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    graphicsManager->SetRayTracingPausedState(false);
}