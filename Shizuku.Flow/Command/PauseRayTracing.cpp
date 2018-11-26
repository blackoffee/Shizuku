#include "PauseRayTracing.h"
#include "Graphics/GraphicsManager.h"

using namespace Shizuku::Flow::Command;

PauseRayTracing::PauseRayTracing(GraphicsManager &graphicsManager) : Command(graphicsManager)
{
}

void PauseRayTracing::Start()
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    graphicsManager->SetRayTracingPausedState(true);
}

void PauseRayTracing::End()
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    graphicsManager->SetRayTracingPausedState(false);
}