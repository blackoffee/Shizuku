#include "PauseRayTracing.h"
#include "Graphics/GraphicsManager.h"
#include "Graphics/CudaLbm.h"

PauseRayTracing::PauseRayTracing(Panel &rootPanel) : Command(rootPanel)
{
    m_rootPanel = &rootPanel;
    m_state = INACTIVE;
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