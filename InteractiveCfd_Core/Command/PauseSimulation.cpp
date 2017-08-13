#include "PauseSimulation.h"
#include "Graphics/GraphicsManager.h"
#include "Graphics/CudaLbm.h"

PauseSimulation::PauseSimulation(Panel &rootPanel) : Command(rootPanel)
{
    m_rootPanel = &rootPanel;
    m_state = INACTIVE;
}

void PauseSimulation::Start()
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    graphicsManager->GetCudaLbm()->SetPausedState(true);
}

void PauseSimulation::End()
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    graphicsManager->GetCudaLbm()->SetPausedState(false);
}