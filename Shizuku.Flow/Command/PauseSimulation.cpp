#include "PauseSimulation.h"
#include "Graphics/GraphicsManager.h"
#include "Graphics/CudaLbm.h"

PauseSimulation::PauseSimulation(GraphicsManager &graphicsManager) : Command(graphicsManager)
{
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