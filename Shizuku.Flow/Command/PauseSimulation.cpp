#include "PauseSimulation.h"
#include "Graphics/GraphicsManager.h"
#include "Graphics/CudaLbm.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

PauseSimulation::PauseSimulation(Flow& p_flow) : Command(p_flow)
{
}

void PauseSimulation::Start()
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    graphicsManager->GetCudaLbm()->SetPausedState(true);
}

void PauseSimulation::End()
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    graphicsManager->GetCudaLbm()->SetPausedState(false);
}