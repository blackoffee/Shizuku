#include "SetContourMinMax.h"
#include "Graphics/GraphicsManager.h"
#include "common.h"

using namespace Shizuku::Flow::Command;

SetContourMinMax::SetContourMinMax(GraphicsManager &p_graphicsManager) : Command(p_graphicsManager)
{
}

void SetContourMinMax::Start(const Shizuku::Core::MinMax<float>& p_minMax)
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    graphicsManager->SetContourMinMax(p_minMax);
}

