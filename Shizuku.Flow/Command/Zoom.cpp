#include "Zoom.h"
#include "Graphics/GraphicsManager.h"

using namespace Shizuku::Flow::Command;

Zoom::Zoom(GraphicsManager &graphicsManager) : Command(graphicsManager)
{
}

void Zoom::Start(const int dir, const float mag)
{
    GetGraphicsManager()->Zoom(dir, mag);
}


