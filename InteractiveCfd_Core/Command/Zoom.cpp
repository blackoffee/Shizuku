#include "Zoom.h"
#include "Graphics/GraphicsManager.h"

Zoom::Zoom(GraphicsManager &graphicsManager) : Command(graphicsManager)
{
}

void Zoom::Start(const int dir, const float mag)
{
    GetGraphicsManager()->Zoom(dir, mag);
}


