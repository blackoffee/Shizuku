#include "Zoom.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

Zoom::Zoom(Flow& p_flow) : Command(p_flow)
{
}

void Zoom::Start(const int dir, const float mag)
{
    m_flow->Graphics()->Zoom(dir, mag);
}


