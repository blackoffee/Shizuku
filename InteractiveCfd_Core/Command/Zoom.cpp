#include "Zoom.h"
#include "GraphicsManager.h"

Zoom::Zoom(Panel &rootPanel) : Command(rootPanel)
{
    m_rootPanel = &rootPanel;
}

void Zoom::Start(const int dir, const float mag)
{
    GetGraphicsManager()->Zoom(dir, mag);
}


