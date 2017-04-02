#include "Pan.h"
#include "Graphics/GraphicsManager.h"

Pan::Pan(Panel &rootPanel) : Command(rootPanel)
{
    m_rootPanel = &rootPanel;
    m_state = INACTIVE;
}

void Pan::Start(const float initialX, const float initialY)
{
    m_state = ACTIVE;
    m_initialX = initialX;
    m_initialY = initialY;
}

void Pan::Track(const float currentX, const float currentY)
{
    float dx = currentX - m_initialX;
    float dy = currentY - m_initialY;
    if (m_state == ACTIVE)
    {
        GetGraphicsManager()->Pan(dx, dy);
    }
    m_initialX = currentX;
    m_initialY = currentY;
}

void Pan::End()
{
    m_state = INACTIVE;
}