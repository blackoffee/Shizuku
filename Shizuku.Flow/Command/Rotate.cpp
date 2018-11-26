#include "Rotate.h"
#include "Graphics/GraphicsManager.h"

using namespace Shizuku::Flow::Command;

Rotate::Rotate(GraphicsManager &rootGraphicsManager) : Command(rootGraphicsManager)
{
    m_state = INACTIVE;
}

void Rotate::Start(const float initialX, const float initialY)
{
    m_state = ACTIVE;
    m_initialX = initialX;
    m_initialY = initialY;
}

void Rotate::Track(const float currentX, const float currentY)
{
    float dx = (currentX - m_initialX)*45.f;
    float dy = (currentY - m_initialY)*45.f;
    if (m_state == ACTIVE)
    {
        GetGraphicsManager()->Rotate(dx, dy);
    }
    m_initialX = currentX;
    m_initialY = currentY;
}

void Rotate::End()
{
    m_state = INACTIVE;
}

