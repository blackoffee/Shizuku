#include "MoveObstruction.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

MoveObstruction::MoveObstruction(Flow& p_flow) : Command(p_flow)
{
    m_currentObst = -1;
    m_state = INACTIVE;
}

void MoveObstruction::Start(const float currentX, const float currentY)
{
    m_initialX = currentX;
    m_initialY = currentY;
    m_currentObst = m_flow->Graphics()->PickObstruction(currentX, currentY);
    if (m_currentObst >= 0)
    {
        m_state = ACTIVE;
    }
    else
    {
        m_state = INACTIVE;
    }
}

void MoveObstruction::Track(const float currentX, const float currentY)
{
    if (m_state == ACTIVE)
    {
        float dx = currentX - m_initialX;
        float dy = currentY - m_initialY;
        m_flow->Graphics()->MoveObstruction(m_currentObst, currentX, currentY, dx, dy);
    }
    m_initialX = currentX;
    m_initialY = currentY;
}

void MoveObstruction::End()
{
    m_currentObst = -1;
    m_state = INACTIVE;
}

