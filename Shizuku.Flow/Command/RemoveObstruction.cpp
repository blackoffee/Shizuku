#include "RemoveObstruction.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

RemoveObstruction::RemoveObstruction(Flow& p_flow) : Command(p_flow)
{
    m_currentObst = -1;
    m_state = INACTIVE;
}

void RemoveObstruction::Start(const float currentX, const float currentY)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    m_currentObst = graphicsManager->PickObstruction(currentX, currentY);
        if (m_currentObst >= 0)
    {
        m_state = ACTIVE;
    }
    else
    {
        m_state = INACTIVE;
    }
}

void RemoveObstruction::End(const float currentX, const float currentY)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    if (m_state == ACTIVE)
    {
        if (m_currentObst == graphicsManager->PickObstruction(currentX, currentY))
        {
            graphicsManager->RemoveSpecifiedObstruction(m_currentObst); 
        }
    }
    m_currentObst = -1;
    m_state = INACTIVE;
}

