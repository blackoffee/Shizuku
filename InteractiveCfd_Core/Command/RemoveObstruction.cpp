#include "RemoveObstruction.h"
#include "Graphics/GraphicsManager.h"

RemoveObstruction::RemoveObstruction(Panel &rootPanel) : Command(rootPanel)
{
    m_rootPanel = &rootPanel;
    m_currentObst = -1;
    m_state = INACTIVE;
}

void RemoveObstruction::Start(const float currentX, const float currentY)
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
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
    GraphicsManager* graphicsManager = GetGraphicsManager();
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

