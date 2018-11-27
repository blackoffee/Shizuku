#include "RemoveObstruction.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"
#include "Parameter/ScreenPointParameter.h"

using namespace Shizuku::Flow::Command;

RemoveObstruction::RemoveObstruction(Flow& p_flow) : Command(p_flow)
{
    m_currentObst = -1;
    m_state = INACTIVE;
}

void RemoveObstruction::Start(boost::any const p_param)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    try
    {
        const ScreenPointParameter& pos = boost::any_cast<ScreenPointParameter>(p_param);
        m_currentObst = graphicsManager->PickObstruction(pos.position);

        if (m_currentObst >= 0)
        {
            m_state = ACTIVE;
        }
        else
        {
            m_state = INACTIVE;
        }
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}

void RemoveObstruction::End(boost::any const p_param)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    try
    {
        const ScreenPointParameter& pos = boost::any_cast<ScreenPointParameter>(p_param);
        if (m_state == ACTIVE)
        {
            if (m_currentObst == graphicsManager->PickObstruction(pos.position))
            {
                graphicsManager->RemoveSpecifiedObstruction(m_currentObst); 
            }
        }
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}
