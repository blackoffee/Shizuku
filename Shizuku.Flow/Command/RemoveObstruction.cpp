#include "RemoveObstruction.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"
#include "Parameter/ScreenPointParameter.h"

using namespace Shizuku::Flow::Command;

RemoveObstruction::RemoveObstruction(Flow& p_flow) : Command(p_flow)
{
    m_currentObst = -1;
    m_state = Inactive;
}

void RemoveObstruction::Start(boost::any const p_param)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    try
    {
        const ScreenPointParameter& pos = boost::any_cast<ScreenPointParameter>(p_param);
        m_currentObst = graphicsManager->PickObstruction(pos.Position);

        if (m_currentObst >= 0)
        {
            m_state = Active;
        }
        else
        {
            m_state = Inactive;
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
        if (m_state == Active)
        {
            if (m_currentObst == graphicsManager->PickObstruction(pos.Position))
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
