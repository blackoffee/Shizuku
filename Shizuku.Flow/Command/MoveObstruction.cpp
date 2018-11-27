#include "MoveObstruction.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"
#include "Parameter/ScreenPointParameter.h"

using namespace Shizuku::Core::Types;
using namespace Shizuku::Flow::Command;

MoveObstruction::MoveObstruction(Flow& p_flow) : Command(p_flow)
{
    m_currentObst = -1;
    m_state = INACTIVE;
    m_initialPos = Point<int>(0, 0);
}

void MoveObstruction::Start(boost::any const p_param)
{
    try
    {
        const ScreenPointParameter& pos = boost::any_cast<ScreenPointParameter>(p_param);
        m_initialPos = pos.position;
        m_currentObst = m_flow->Graphics()->PickObstruction(pos.position);

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

void MoveObstruction::Track(boost::any const p_param)
{
    try
    {
        const ScreenPointParameter& pos = boost::any_cast<ScreenPointParameter>(p_param);
        if (m_state == ACTIVE)
        {
            const Point<int> posDiff = pos.position - m_initialPos;
            m_flow->Graphics()->MoveObstruction(m_currentObst, pos.position, posDiff);

            if (m_currentObst >= 0)
            {
                m_state = ACTIVE;
            }
            else
            {
                m_state = INACTIVE;
            }
        }

        m_initialPos = pos.position;
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}

void MoveObstruction::End(boost::any const p_param)
{
    m_currentObst = -1;
    m_state = INACTIVE;
}

