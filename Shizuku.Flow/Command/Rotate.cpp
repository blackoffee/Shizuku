#include "Rotate.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"
#include "Parameter/ScreenPointParameter.h"

using namespace Shizuku::Flow::Command;

Rotate::Rotate(Flow& p_flow) : Command(p_flow)
{
    m_state = Inactive;
}

void Rotate::Start(boost::any const p_param)
{
    try
    {
        const ScreenPointParameter& pos = boost::any_cast<ScreenPointParameter>(p_param);
        m_state = Active;
        m_initialPos = pos.Position;
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}

void Rotate::Track(boost::any const p_param)
{
    try
    {
        const ScreenPointParameter& pos = boost::any_cast<ScreenPointParameter>(p_param);
        if (m_state == Active)
        {
            Point<int> posDiff = pos.Position - m_initialPos;
            m_flow->Graphics()->Rotate(posDiff);
        }

        m_initialPos = pos.Position;
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}

void Rotate::End(boost::any const p_param)
{
    m_state = Inactive;
}

