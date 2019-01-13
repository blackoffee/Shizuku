#include "Pan.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"
#include "Parameter/ScreenPointParameter.h"

using namespace Shizuku::Flow::Command;

Pan::Pan(Flow& p_flow) : Command(p_flow)
{
    m_state = Inactive;
}

void Pan::Start(boost::any const p_param)
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

void Pan::Track(boost::any const p_param)
{
    try
    {
        const ScreenPointParameter& pos = boost::any_cast<ScreenPointParameter>(p_param);
        if (m_state == Active)
        {
            Point<int> posDiff = pos.Position - m_initialPos;
            m_flow->Graphics()->Pan(posDiff);
        }

        m_initialPos = pos.Position;
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}

void Pan::End(boost::any const p_param)
{
    m_state = Inactive;
}

