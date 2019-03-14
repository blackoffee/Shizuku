#include "PreSelectObstruction.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"
#include "Parameter/ScreenPointParameter.h"

using namespace Shizuku::Core::Types;
using namespace Shizuku::Flow::Command;

PreSelectObstruction::PreSelectObstruction(Flow& p_flow) : Command(p_flow)
{
    m_state = Inactive;
}

void PreSelectObstruction::Start(boost::any const p_param)
{
    m_state = Active;
}

void PreSelectObstruction::Track(boost::any const p_param)
{
    try
    {
        if (m_state == Active)
        {
            const ScreenPointParameter& pos = boost::any_cast<ScreenPointParameter>(p_param);
            m_flow->Graphics()->PreSelectObstruction(pos.Position);
        }
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}

void PreSelectObstruction::End(boost::any const p_param)
{
    m_state = Inactive;
}

