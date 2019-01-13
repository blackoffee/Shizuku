#include "DeleteSelectedObstructions.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"
#include "Parameter/ScreenPointParameter.h"

using namespace Shizuku::Core::Types;
using namespace Shizuku::Flow::Command;

DeleteSelectedObstructions::DeleteSelectedObstructions(Flow& p_flow) : Command(p_flow)
{
    m_state = Inactive;
}

void DeleteSelectedObstructions::Start(boost::any const p_param)
{
    m_state = Active;
}

void DeleteSelectedObstructions::Track(boost::any const p_param)
{
}

void DeleteSelectedObstructions::End(boost::any const p_param)
{
	if (m_state == Active)
	{
		GraphicsManager* graphicsManager= m_flow->Graphics();
		graphicsManager->DeleteSelectedObstructions();
		m_state = Inactive;
	}
}

