#include "AddPreSelectionToSelection.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"
#include "Parameter/ScreenPointParameter.h"

using namespace Shizuku::Core::Types;
using namespace Shizuku::Flow::Command;

AddPreSelectionToSelection::AddPreSelectionToSelection(Flow& p_flow) : Command(p_flow)
{
    m_state = Inactive;
}

void AddPreSelectionToSelection::Start(boost::any const p_param)
{
    m_state = Active;
}

void AddPreSelectionToSelection::Track(boost::any const p_param)
{
}

void AddPreSelectionToSelection::End(boost::any const p_param)
{
	if (m_state == Active)
	{
		GraphicsManager* graphicsManager= m_flow->Graphics();
		graphicsManager->AddPreSelectionToSelection();
		m_state = Inactive;
	}
}

