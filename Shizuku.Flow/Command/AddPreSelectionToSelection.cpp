#include "AddPreSelectionToSelection.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

AddPreSelectionToSelection::AddPreSelectionToSelection(Flow& p_flow) : Command(p_flow)
{
}

void AddPreSelectionToSelection::Start(boost::any const p_param)
{

	GraphicsManager* graphicsManager= m_flow->Graphics();
	graphicsManager->AddPreSelectionToSelection();
}

void AddPreSelectionToSelection::Track(boost::any const p_param)
{
}

void AddPreSelectionToSelection::End(boost::any const p_param)
{
}

