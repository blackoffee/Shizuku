#include "TogglePreSelection.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

TogglePreSelection::TogglePreSelection(Flow& p_flow) : Command(p_flow)
{
}

void TogglePreSelection::Start(boost::any const p_param)
{

	GraphicsManager* graphicsManager= m_flow->Graphics();
	graphicsManager->TogglePreSelection();
}

void TogglePreSelection::Track(boost::any const p_param)
{
}

void TogglePreSelection::End(boost::any const p_param)
{
}

