#include "Command.h"
#include "GraphicsManager.h"
#include "Panel.h"

Command::Command(Panel &rootPanel)
{
    m_rootPanel = &rootPanel;
}

Panel* Command::GetRootPanel()
{
    return m_rootPanel;
}

GraphicsManager* Command::GetGraphicsManager()
{
    return m_rootPanel->GetPanel("Graphics")->GetGraphicsManager();
}

