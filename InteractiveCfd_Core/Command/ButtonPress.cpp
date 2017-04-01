#include "ButtonPress.h"
#include "Panel.h"

ButtonPress::ButtonPress(Panel &rootPanel) : Command(rootPanel)
{
    m_rootPanel = &rootPanel;
    m_state = INACTIVE;
}

void ButtonPress::Start(Button* button)
{
    m_button = button;
    m_state = ACTIVE;
}

void ButtonPress::End(Button* button)
{
    if (m_button == button)
    {
        m_button->Callback();
    }
    m_state = INACTIVE;
}
