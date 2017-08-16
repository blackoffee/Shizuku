#include "ButtonGroup.h"
#include "Button.h"
#include "Slider.h"
#include "Panel.h"

ButtonGroup::ButtonGroup()
{
}

ButtonGroup::ButtonGroup(const std::string name, std::vector<Button*> &buttons)
{
    m_name = name;
    m_buttons = buttons;
}

std::string ButtonGroup::GetName()
{
    return m_name;
}

void ButtonGroup::AddButton(Button* button)
{
    m_buttons.push_back(button);
}

void ButtonGroup::ExclusiveEnable(Button* button)
{
    for (std::vector<Button*>::iterator it = m_buttons.begin(); it != m_buttons.end(); ++it)
    {
        const bool isSelectedButton = (*it == button);
        (*it)->SetHighlight(isSelectedButton);

        Slider* sliderForButton = (*it)->GetRootPanel()->GetSlider((*it)->GetName());
        if (sliderForButton == NULL)
        {
            continue;
        }
        if (*it == button)
        {
            sliderForButton->Show();
        }
        else
        {
            sliderForButton->Hide();
        }
    }
}

Button* ButtonGroup::GetCurrentEnabledButton()
{
    for (std::vector<Button*>::iterator it = m_buttons.begin(); it != m_buttons.end(); ++it)
    {
        if ((*it)->m_highlighted == true)
        {
            return *it;
        }
    }
    return NULL;
}

