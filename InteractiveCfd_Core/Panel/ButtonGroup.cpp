#include "ButtonGroup.h"
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
        Slider* sliderForButton(NULL);
        if ((*it)->GetRootPanel()->GetSlider((*it)->GetName()) != NULL){
            sliderForButton = (*it)->GetRootPanel()->GetSlider((*it)->GetName());
        }
        if (*it == button)
        {
            (*it)->m_highlighted = true;
            (*it)->SetBackgroundColor(Color::LIGHT_GRAY);
            if (sliderForButton != NULL)
            {
                sliderForButton->Show();
            }
        }
        else
        {
            (*it)->m_highlighted = false;
            (*it)->SetBackgroundColor(Color::GRAY);
            if (sliderForButton != NULL)
            {
                sliderForButton->Hide();
            }
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

