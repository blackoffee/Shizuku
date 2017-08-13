#include "Button.h"

Button::Button(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition, 
    const std::string name, const Color color, Panel* parent) 
    : Panel(rectFloat, sizeDefinition, name, color, parent)
{
    SetDisplayText(GetName());
}

Button::Button(const RectInt rectInt, const SizeDefinitionMethod sizeDefinition, 
    const std::string name, const Color color, Panel* parent) 
    : Panel(rectInt, sizeDefinition, name, color, parent)
{
    SetDisplayText(GetName());
}

void Button::SetCallback(void(*callback)(Panel &rootPanel))
{
    m_callback = callback;
}

void Button::Callback()
{
    if (m_callback != NULL)
    {
        m_callback(*GetRootPanel());
    }
}

void Button::ClickDown()
{
    if (m_callback != NULL)
    {
        m_callback(*GetRootPanel());
    }
}

void Button::SetHighlight(const bool state)
{
    m_highlighted = state;

    if (state == true)
    {
        SetBackgroundColor(Color::LIGHT_GRAY);
    }
    else
    {
        SetBackgroundColor(Color::GRAY);
    }
}
