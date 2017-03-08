#include "Command.h"

Command::Command()
{
}

void Command::Initialize(Panel &rootPanel)
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

Zoom::Zoom()
{
}

void Zoom::Start(const int dir, const float mag)
{
    GetGraphicsManager()->Zoom(dir, mag);
}



Pan::Pan()
{
    m_state = UNACTIVE;
}

void Pan::Start(const float initialX, const float initialY)
{
    m_state = ACTIVE;
    m_initialX = initialX;
    m_initialY = initialY;
}

void Pan::Track(const float currentX, const float currentY)
{
    float dx = currentX - m_initialX;
    float dy = currentY - m_initialY;
    if (m_state == ACTIVE)
    {
        GetGraphicsManager()->Pan(dx, dy);
    }
    m_initialX = currentX;
    m_initialY = currentY;
}

void Pan::End()
{
    m_state = UNACTIVE;
}


Rotate::Rotate()
{
    m_state = UNACTIVE;
}

void Rotate::Start(const float initialX, const float initialY)
{
    m_state = ACTIVE;
    m_initialX = initialX;
    m_initialY = initialY;
}

void Rotate::Track(const float currentX, const float currentY)
{
    float dx = (currentX - m_initialX)*45.f;
    float dy = (currentY - m_initialY)*45.f;
    if (m_state == ACTIVE)
    {
        GetGraphicsManager()->Rotate(dx, dy);
    }
    m_initialX = currentX;
    m_initialY = currentY;
}

void Rotate::End()
{
    m_state = UNACTIVE;
}

ButtonPress::ButtonPress()
{
    m_state = UNACTIVE;
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
    m_state = UNACTIVE;
}

SliderDrag::SliderDrag()
{
    m_state = UNACTIVE;
    m_sliderBar = NULL;
}

void SliderDrag::Start(SliderBar* sliderBar, const float currentX, const float currentY)
{
    m_sliderBar = sliderBar;
    m_state = ACTIVE;
    m_initialX = currentX;
    m_initialY = currentY;
}

void SliderDrag::Track(const float currentX, const float currentY)
{
    float dx = currentX - m_initialX;
    float dy = currentY - m_initialY;
    if (m_state == ACTIVE)
    {
        int dummy = 0;
        m_sliderBar->Drag(dummy, dummy, dx, dy, dummy);
    }
    m_initialX = currentX;
    m_initialY = currentY;
}

void SliderDrag::End()
{
    m_sliderBar = NULL;
    m_state = UNACTIVE;
}
