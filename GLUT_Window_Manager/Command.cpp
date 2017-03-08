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

void Zoom::Start(Panel &rootPanel, const int dir, const float mag)
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

