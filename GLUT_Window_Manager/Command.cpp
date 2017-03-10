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
    m_state = INACTIVE;
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
    m_state = INACTIVE;
}


Rotate::Rotate()
{
    m_state = INACTIVE;
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
    m_state = INACTIVE;
}

ButtonPress::ButtonPress()
{
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

SliderDrag::SliderDrag()
{
    m_state = INACTIVE;
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
    m_state = INACTIVE;
}

AddObstruction::AddObstruction()
{
    m_state = INACTIVE;
}

void AddObstruction::Start(const float currentX, const float currentY)
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    int simX, simY;
    graphicsManager->GetSimCoordFromMouseRay(simX, simY, currentX, currentY, -0.5f);
    graphicsManager->AddObstruction(simX, simY);
}

RemoveObstruction::RemoveObstruction()
{
    m_currentObst = -1;
    m_state = INACTIVE;
}

void RemoveObstruction::Start(const float currentX, const float currentY)
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    m_currentObst = graphicsManager->PickObstruction(currentX, currentY);
        if (m_currentObst >= 0)
    {
        m_state = ACTIVE;
    }
    else
    {
        m_state = INACTIVE;
    }
}

void RemoveObstruction::End(const float currentX, const float currentY)
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
    if (m_state == ACTIVE)
    {
        if (m_currentObst == graphicsManager->PickObstruction(currentX, currentY))
        {
            graphicsManager->RemoveSpecifiedObstruction(m_currentObst); 
        }
    }
    m_currentObst = -1;
    m_state = INACTIVE;
}

MoveObstruction::MoveObstruction()
{
    m_currentObst = -1;
    m_state = INACTIVE;
}

void MoveObstruction::Start(const float currentX, const float currentY)
{
    m_initialX = currentX;
    m_initialY = currentY;
    m_currentObst = GetGraphicsManager()->PickObstruction(currentX, currentY);
    if (m_currentObst >= 0)
    {
        m_state = ACTIVE;
    }
    else
    {
        m_state = INACTIVE;
    }
}

void MoveObstruction::Track(const float currentX, const float currentY)
{
    if (m_state == ACTIVE)
    {
        float dx = currentX - m_initialX;
        float dy = currentY - m_initialY;
        GetGraphicsManager()->MoveObstruction(m_currentObst, currentX, currentY, dx, dy);
    }
    m_initialX = currentX;
    m_initialY = currentY;
}

void MoveObstruction::End()
{
    m_currentObst = -1;
    m_state = INACTIVE;
}

