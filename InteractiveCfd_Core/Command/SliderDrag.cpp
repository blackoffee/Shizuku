#include "SliderDrag.h"
#include "Panel/Panel.h"
#include "Panel/SliderBar.h"

SliderDrag::SliderDrag(Panel &rootPanel) : Command(rootPanel)
{
    m_rootPanel = &rootPanel;
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
