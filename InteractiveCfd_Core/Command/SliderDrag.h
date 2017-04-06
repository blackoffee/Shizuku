#pragma once
#include "Command.h"

class FW_API SliderDrag : public Command
{
    float m_initialX;
    float m_initialY;
    SliderBar* m_sliderBar;
public:
    SliderDrag(Panel &rootPanel);
    void Start(SliderBar* sliderBar, const float currentX, const float currentY);
    void Track(const float currentX, const float currentY);
    void End();
};

