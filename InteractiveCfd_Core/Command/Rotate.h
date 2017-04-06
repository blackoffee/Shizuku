#pragma once
#include "Command.h"

class FW_API Rotate : public Command
{
    float m_initialX;
    float m_initialY;
public:
    Rotate(Panel &rootPanel);
    void Start(const float initialX, const float initialY);
    void Track(const float currentX, const float currentY);
    void End();
};