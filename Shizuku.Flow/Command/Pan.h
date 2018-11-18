#pragma once
#include "Command.h"

class FW_API Pan : public Command
{
    float m_initialX;
    float m_initialY;
public:
    Pan(GraphicsManager &graphicsManager);
    void Start(const float initialX, const float initialY);
    void Track(const float currentX, const float currentY);
    void End();
};


