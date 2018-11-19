#pragma once
#include "Command.h"

class FLOW_API MoveObstruction : public Command
{
    int m_currentObst;
    float m_initialX;
    float m_initialY;
public:
    MoveObstruction(GraphicsManager &graphicsManager);
    void Start(const float currentX, const float currentY);
    void Track(const float currentX, const float currentY);
    void End();
};