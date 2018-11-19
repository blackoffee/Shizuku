#pragma once
#include "Command.h"

class FLOW_API RemoveObstruction : public Command
{
    int m_currentObst;
public:
    RemoveObstruction(GraphicsManager &rootGraphicsManager);
    void Start(const float currentX, const float currentY);
    void End(const float currentX, const float currentY);
};