#pragma once
#include "Command.h"

class FLOW_API AddObstruction : public Command
{
public:
    AddObstruction(GraphicsManager &graphicsManager);
    void Start(const float currentX, const float currentY);
};

