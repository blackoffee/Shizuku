#pragma once
#include "Command.h"

class FW_API AddObstruction : public Command
{
public:
    AddObstruction(GraphicsManager &graphicsManager);
    void Start(const float currentX, const float currentY);
};

