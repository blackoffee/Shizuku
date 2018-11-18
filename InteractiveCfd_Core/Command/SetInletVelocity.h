#pragma once
#include "Command.h"

class FW_API SetInletVelocity : public Command
{
public:
    SetInletVelocity(GraphicsManager &graphicsManager);
    void Start(const float p_velocity);
};

