#pragma once
#include "Command.h"

class FLOW_API SetInletVelocity : public Command
{
public:
    SetInletVelocity(GraphicsManager &graphicsManager);
    void Start(const float p_velocity);
};

