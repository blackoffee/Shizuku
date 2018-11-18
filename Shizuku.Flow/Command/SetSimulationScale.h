#pragma once
#include "Command.h"

class FW_API SetSimulationScale : public Command
{
public:
    SetSimulationScale(GraphicsManager &graphicsManager);
    void Start(const float p_scale);
};

