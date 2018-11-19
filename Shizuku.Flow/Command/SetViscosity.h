#pragma once
#include "Command.h"

class FLOW_API SetViscosity : public Command
{
public:
    SetViscosity(GraphicsManager &graphicsManager);
    void Start(const float p_viscosity);
};

