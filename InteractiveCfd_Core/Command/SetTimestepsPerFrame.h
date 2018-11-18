#pragma once
#include "Command.h"

class FW_API SetTimestepsPerFrame : public Command
{
public:
    SetTimestepsPerFrame(GraphicsManager &graphicsManager);
    void Start(const int p_steps);
};

