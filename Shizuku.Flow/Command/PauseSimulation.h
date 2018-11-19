#pragma once
#include "Command.h"

class FLOW_API PauseSimulation : public Command
{
public:
    PauseSimulation(GraphicsManager &graphicsManager);
    void Start();
    void End();
};
