#pragma once
#include "Command.h"

class FW_API PauseRayTracing : public Command
{
public:
    PauseRayTracing(GraphicsManager &graphicsManager);
    void Start();
    void End();
};
