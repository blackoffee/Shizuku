#pragma once
#include "Command.h"

class FW_API PauseRayTracing : public Command
{
public:
    PauseRayTracing(Panel &rootPanel);
    void Start();
    void End();
};
