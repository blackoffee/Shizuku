#pragma once
#include "Command.h"

class FW_API PauseSimulation : public Command
{
public:
    PauseSimulation(Panel &rootPanel);
    void Start();
    void End();
};
