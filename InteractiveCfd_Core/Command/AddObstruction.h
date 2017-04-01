#pragma once
#include "Command.h"

class FW_API AddObstruction : public Command
{
public:
    AddObstruction(Panel &rootPanel);
    void Start(const float currentX, const float currentY);
};

