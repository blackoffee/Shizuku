#pragma once
#include "Command.h"

class FW_API Zoom : public Command
{
public:
    Zoom(GraphicsManager &graphicsManager);
    void Start(const int dir, const float mag);
};

