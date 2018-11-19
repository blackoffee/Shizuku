#pragma once
#include "Command.h"

class FLOW_API Zoom : public Command
{
public:
    Zoom(GraphicsManager &graphicsManager);
    void Start(const int dir, const float mag);
};

