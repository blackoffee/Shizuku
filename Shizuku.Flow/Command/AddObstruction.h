#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API AddObstruction : public Command
    {
    public:
        AddObstruction(GraphicsManager &graphicsManager);
        void Start(const float currentX, const float currentY);
    };
} } }
