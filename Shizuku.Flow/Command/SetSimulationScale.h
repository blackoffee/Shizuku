#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API SetSimulationScale : public Command
    {
    public:
        SetSimulationScale(GraphicsManager &graphicsManager);
        void Start(const float p_scale);
    };
} } }
