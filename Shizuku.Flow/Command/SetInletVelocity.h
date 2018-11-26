#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API SetInletVelocity : public Command
    {
    public:
        SetInletVelocity(GraphicsManager &graphicsManager);
        void Start(boost::any const p_param);
    };
} } }

