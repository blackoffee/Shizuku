#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API PauseSimulation : public Command
    {
    public:
        PauseSimulation(GraphicsManager &graphicsManager);
        void Start();
        void End();
    };
} } }
