#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API PauseRayTracing : public Command
    {
    public:
        PauseRayTracing(GraphicsManager &graphicsManager);
        void Start();
        void End();
    };
} } }
