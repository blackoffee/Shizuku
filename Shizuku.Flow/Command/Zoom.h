#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API Zoom : public Command
    {
    public:
        Zoom(GraphicsManager &graphicsManager);
        void Start(const int dir, const float mag);
    };
} } }
