#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API SetTimestepsPerFrame : public Command
    {
    public:
        SetTimestepsPerFrame(Flow& p_flow);
        void Start(const int p_steps);
    };
} } }
