#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API SetSimulationScale : public Command
    {
    public:
        SetSimulationScale(Flow& p_flow);
        void Start(const float p_scale);
    };
} } }
