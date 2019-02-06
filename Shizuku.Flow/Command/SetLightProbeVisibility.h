#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API SetLightProbeVisibility : public Command
    {
    public:
        SetLightProbeVisibility(Flow& p_flow);
        void Start(boost::any const p_param);
    };
} } }

