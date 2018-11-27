#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API AddObstruction : public Command
    {
    public:
        AddObstruction(Flow& p_flow);
        void Start(boost::any const p_param);
    };
} } }
