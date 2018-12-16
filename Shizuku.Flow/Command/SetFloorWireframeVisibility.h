#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API SetFloorWireframeVisibility : public Command
    {
    public:
        SetFloorWireframeVisibility(Flow& p_flow);
        void Start(boost::any const p_param);
    };
} } }

