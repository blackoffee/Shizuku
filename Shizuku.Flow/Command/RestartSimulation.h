#pragma once
#include "Command.h"
#include <boost/none.hpp>

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API RestartSimulation : public Command
    {
    public:
        RestartSimulation(Flow& p_flow);
        void Start(boost::any const p_param = boost::none);
    };
} } }
