#pragma once
#include "Command.h"
#include "Shizuku.Core/Types/Point.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API ProbeLightPaths : public Command
    {
    public:
        ProbeLightPaths(Flow& p_flow);
        void Start(boost::any const p_param);
        void Track(boost::any const p_param);
        void End(boost::any const p_param);
    };
} } }