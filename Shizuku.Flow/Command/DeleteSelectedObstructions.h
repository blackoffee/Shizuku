#pragma once
#include "Command.h"
#include "Shizuku.Core/Types/Point.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API DeleteSelectedObstructions : public Command
    {
    public:
        DeleteSelectedObstructions(Flow& p_flow);
        void Start(boost::any const p_param);
    };
} } }