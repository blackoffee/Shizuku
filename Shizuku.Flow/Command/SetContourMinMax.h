#pragma once
#include "Command.h"
#include "Shizuku.Core/Types/MinMax.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API SetContourMinMax : public Command
    {
    public:
        SetContourMinMax(Flow& p_flow);
        void Start(boost::any const p_param);
    };
} } }