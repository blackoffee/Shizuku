#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API RemoveObstruction : public Command
    {
        int m_currentObst;
    public:
        RemoveObstruction(Flow& p_flow);
        void Start(boost::any const p_param);
        void End(boost::any const p_param);
    };
} } }