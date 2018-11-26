#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API MoveObstruction : public Command
    {
        int m_currentObst;
        float m_initialX;
        float m_initialY;
    public:
        MoveObstruction(Flow& p_flow);
        void Start(const float currentX, const float currentY);
        void Track(const float currentX, const float currentY);
        void End();
    };
} } }