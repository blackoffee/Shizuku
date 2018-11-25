#pragma once
#include "Command.h"

//namespace Shizuku{
//    namespace Flow{
//        struct Parameter;
//    }
//}

//namespace Shizuku{ namespace Flow{
    class FLOW_API SetInletVelocity : public Command
    {
    public:
        SetInletVelocity(GraphicsManager &graphicsManager);
        void Start(boost::any const p_param);
    };
//}}

//class FLOW_API SetInletVelocity : public Command
//{
//public:
//    SetInletVelocity(GraphicsManager &graphicsManager);
//    void Start(const float p_velocity);
//};

