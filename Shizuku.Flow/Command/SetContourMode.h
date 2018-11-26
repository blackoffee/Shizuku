#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    enum FLOW_API ContourMode{
        VelocityMagnitude,
        VelocityU,
        VelocityV,
        Pressure,
        StrainRate,
        Water,
        NUMB_CONTOUR_MODE //max
    };


    class FLOW_API SetContourMode : public Command
    {
    public:
        SetContourMode(GraphicsManager &graphicsManager);
        void Start(const ContourMode p_contourMode);
    };
} } }
