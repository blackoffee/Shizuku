#pragma once
#include "Command.h"

enum FW_API ContourMode{
    VelocityMagnitude,
    VelocityU,
    VelocityV,
    Pressure,
    StrainRate,
    Water,
    NUMB_CONTOUR_MODE //max
};

class FW_API SetContourMode : public Command
{
public:
    SetContourMode(GraphicsManager &graphicsManager);
    void Start(const ContourMode p_contourMode);
};

