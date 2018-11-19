#pragma once
#include "Command.h"
#include "Shizuku.Core/Types/MinMax.h"

class FLOW_API SetContourMinMax : public Command
{
public:
    SetContourMinMax(GraphicsManager &graphicsManager);
    void Start(const Shizuku::Core::MinMax<float>& p_contourMode);
};

