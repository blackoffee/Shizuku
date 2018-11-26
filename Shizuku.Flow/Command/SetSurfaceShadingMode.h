#pragma once
#include "Command.h"

namespace Shizuku{ namespace Flow{ namespace Command{
    enum FLOW_API SurfaceShadingMode
    {
        RayTracing,
        SimplifiedRayTracing,
        Phong,
        NUMB_SHADING_MODE
    };

    class FLOW_API SetSurfaceShadingMode : public Command
    {
    public:
        SetSurfaceShadingMode(Flow& p_flow);
        void Start(const SurfaceShadingMode p_mode);
    };
} } }
