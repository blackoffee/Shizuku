#pragma once

namespace Shizuku{ namespace Flow
{
    enum ShadingMode
    {
        RayTracing,
        SimplifiedRayTracing, //! reflection/refraction for floor and environment only.
        SimplifiedTransparency,
        Phong,
        NUMB_SHADING_MODE
    };
} }