#pragma once

#include "Shizuku.Core/Types/Box.h"
#include "Shizuku.Core/Types/Point3D.h"

#include <glm/glm.hpp>

using namespace Shizuku::Core;

namespace Shizuku{ namespace Flow {namespace Algorithms
{
    static class Intersection
    {
    public:
        // Returns the distance of intersection from ray origin
        static bool IntersectAABBWithRay(float& p_dist, const glm::vec3& p_rayOrigin, const glm::vec3& p_rayDir,
            const Types::Point3D<float>& p_boxPos, const Types::Box<float>& p_box);
    };
}}}
