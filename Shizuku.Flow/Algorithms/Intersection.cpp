#include "Intersection.h"
#include <algorithm>

using namespace Shizuku::Core;
using namespace Shizuku::Flow::Algorithms;

bool Intersection::IntersectAABBWithRay(float& p_dist, const glm::vec3& p_rayOrigin,
	const glm::vec3& p_rayDir, const Types::Point3D<float>& p_boxPos, const Types::Box<float>& p_box)
{
	const glm::vec3 boxMin = glm::vec3(p_boxPos.X - 0.5*p_box.Width, p_boxPos.Y - 0.5*p_box.Height, p_boxPos.Z - 0.5*p_box.Depth);
	const glm::vec3 boxMax = glm::vec3(p_boxPos.X + 0.5*p_box.Width, p_boxPos.Y + 0.5*p_box.Height, p_boxPos.Z + 0.5*p_box.Depth);

	const glm::vec3 t0 = (boxMin - p_rayOrigin) / p_rayDir;
	const glm::vec3 t1 = (boxMax - p_rayOrigin) / p_rayDir;

	glm::vec3 tMin = t0;
	glm::vec3 tMax = t1;

	if (t0.x > t1.x)
		std::swap(tMin.x, tMax.x);
	if (t0.y > t1.y)
		std::swap(tMin.y, tMax.y);
	if (t0.z > t1.z)
		std::swap(tMin.z, tMax.z);

    if (tMax.x < 0 || tMax.y < 0 || tMax.z < 0)
        return false;

    if (tMin.x > tMax.y || tMin.x > tMax.z || tMin.y > tMax.x || tMin.y > tMax.z || tMin.z > tMax.x || tMin.z > tMax.y)
        return false;

	p_dist = std::max(std::max(tMin.x, tMin.y), tMin.z);
    return true;
}
