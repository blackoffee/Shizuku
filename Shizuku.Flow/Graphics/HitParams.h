#pragma once

#include "Shizuku.Core/Types/Point.h"
#include "Shizuku.Core/Rect.h"

#include <glm/glm.hpp>
#include <boost/optional.hpp>

using namespace Shizuku::Core;

namespace Shizuku { namespace Flow{
	struct HitParams
	{
		Types::Point<int> ScreenPos;
		glm::mat4 Modelview;
		glm::mat4 Projection;
		Rect<int> ViewSize;
	};

	struct HitResult
	{
		bool Hit;
		boost::optional<float> Dist;
	};
} }
