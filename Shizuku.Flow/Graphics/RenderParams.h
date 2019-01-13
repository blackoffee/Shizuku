#pragma once

#include "Schema.h"
#include <glm/glm.hpp>

namespace Shizuku { namespace Flow{
	struct RenderParams
	{
		glm::mat4 ModelView;
		glm::mat4 Projection;
		glm::vec3 Camera;
		Schema Schema;
	};
} }
