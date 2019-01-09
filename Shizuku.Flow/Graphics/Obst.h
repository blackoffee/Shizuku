#pragma once

#include "ObstDefinition.h"
#include "Pillar.h"

namespace Shizuku { namespace Flow{
    struct Obst
    {
		Pillar Vis;
		ObstDefinition Def;

		Obst(const Pillar& p_pillar, const ObstDefinition p_def) : Vis(p_pillar), Def(p_def)
		{
		}
    };
} }
