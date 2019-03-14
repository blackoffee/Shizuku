#pragma once

#include "Shizuku.Core/Types/Point.h"
#include "Shizuku.Core/Rect.h"

#include <boost/optional.hpp>

using namespace Shizuku::Core;

namespace Shizuku { namespace Flow{ namespace Info{
	struct ObstInfo
	{
		bool Selected;
		bool PreSelected;
		Types::Point<float> PositionInModel;
		Rect<float> Size;
	};
} } }
