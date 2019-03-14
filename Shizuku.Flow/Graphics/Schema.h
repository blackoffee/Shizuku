#pragma once

#include "Shizuku.Core/Types/Color.h"

using namespace Shizuku::Core;

namespace Shizuku { namespace Flow{
    struct Schema
    {
        Types::Color Background;
        Types::Color Obst;
        Types::Color ObstHighlight;
    };
} }
