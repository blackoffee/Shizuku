#pragma once

namespace Shizuku { namespace Flow{
    enum Shape
    {
        SQUARE = 0,
        CIRCLE = 1,
        HORIZONTAL_LINE = 2,
        VERTICAL_LINE = 3 
    };

    enum State
    {
        ACTIVE = 0,
        INACTIVE = 1,
    };

    struct Obstruction
    {
        int shape;
        float x;
        float y;
        float r1;
        float r2;
        float u;
        float v;
        int state;
    };
} }