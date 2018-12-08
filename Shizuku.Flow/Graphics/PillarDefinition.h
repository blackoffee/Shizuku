#pragma once

#include "Shizuku.Core/Rect.h"
#include "Shizuku.Core/Types/Point.h"

using namespace Shizuku::Core;

namespace Shizuku { namespace Flow{
    class PillarDefinition
    {
    private:
        Types::Point<float> m_position;
        Rect<float> m_size;
    public:
        PillarDefinition();
        PillarDefinition(const Types::Point<float>& p_pos, const Rect<float>& p_size);

        Types::Point<float>& Pos();
        Rect<float>& Size();
        void SetPosition(const Types::Point<float>& p_pos);
        void SetSize(const Rect<float>& p_size);
    };
} }