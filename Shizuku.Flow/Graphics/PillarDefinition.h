#pragma once

#include "Shizuku.Core/Types/Box.h"
#include "Shizuku.Core/Types/Point.h"

using namespace Shizuku::Core;

namespace Shizuku { namespace Flow{
    class PillarDefinition
    {
    private:
        Types::Point<float> m_position;
        Types::Box<float> m_size;
    public:
        PillarDefinition();
        PillarDefinition(const Types::Point<float>& p_pos, const Types::Box<float>& p_size);

        Types::Point<float>& Pos();
        Types::Box<float>& Size();
        void SetPosition(const Types::Point<float>& p_pos);
        void SetSize(const Types::Box<float>& p_size);
    };
} }