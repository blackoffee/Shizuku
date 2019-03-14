#pragma once

#include "ObstDefinition.h"
#include "Pillar.h"
#include "RenderParams.h"
#include "HitParams.h"

#include "Shizuku.Core/Types/Point.h"
#include "Shizuku.Core/Ogl/Ogl.h"

using namespace Shizuku::Core;

namespace Shizuku { namespace Flow{
    class Obst
    {
    private:
        Pillar m_vis;
        ObstDefinition m_def;
        float m_height;

    public:
        Obst(std::shared_ptr<Shizuku::Core::Ogl> p_ogl, const ObstDefinition& p_def, const float p_height);

        const ObstDefinition& Def();
        void SetDef(const ObstDefinition& p_def);
        void SetHeight(const float p_height);
        void SetHighlight(const bool p_highlight);

        void Render(const RenderParams& p_params);
        HitResult Hit(const HitParams& p_params);

        //! Query if point on xy plane is inside obstruction
        HitResult Hit(const Types::Point<float>& p_modelSpace);
    };
} }
