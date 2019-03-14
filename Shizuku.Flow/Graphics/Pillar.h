#pragma once

#include "PillarDefinition.h"
#include "HitParams.h"
#include "RenderParams.h"
#include "Shizuku.Core/Types/Box.h"
#include "Shizuku.Core/Types/Point.h"
#include "Shizuku.Core/Rect.h"
#include <glm/glm.hpp>
#include <memory>

using namespace Shizuku::Core;

namespace Shizuku{
namespace Core{
    class Ogl;
    class ShaderProgram;
}
}

namespace Shizuku { namespace Flow{
    class Pillar
    {
    private:
        std::shared_ptr<Ogl> m_ogl;
        PillarDefinition m_def;
        std::shared_ptr<ShaderProgram> m_shaderProgram;
        void PrepareBuffers();
        void PrepareShader();
        bool m_initialized;
        bool m_highlighted;
    public:
        Pillar(std::shared_ptr<Ogl> p_ogl);

        const PillarDefinition& Def();

        void Initialize();
        bool IsInitialized();
        void SetDefinition(const PillarDefinition& p_def);
        void SetPosition(const Types::Point<float>& p_pos);
        void SetSize(const Types::Box<float>& p_size);
        void Highlight(const bool p_highlight);

        HitResult Hit(const HitParams& p_params);

        void Render(const RenderParams& p_params);
    };
} }