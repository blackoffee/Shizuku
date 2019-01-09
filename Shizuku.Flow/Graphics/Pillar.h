#pragma once

#include "PillarDefinition.h"
#include "HitParams.h"
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
    public:
        Pillar(std::shared_ptr<Ogl> p_ogl);

        void Initialize();
        bool IsInitialized();
        void SetDefinition(const PillarDefinition& p_def);
        void SetPosition(const Types::Point<float>& p_pos);
        void SetSize(const Types::Box<float>& p_size);

		bool Hit(float& p_dist, const HitParams& p_params);

        void Draw(const glm::mat4& p_view, const glm::mat4& p_proj, const glm::vec3 p_cameraPos);
    };
} }