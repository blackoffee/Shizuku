#pragma once

#include "Shizuku.Core/Types/Point.h"
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
        Types::Point<float> m_position;
        Types::Point<float> m_size;
        std::shared_ptr<ShaderProgram> m_shaderProgram;
        void PrepareBuffers();
        void PrepareShader();
    public:
        Pillar(std::shared_ptr<Ogl> p_ogl);

        void Initialize();
        void SetPosition(const Types::Point<float>& p_pos);
        void SetSize(const Types::Point<float>& p_size);

        void Draw(const glm::mat4& p_model, const glm::mat4& p_proj);
    };
} }