#pragma once

#include "HitParams.h"
#include "RenderParams.h"
#include "Domain.h"
#include "Shizuku.Core/Types/Box.h"
#include "Shizuku.Core/Types/Point.h"
#include "Shizuku.Core/Rect.h"
#include "Shizuku.Core/Ogl/Ogl.h"
#include <memory>

using namespace Shizuku::Core;

namespace Shizuku{
namespace Core{
    class ShaderProgram;
}
}

namespace Shizuku { namespace Flow{
    class Floor
    {
    private:
        std::shared_ptr<Ogl> m_ogl;
        std::shared_ptr<ShaderProgram> m_floorShader;
        std::shared_ptr<ShaderProgram> m_lightRayShader;
        std::shared_ptr<ShaderProgram> m_causticsShader;
		std::shared_ptr<Ogl::Buffer> m_vbo;
        bool m_initialized;
		GLuint m_causticsTex;
		GLuint m_floorTex;
		GLuint m_floorFbo;

		void CompileShaders();
		void PrepareIndices();
		void PrepareTextures();
		void PrepareVaos();

    public:
        Floor(std::shared_ptr<Ogl> p_ogl);

		void SetVbo(std::shared_ptr<Ogl::Buffer> p_vbo);
        void Initialize();
        bool IsInitialized();

		GLuint CausticsTex();

		HitResult Hit(const HitParams& p_params);

		void RenderCausticsToTexture(Domain &domain, const Rect<int>& p_viewSize);
        void Render(Domain &p_domain, const RenderParams& p_params, const bool p_drawWireframe);
    };
} }
