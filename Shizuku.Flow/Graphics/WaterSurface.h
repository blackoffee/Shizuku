#pragma once
#include "ShadingMode.h"
#include "Pillar.h"
#include "PillarDefinition.h"
#include "RenderParams.h"
#include "common.h"

#include "Shizuku.Core/Ogl/Ogl.h"
#include "Shizuku.Core/Rect.h"
#include "Shizuku.Core/Types/MinMax.h"
#include "Shizuku.Core/Types/Point.h"

#include "cuda_runtime.h"
#include <GLEW/glew.h>
#include "cuda_gl_interop.h"  // needs GLEW
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <memory>
#include <map>

class CudaLbm;
class Domain;

namespace Shizuku{
namespace Core{
    class ShaderProgram;
}
namespace Flow
{
	class Floor;
    class ObstDefinition;
}
}

using namespace Shizuku::Core;
using namespace Shizuku::Flow;

namespace Shizuku { namespace Flow {
	class WaterSurface
	{
	private:
		class Ssbo
		{
		public:
			GLuint m_id;
			std::string m_name;
		};
		std::shared_ptr<CudaLbm> m_cudaLbm;
		cudaGraphicsResource* m_cudaPosColorResource;
		cudaGraphicsResource* m_cudaNormalResource;
		cudaGraphicsResource* m_cudaEnvTextureResource;
		GLuint m_floorLightTexture;
		GLuint m_envTexture;
		GLuint m_poolFloorTexture;
		GLuint m_outputFbo;
		GLuint m_outputTexture;
		GLuint m_outputRbo;
		std::shared_ptr<ShaderProgram> m_surfaceRayTrace;
		std::shared_ptr<ShaderProgram> m_surfaceContour;
		std::shared_ptr<ShaderProgram> m_lightingProgram;
		std::shared_ptr<ShaderProgram> m_obstProgram;
		std::shared_ptr<ShaderProgram> m_outputProgram;
		std::shared_ptr<Ogl::Buffer> m_vbo;
		std::vector<Ssbo> m_ssbos;
		float m_omega;
		float m_inletVelocity;
		void CreateElementArrayBuffer();

		void RenderSurface(Domain &p_domain, const RenderParams& p_params, const Rect<int>& p_viewSize,
			const float obstHeight, const int obstCount, GLuint p_causticsTex);
		void RenderSurfaceContour(const ContourVariable p_contour, Domain &p_domain, const RenderParams& p_params);
		void RenderCameraPos(const RenderParams& p_params);

		std::shared_ptr<Pillar> m_cameraDatum;

	public:
		WaterSurface();

		std::shared_ptr<Ogl::Buffer> GetVbo();

		std::shared_ptr<Shizuku::Core::Ogl> Ogl;

		void CreateCudaLbm();
		std::shared_ptr<CudaLbm> GetCudaLbm();
		cudaGraphicsResource* GetCudaPosColorResource();
		cudaGraphicsResource* GetCudaNormalResource();
		cudaGraphicsResource* GetCudaEnvTextureResource();
		template <typename T> void CreateShaderStorageBuffer(T defaultValue,
			const unsigned int sizeInInts, const std::string name);
		GLuint GetShaderStorageBuffer(const std::string name);
		void CreateVboForCudaInterop();
		void CompileShaders();
		void AllocateStorageBuffers();
		void SetUpEnvironmentTexture();
		void SetUpOutputTexture(const Rect<int>& p_viewSize);
		void SetUpSurfaceVao();
		void SetUpOutputVao();
		void SetUpWallVao();
		void InitializeObstSsbo();
		void InitializeComputeShaderData();

		void BindFloorLightTexture();
		void BindEnvTexture();
		void UnbindFloorTexture();

		void SetOmega(const float omega);
		float GetOmega();
		void SetInletVelocity(const float u);
		float GetInletVelocity();
		void UpdateLbmInputs(const float u, const float omega);

		void RunComputeShader(const glm::vec3 p_cameraPosition, const ContourVariable p_contVar, const Types::MinMax<float>& p_minMax);
		void UpdateObstructionsUsingComputeShader(const int obstId, Shizuku::Flow::ObstDefinition &newObst, const float scaleFactor);
		int RayCastMouseClick(glm::vec3 &rayCastIntersection, const glm::vec3 rayOrigin,
			const glm::vec3 rayDir);

		void Render(const ContourVariable p_contour, Domain &domain, const RenderParams& p_params,
			const bool p_drawWireframe, const Rect<int>& p_viewSize, const float obstHeight, const int obstCount, GLuint p_causticsTex);

		void UpdateCameraDatum(const PillarDefinition& p_def);
	};
} }
