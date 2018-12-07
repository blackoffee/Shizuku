#pragma once
#include "common.h"
#include "ShadingMode.h"
#include "Pillar.h"
#include "Shizuku.Core/Rect.h"
#include "Shizuku.Core/Types/MinMax.h"
#include "cuda_runtime.h"
#include <GLEW/glew.h>
#include "cuda_gl_interop.h"  // needs GLEW
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <memory>

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

class CudaLbm;
class Domain;

namespace Shizuku{
namespace Core{
    class Ogl;
    class ShaderProgram;
}
}

using namespace Shizuku::Core;
using namespace Shizuku::Flow;

class ShaderManager
{
private:
    class Ssbo
    {
    public:
        GLuint m_id;
        std::string m_name;
    };
    std::shared_ptr<CudaLbm> m_cudaLbm;
    cudaGraphicsResource* m_cudaGraphicsResource;
    cudaGraphicsResource* m_cudaFloorLightTextureResource;
    cudaGraphicsResource* m_cudaEnvTextureResource;
    GLuint m_floorLightTexture;
    GLuint m_envTexture;
    GLuint m_poolFloorTexture;
    GLuint m_floorFbo;
    GLuint m_outputFbo;
    GLuint m_outputTexture;
    GLuint m_outputRbo;
    std::shared_ptr<ShaderProgram> m_shaderProgram;
    std::shared_ptr<ShaderProgram> m_lightingProgram;
    std::shared_ptr<ShaderProgram> m_obstProgram;
    std::shared_ptr<ShaderProgram> m_causticsProgram;
    std::shared_ptr<ShaderProgram> m_outputProgram;
    std::shared_ptr<ShaderProgram> m_floorProgram;
    std::vector<Ssbo> m_ssbos;
    float m_omega;
    float m_inletVelocity;
    void CreateElementArrayBuffer();

    void RenderFloor(const glm::mat4 &p_modelMatrix, const glm::mat4 &p_projectionMatrix);
    void RenderSurface(const ShadingMode p_shadingMode, Domain &p_domain,
    const glm::mat4 &p_modelMatrix, const glm::mat4 &p_projectionMatrix);

    std::shared_ptr<Pillar> m_pillar;

public:
    ShaderManager();

    std::shared_ptr<Shizuku::Core::Ogl> Ogl;

    void CreateCudaLbm();
    std::shared_ptr<CudaLbm> GetCudaLbm();
    cudaGraphicsResource* GetCudaSolutionGraphicsResource();
    cudaGraphicsResource* GetCudaFloorLightTextureResource();
    cudaGraphicsResource* GetCudaEnvTextureResource();
    template <typename T> void CreateShaderStorageBuffer(T defaultValue,
        const unsigned int sizeInInts, const std::string name);
    GLuint GetShaderStorageBuffer(const std::string name);
    void CreateVboForCudaInterop(unsigned int size);
    std::shared_ptr<ShaderProgram> GetShaderProgram();
    std::shared_ptr<ShaderProgram> GetLightingProgram();
    std::shared_ptr<ShaderProgram> GetObstProgram();
    std::shared_ptr<ShaderProgram> GetCausticsProgram();
    void CompileShaders();
    void AllocateStorageBuffers();
    void SetUpEnvironmentTexture();
    void SetUpFloorTexture();
    void SetUpCausticsTexture();
    void SetUpOutputTexture(const Rect<int>& p_viewSize);
    void SetUpSurfaceVao();
    void SetUpOutputVao();
    void SetUpWallVao();
    void SetUpFloorVao();
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
    void UpdateObstructionsUsingComputeShader(const int obstId, Obstruction &newObst, const float scaleFactor);
    int RayCastMouseClick(glm::vec3 &rayCastIntersection, const glm::vec3 rayOrigin,
        const glm::vec3 rayDir);

    void RenderCausticsToTexture(Domain &domain, const Rect<int>& p_viewSize);
    void Render(const ShadingMode p_shadingMode , Domain &domain,
        const glm::mat4 &modelMatrix, const glm::mat4 &projectionMatrix);
};
