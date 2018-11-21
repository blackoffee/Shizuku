#pragma once
#include "common.h"
#include "ShadingMode.h"
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

#ifdef SHIZUKU_FLOW_EXPORTS  
#define FLOW_API __declspec(dllexport)   
#else  
#define FLOW_API __declspec(dllimport)   
#endif  

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

class FLOW_API ShaderManager
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
    GLuint m_floorFbo;
    GLuint m_outputFbo;
    GLuint m_outputTexture;
    GLuint m_outputRbo;
    std::shared_ptr<ShaderProgram> m_shaderProgram;
    std::shared_ptr<ShaderProgram> m_lightingProgram;
    std::shared_ptr<ShaderProgram> m_obstProgram;
    std::shared_ptr<ShaderProgram> m_floorProgram;
    std::shared_ptr<ShaderProgram> m_outputProgram;
    std::vector<Ssbo> m_ssbos;
    float m_omega;
    float m_inletVelocity;
    void CreateElementArrayBuffer();

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
    std::shared_ptr<ShaderProgram> GetFloorProgram();
    void CompileShaders();
    void AllocateStorageBuffers();
    void SetUpTextures(const Rect<int>& p_viewSize);
    void SetUpSurfaceVao();
    void SetUpOutputVao();
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

    void RunComputeShader(const glm::vec3 p_cameraPosition, const ContourVariable p_contVar, const MinMax<float>& p_minMax);
    void UpdateObstructionsUsingComputeShader(const int obstId, Obstruction &newObst, const float scaleFactor);
    int RayCastMouseClick(glm::vec3 &rayCastIntersection, const glm::vec3 rayOrigin,
        const glm::vec3 rayDir);
    void RenderFloorToTexture(Domain &domain, const Rect<int>& p_viewSize);
    void RenderSurface(const ShadingMode p_shadingMode , Domain &domain,
        const glm::mat4 &modelMatrix, const glm::mat4 &projectionMatrix);
};
