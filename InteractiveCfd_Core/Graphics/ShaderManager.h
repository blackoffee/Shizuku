#pragma once
#include "common.h"
#include "cuda_runtime.h"
#include <GLEW/glew.h>
#include "cuda_gl_interop.h"  // needs GLEW
#include <glm/glm.hpp>
#include <vector>
#include <string>

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class CudaLbm;
class ShaderProgram;
class Domain;

class FW_API ShaderManager
{
private:
    class Ssbo
    {
    public:
        GLuint m_id;
        std::string m_name;
    };
    CudaLbm* m_cudaLbm;
    cudaGraphicsResource* m_cudaGraphicsResource;
    GLuint m_vao;
    GLuint m_vbo;
    GLuint m_elementArrayBuffer;
    ShaderProgram* m_shaderProgram;
    ShaderProgram* m_lightingProgram;
    ShaderProgram* m_obstProgram;
    std::vector<Ssbo> m_ssbos;
    float m_omega;
    float m_inletVelocity;
public:
    ShaderManager();

    void CreateCudaLbm();
    CudaLbm* GetCudaLbm();
    cudaGraphicsResource* GetCudaSolutionGraphicsResource();
    GLuint GetVbo();
    GLuint GetElementArrayBuffer();
    void CreateVbo(const unsigned int size, const unsigned int vboResFlags);
    void DeleteVbo();
    void CreateElementArrayBuffer();
    void DeleteElementArrayBuffer();
    template <typename T> void CreateShaderStorageBuffer(T defaultValue,
        const unsigned int sizeInInts, const std::string name);
    GLuint GetShaderStorageBuffer(const std::string name);
    void CreateVboForCudaInterop(unsigned int size);
    void CleanUpGLInterOp();
    ShaderProgram* GetShaderProgram();
    ShaderProgram* GetLightingProgram();
    ShaderProgram* GetObstProgram();
    void CompileShaders();
    void AllocateStorageBuffers();
    void InitializeObstSsbo();
    void InitializeComputeShaderData();
    
    void SetOmega(const float omega);
    float GetOmega();
    void SetInletVelocity(const float u);
    float GetInletVelocity();
    void UpdateLbmInputs(const float u, const float omega);
    void RunComputeShader(const float3 cameraPosition, const ContourVariable contVar,
        const float contMin, const float contMax);
    void UpdateObstructionsUsingComputeShader(const int obstId, Obstruction &newObst);
    int RayCastMouseClick(float3 &rayCastIntersection, const float3 rayOrigin,
        const float3 rayDir);
    void RenderVbo(const bool renderFloor, Domain &domain, const glm::mat4 &modelMatrix,
        const glm::mat4 &projectionMatrix);
    void RenderVboUsingShaders(const bool renderFloor, Domain &domain,
        const glm::mat4 &modelMatrix, const glm::mat4 &projectionMatrix);
};

void SetUniform(GLuint shaderId, const GLchar* varName, const int varValue);
void SetUniform(GLuint shaderId, const GLchar* varName, const float varValue);
void SetUniform(GLuint shaderId, const GLchar* varName, const bool varValue);
void SetUniform(GLuint shaderId, const GLchar* varName, const float3 varValue);
void RunSubroutine(GLuint shaderId, const GLchar* subroutineName,
    const int3 workGroupSize);
