#include "ShaderManager.h"
#include "GraphicsManager.h"
#include "Shizuku.Core/Ogl/Shader.h"
#include "Shizuku.Core/Ogl/Ogl.h"
#include "CudaLbm.h"
#include "Domain.h"
#include "helper_cuda.h"
#include <SOIL/SOIL.h>
#include <glm/gtc/type_ptr.hpp>
#include <GLEW/glew.h>
#include <string>
#include <cstring>
#include <iostream>
#include <assert.h>

using namespace Shizuku::Core;
using namespace Shizuku::Core::Types;
using namespace Shizuku::Flow;

ShaderManager::ShaderManager()
{
    m_shaderProgram = std::make_shared<ShaderProgram>();
    m_lightingProgram = std::make_shared<ShaderProgram>();
    m_obstProgram = std::make_shared<ShaderProgram>();
    m_causticsProgram = std::make_shared<ShaderProgram>();
    m_outputProgram = std::make_shared<ShaderProgram>();
    m_floorProgram = std::make_shared<ShaderProgram>();

    Ogl = std::make_shared < Shizuku::Core::Ogl >();
}

void ShaderManager::CreateCudaLbm()
{
    m_cudaLbm = std::make_shared<CudaLbm>();
}

std::shared_ptr<CudaLbm> ShaderManager::GetCudaLbm()
{
    return m_cudaLbm;
}

cudaGraphicsResource* ShaderManager::GetCudaSolutionGraphicsResource()
{
    return m_cudaGraphicsResource;
}

cudaGraphicsResource* ShaderManager::GetCudaFloorLightTextureResource()
{
    return m_cudaFloorLightTextureResource;
}

cudaGraphicsResource* ShaderManager::GetCudaEnvTextureResource()
{
    return m_cudaEnvTextureResource;
}

void ShaderManager::CreateVboForCudaInterop(const unsigned int size)
{
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
    std::shared_ptr<Ogl::Buffer> vbo = Ogl->CreateBuffer<float>(GL_ARRAY_BUFFER, 0, size, "surface", GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, vbo->GetId(), cudaGraphicsMapFlagsWriteDiscard);
    CreateElementArrayBuffer();
}

void ShaderManager::CreateElementArrayBuffer()
{
    const int numberOfElements = 2*(MAX_XDIM - 1)*(MAX_YDIM - 1);
    const int numberOfNodes = MAX_XDIM*MAX_YDIM;
    GLuint* elementIndices = new GLuint[numberOfElements * 3 * 2];
    for (int j = 0; j < MAX_YDIM-1; j++){
        for (int i = 0; i < MAX_XDIM-1; i++){
            //going clockwise, since y orientation will be flipped when rendered
            elementIndices[j*(MAX_XDIM-1)*6+i*6+0] = (i)+(j)*MAX_XDIM;
            elementIndices[j*(MAX_XDIM-1)*6+i*6+1] = (i+1)+(j)*MAX_XDIM;
            elementIndices[j*(MAX_XDIM-1)*6+i*6+2] = (i+1)+(j+1)*MAX_XDIM;

            elementIndices[j*(MAX_XDIM-1)*6+i*6+3] = (i)+(j)*MAX_XDIM;
            elementIndices[j*(MAX_XDIM-1)*6+i*6+4] = (i+1)+(j+1)*MAX_XDIM;
            elementIndices[j*(MAX_XDIM-1)*6+i*6+5] = (i)+(j+1)*MAX_XDIM;
        }
    }
    for (int j = 0; j < MAX_YDIM-1; j++){
        for (int i = 0; i < MAX_XDIM-1; i++){
            //going clockwise, since y orientation will be flipped when rendered
            elementIndices[numberOfElements*3+j*(MAX_XDIM-1)*6+i*6+0] = numberOfNodes+(i)+(j)*MAX_XDIM;
            elementIndices[numberOfElements*3+j*(MAX_XDIM-1)*6+i*6+1] = numberOfNodes+(i+1)+(j)*MAX_XDIM;
            elementIndices[numberOfElements*3+j*(MAX_XDIM-1)*6+i*6+2] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;

            elementIndices[numberOfElements*3+j*(MAX_XDIM-1)*6+i*6+3] = numberOfNodes+(i)+(j)*MAX_XDIM;
            elementIndices[numberOfElements*3+j*(MAX_XDIM-1)*6+i*6+4] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;
            elementIndices[numberOfElements*3+j*(MAX_XDIM-1)*6+i*6+5] = numberOfNodes+(i)+(j+1)*MAX_XDIM;
        }
    }

    Ogl->CreateBuffer(GL_ELEMENT_ARRAY_BUFFER, elementIndices, numberOfElements * 3 * 2, "surface_indices", GL_DYNAMIC_DRAW);
    free(elementIndices);
}

template <typename T>
void ShaderManager::CreateShaderStorageBuffer(T defaultValue, const unsigned int numberOfElements, const std::string name)
{
    T* data = new T[numberOfElements];
    for (int i = 0; i < numberOfElements; i++)
    {
        data[i] = defaultValue;
    }

    GLuint temp = Ogl->CreateBuffer(GL_SHADER_STORAGE_BUFFER, data, numberOfElements, name, GL_STATIC_DRAW)->GetId();

    m_ssbos.push_back(Ssbo{ temp, name });
}

GLuint ShaderManager::GetShaderStorageBuffer(const std::string name)
{
    for (std::vector<Ssbo>::iterator it = m_ssbos.begin(); it != m_ssbos.end(); ++it)
    {
        if (it->m_name == name)
        {
            return it->m_id;
        }
    }
    return NULL;
}

std::shared_ptr<ShaderProgram> ShaderManager::GetShaderProgram()
{
    return m_shaderProgram;
}

std::shared_ptr<ShaderProgram> ShaderManager::GetLightingProgram()
{
    return m_lightingProgram;
}

std::shared_ptr<ShaderProgram> ShaderManager::GetObstProgram()
{
    return m_obstProgram;
}

std::shared_ptr<ShaderProgram> ShaderManager::GetCausticsProgram()
{
    return m_causticsProgram;
}

void ShaderManager::CompileShaders()
{
    GetShaderProgram()->Initialize("Surface");
    GetShaderProgram()->CreateShader("SurfaceShader.vert.glsl", GL_VERTEX_SHADER);
    GetShaderProgram()->CreateShader("SurfaceShader.frag.glsl", GL_FRAGMENT_SHADER);
    GetLightingProgram()->Initialize("Lighting");
    GetLightingProgram()->CreateShader("SurfaceShader.comp.glsl", GL_COMPUTE_SHADER);
    GetObstProgram()->Initialize("Obstructions");
    GetObstProgram()->CreateShader("Obstructions.comp.glsl", GL_COMPUTE_SHADER);
    GetCausticsProgram()->Initialize("Caustics");
    GetCausticsProgram()->CreateShader("Caustics.vert.glsl", GL_VERTEX_SHADER);
    GetCausticsProgram()->CreateShader("Caustics.frag.glsl", GL_FRAGMENT_SHADER);
    m_outputProgram->Initialize("Output");
    m_outputProgram->CreateShader("Output.vert.glsl", GL_VERTEX_SHADER);
    m_outputProgram->CreateShader("Output.frag.glsl", GL_FRAGMENT_SHADER);
    m_floorProgram->Initialize("Floor");
    m_floorProgram->CreateShader("Floor.vert.glsl", GL_VERTEX_SHADER);
    m_floorProgram->CreateShader("Floor.frag.glsl", GL_FRAGMENT_SHADER);
}

void ShaderManager::AllocateStorageBuffers()
{

    CreateShaderStorageBuffer(GLfloat(0), MAX_XDIM*MAX_YDIM*9, "LbmA");
    CreateShaderStorageBuffer(GLfloat(0), MAX_XDIM*MAX_YDIM*9, "LbmB");
    CreateShaderStorageBuffer(GLint(0), MAX_XDIM*MAX_YDIM, "Floor");
    CreateShaderStorageBuffer(Obstruction{}, MAXOBSTS, "Obstructions");
    CreateShaderStorageBuffer(float4{0,0,0,1e6}, 1, "RayIntersection");
}

void ShaderManager::SetUpFloorTexture()
{
    int width, height;
    unsigned char* image = SOIL_load_image("Floor.png", &width, &height, 0, SOIL_LOAD_RGB);
    std::cout << SOIL_last_result() << std::endl;
    assert(image != NULL);
    float* tex = new float[4 * width*height];
    for (int i = 0; i < width*height; ++i)
    {
        tex[4 * i] = image[3 * i]/255.f;
        tex[4 * i + 1] = image[3 * i + 1]/255.f;
        tex[4 * i + 2] = image[3 * i + 2]/255.f;
        tex[4 * i + 3] = 1.f;// color[3];
    }

    glGenTextures(1, &m_poolFloorTexture);
    glBindTexture(GL_TEXTURE_2D, m_poolFloorTexture);

    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glGenerateMipmap(GL_TEXTURE_2D);

    SOIL_free_image_data(image);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void ShaderManager::SetUpEnvironmentTexture()
{
    int width, height;
    unsigned char* image = SOIL_load_image("Environment.png", &width, &height, 0, SOIL_LOAD_RGB);
    assert(image != NULL);
    float* tex = new float[4 * width*height];
    for (int i = 0; i < width*height; ++i)
    {
        tex[4 * i] = image[3 * i];
        tex[4 * i + 1] = image[3 * i + 1];
        tex[4 * i + 2] = image[3 * i + 2];
        tex[4 * i + 3] = unsigned char(255);// color[3];
    }

    glGenTextures(1, &m_envTexture);
    glBindTexture(GL_TEXTURE_2D, m_envTexture);

    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    cudaGraphicsGLRegisterImage(&m_cudaEnvTextureResource, m_envTexture, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);

    SOIL_free_image_data(image);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void ShaderManager::SetUpCausticsTexture()
{
    // set up FBO and texture to render to 
    glGenFramebuffers(1, &m_floorFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_floorFbo);

    glGenTextures(1, &m_floorLightTexture);
    glBindTexture(GL_TEXTURE_2D, m_floorLightTexture);

    const GLuint textureWidth = 1024;
    const GLuint textureHeight = 1024;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, textureWidth, textureHeight, 0, GL_RGBA, GL_FLOAT, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_floorLightTexture, 0);

    GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, drawBuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw "Framebuffer creation failed";

    cudaGraphicsGLRegisterImage(&m_cudaFloorLightTextureResource, m_floorLightTexture, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ShaderManager::SetUpOutputTexture(const Rect<int>& p_viewSize)
{
    //! Output Fbo
    glGenTextures(1, &m_outputTexture);
    glBindTexture(GL_TEXTURE_2D, m_outputTexture);

    const GLuint outWidth = p_viewSize.Width;
    const GLuint outHeight = p_viewSize.Height;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, outWidth, outHeight, 0, GL_RGBA, GL_FLOAT, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);

    glGenRenderbuffers(1, &m_outputRbo);
    glBindRenderbuffer(GL_RENDERBUFFER, m_outputRbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, outWidth, outHeight);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // set up FBO and texture to render to 
    glGenFramebuffers(1, &m_outputFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_outputFbo);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_outputTexture, 0);

    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_outputRbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw "Framebuffer creation failed";

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ShaderManager::InitializeObstSsbo()
{
    std::shared_ptr<CudaLbm> cudaLbm = GetCudaLbm();
    Obstruction* obst_h = cudaLbm->GetHostObst();

    std::shared_ptr<Ogl::Buffer> obstSsbo = Ogl->GetBuffer("Obstructions");
    Ogl->BindSSBO(0, *obstSsbo, GL_SHADER_STORAGE_BUFFER);
    glBufferData(GL_SHADER_STORAGE_BUFFER, MAXOBSTS*sizeof(Obstruction), obst_h, GL_STATIC_DRAW);
    Ogl->UnbindBO(GL_SHADER_STORAGE_BUFFER);
}

int ShaderManager::RayCastMouseClick(glm::vec3 &rayCastIntersection, const glm::vec3 rayOrigin,
    const glm::vec3 rayDir)
{
    Domain domain = *m_cudaLbm->GetDomain();
    int xDim = domain.GetXDim();
    int yDim = domain.GetYDim();
    glm::vec4 intersectionCoord{ 0, 0, 0, 0 };

    std::shared_ptr<Ogl::Buffer> rayIntSsbo = Ogl->GetBuffer("RayIntersection");
    Ogl->BindSSBO(4, *rayIntSsbo);

    std::shared_ptr<ShaderProgram> const shader = GetLightingProgram();
    shader->Use();

    shader->SetUniform("maxXDim", MAX_XDIM);
    shader->SetUniform("maxyDim", MAX_YDIM);
    shader->SetUniform("maxObsts", MAXOBSTS);
    shader->SetUniform("xDim", xDim);
    shader->SetUniform("yDim", yDim);
    shader->SetUniform("xDimVisible", domain.GetXDimVisible());
    shader->SetUniform("yDimVisible", domain.GetYDimVisible());
    shader->SetUniform("rayOrigin", rayOrigin);
    shader->SetUniform("rayDir", rayDir);

    shader->RunSubroutine("ResetRayCastData", glm::ivec3{ 1, 1, 1 });
    shader->RunSubroutine("RayCast", glm::ivec3{ xDim, yDim, 1 });

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, rayIntSsbo->GetId());
    GLfloat* intersect = (GLfloat*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
        sizeof(glm::vec4), GL_MAP_READ_BIT);
    std::memcpy(&intersectionCoord, intersect, sizeof(glm::vec4));
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    int returnVal;
    if (intersectionCoord.w > 1e5) //ray did not intersect with any objects
    {
        returnVal = 1;
    }
    else
    {
        shader->RunSubroutine("ResetRayCastData", glm::ivec3{ 1, 1, 1 });
        rayCastIntersection.x = intersectionCoord.x;
        rayCastIntersection.y = intersectionCoord.y;
        rayCastIntersection.z = intersectionCoord.z;
        returnVal = 0;
    }
    shader->Unset();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return returnVal;
}

void ShaderManager::SetUpSurfaceVao()
{
    std::shared_ptr<Ogl::Vao> surface = Ogl->CreateVao("surface");
    surface->Bind();

    Ogl->BindBO(GL_ARRAY_BUFFER, *Ogl->GetBuffer("surface"));
    Ogl->BindBO(GL_ELEMENT_ARRAY_BUFFER, *Ogl->GetBuffer("surface_indices"));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 16, (GLvoid*)(3 * sizeof(GLfloat)));

    surface->Unbind();
}

void ShaderManager::SetUpFloorVao()
{
    std::shared_ptr<Ogl::Vao> floor = Ogl->CreateVao("floor");
    floor->Bind();

    const GLfloat quadVertices[] = {
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
    };

    Ogl->CreateBuffer(GL_ARRAY_BUFFER, quadVertices, 18, "floor", GL_STATIC_DRAW);

    Ogl->BindBO(GL_ARRAY_BUFFER, *Ogl->GetBuffer("floor"));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    floor->Unbind();
}

void ShaderManager::SetUpOutputVao()
{
    std::shared_ptr<Ogl::Vao> output = Ogl->CreateVao("output");
    output->Bind();

    const GLfloat quadVertices[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f,  1.0f, 0.0f,
    };

    Ogl->CreateBuffer(GL_ARRAY_BUFFER, quadVertices, 18, "output", GL_STATIC_DRAW);

    Ogl->BindBO(GL_ARRAY_BUFFER, *Ogl->GetBuffer("output"));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    output->Unbind();
}

void ShaderManager::SetUpWallVao()
{
    std::shared_ptr<Ogl::Vao> wall = Ogl->CreateVao("wall");
    wall->Bind();

    const GLfloat quadVertices[] = {
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
    };

    Ogl->CreateBuffer(GL_ARRAY_BUFFER, quadVertices, 18, "wall", GL_STATIC_DRAW);

    Ogl->BindBO(GL_ARRAY_BUFFER, *Ogl->GetBuffer("wall"));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    wall->Unbind();
}

void ShaderManager::RenderCausticsToTexture(Domain &domain, const Rect<int>& p_viewSize)
{
    std::shared_ptr<Ogl::Vao> surface = Ogl->GetVao("surface");
    surface->Bind();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, m_floorFbo);
    glBindTexture(GL_TEXTURE_2D, m_poolFloorTexture);
    //glBindTexture(GL_TEXTURE_2D, m_floorLightTexture);

    std::shared_ptr<ShaderProgram> causticsShader = GetCausticsProgram();
    causticsShader->Use();

    causticsShader->SetUniform("texCoordScale", 3.f);
    glViewport(0, 0, 1024, 1024);

    int yDimVisible = domain.GetYDimVisible();
    //Draw floor
    glDrawElements(GL_TRIANGLES, (MAX_XDIM - 1)*(yDimVisible - 1)*3*2, GL_UNSIGNED_INT, 
        BUFFER_OFFSET(sizeof(GLuint)*3*2*(MAX_XDIM - 1)*(MAX_YDIM - 1)));

    causticsShader->Unset();

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    surface->Unbind();
    glViewport(0, 0, p_viewSize.Width, p_viewSize.Height);
}

void ShaderManager::RunComputeShader(const glm::vec3 p_cameraPosition, const ContourVariable p_contVar,
        const MinMax<float>& p_minMax)
{
    std::shared_ptr<Ogl::Buffer> ssbo_lbmA = Ogl->GetBuffer("LbmA");
    Ogl->BindSSBO(0, *ssbo_lbmA);
    std::shared_ptr<Ogl::Buffer> ssbo_lbmB = Ogl->GetBuffer("LbmB");
    Ogl->BindSSBO(1, *ssbo_lbmB);
    Ogl->BindBO(GL_SHADER_STORAGE_BUFFER, *Ogl->GetBuffer("surface"));
    std::shared_ptr<Ogl::Buffer> vbo = Ogl->GetBuffer("surface");
    Ogl->BindSSBO(2, *vbo);
    std::shared_ptr<Ogl::Buffer> ssbo_floor = Ogl->GetBuffer("Floor");
    Ogl->BindSSBO(3, *ssbo_floor);
    std::shared_ptr<Ogl::Buffer> ssbo_obsts = Ogl->GetBuffer("Obstructions");
    Ogl->BindSSBO(5, *ssbo_obsts);
    std::shared_ptr<ShaderProgram> const shader = GetLightingProgram();

    shader->Use();

    Domain domain = *m_cudaLbm->GetDomain();
    const int xDim = domain.GetXDim();
    const int yDim = domain.GetYDim();
    shader->SetUniform("maxXDim", MAX_XDIM);
    shader->SetUniform("maxyDim", MAX_YDIM);
    shader->SetUniform("maxObsts", MAXOBSTS);
    shader->SetUniform("xDim", xDim);
    shader->SetUniform("yDim", yDim);
    shader->SetUniform("xDimVisible", domain.GetXDimVisible());
    shader->SetUniform("yDimVisible", domain.GetYDimVisible());
    shader->SetUniform("cameraPosition", p_cameraPosition);
    shader->SetUniform("uMax", m_inletVelocity);
    shader->SetUniform("omega", m_omega);
    shader->SetUniform("contourVar", p_contVar);
    shader->SetUniform("contourMin", p_minMax.Min);
    shader->SetUniform("contourMax", p_minMax.Max);

    for (int i = 0; i < 5; i++)
    {
        shader->RunSubroutine("MarchLbm", glm::ivec3{ xDim, yDim, 1 });
        Ogl->BindSSBO(1, *ssbo_lbmA);
        Ogl->BindSSBO(0, *ssbo_lbmB);

        shader->RunSubroutine("MarchLbm", glm::ivec3{ xDim, yDim, 1 });
        Ogl->BindSSBO(0, *ssbo_lbmA);
        Ogl->BindSSBO(1, *ssbo_lbmB);
    }
    
    shader->RunSubroutine("UpdateFluidVbo", glm::ivec3{ xDim, yDim, 1 });
    shader->RunSubroutine("DeformFloorMeshUsingCausticRay", glm::ivec3{ xDim, yDim, 1 });
    shader->RunSubroutine("ComputeFloorLightIntensitiesFromMeshDeformation", glm::ivec3{ xDim, yDim, 1 });
    shader->RunSubroutine("ApplyCausticLightingToFloor", glm::ivec3{ xDim, yDim, 1 });
    shader->RunSubroutine("PhongLighting", glm::ivec3{ xDim, yDim, 2 });
    shader->RunSubroutine("UpdateObstructionTransientStates", glm::ivec3{ xDim, yDim, 1 });
    shader->RunSubroutine("CleanUpVbo", glm::ivec3{ MAX_XDIM, MAX_YDIM, 2 });
    
    shader->Unset();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ShaderManager::UpdateObstructionsUsingComputeShader(const int obstId, Obstruction &newObst, const float scaleFactor)
{
    std::shared_ptr<Ogl::Buffer> ssbo_obsts = Ogl->GetBuffer("Obstructions");
    Ogl->BindSSBO(0, *ssbo_obsts);
    std::shared_ptr<ShaderProgram> const shader = GetObstProgram();
    shader->Use();

    shader->SetUniform("targetObst.shape", newObst.shape);
    shader->SetUniform("targetObst.x", newObst.x);
    shader->SetUniform("targetObst.y", newObst.y);
    shader->SetUniform("targetObst.r1", newObst.r1);
    shader->SetUniform("targetObst.u", newObst.u);
    shader->SetUniform("targetObst.v", newObst.v);
    shader->SetUniform("targetObst.state", newObst.state);
    shader->SetUniform("targetObstId", obstId);

    shader->RunSubroutine("UpdateObstruction", glm::ivec3{ 1, 1, 1 });

    shader->Unset();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ShaderManager::InitializeComputeShaderData()
{

    std::shared_ptr<Ogl::Buffer> ssbo_lbmA = Ogl->GetBuffer("LbmA");
    Ogl->BindSSBO(0, *ssbo_lbmA);
    std::shared_ptr<Ogl::Buffer> ssbo_lbmB = Ogl->GetBuffer("LbmB");
    Ogl->BindSSBO(1, *ssbo_lbmB);
    std::shared_ptr<Ogl::Buffer> ssbo_obsts = Ogl->GetBuffer("Obstructions");
    Ogl->BindSSBO(5, *ssbo_obsts);

    std::shared_ptr<ShaderProgram> const shader = GetLightingProgram();

    shader->Use();

    Domain domain = *m_cudaLbm->GetDomain();
    shader->SetUniform("maxXDim", MAX_XDIM);
    shader->SetUniform("maxYDim", MAX_YDIM);
    shader->SetUniform("maxObsts", MAXOBSTS);//
    shader->SetUniform("xDim", domain.GetXDim());
    shader->SetUniform("yDim", domain.GetYDim());
    shader->SetUniform("xDimVisible", domain.GetXDim());
    shader->SetUniform("yDimVisible", domain.GetYDim());
    shader->SetUniform("uMax", m_inletVelocity);

    shader->RunSubroutine("InitializeDomain", glm::ivec3{ MAX_XDIM, MAX_YDIM, 1 });

    shader->Unset();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ShaderManager::BindFloorLightTexture()
{
    glBindTexture(GL_TEXTURE_2D, m_floorLightTexture);
}

void ShaderManager::BindEnvTexture()
{
    glBindTexture(GL_TEXTURE_2D, m_envTexture);
}

void ShaderManager::UnbindFloorTexture()
{
    glBindTexture(GL_TEXTURE_2D, 0);
}

void ShaderManager::SetOmega(const float omega)
{
    m_omega = omega;
}

float ShaderManager::GetOmega()
{
    return m_omega;
}

void ShaderManager::SetInletVelocity(const float u)
{
    m_inletVelocity = u;
}

float ShaderManager::GetInletVelocity()
{
    return m_inletVelocity;
}

void ShaderManager::UpdateLbmInputs(const float u, const float omega)
{
    SetInletVelocity(u);
    SetOmega(omega);
}

void ShaderManager::Render(const ShadingMode p_shadingMode, Domain &p_domain,
    const glm::mat4 &p_modelMatrix, const glm::mat4 &p_projectionMatrix)
{
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    //! Disabling for now. Need to recreate correct size textures on window resize
    const bool offscreenRender = false;

    if (offscreenRender)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, m_outputFbo);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    RenderFloor(p_modelMatrix, p_projectionMatrix);

    RenderSurface(p_shadingMode, p_domain, p_modelMatrix, p_projectionMatrix);

    if (offscreenRender)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        //Draw quad
        std::shared_ptr<Ogl::Vao> outputVao = Ogl->GetVao("output");
        outputVao->Bind();

        m_outputProgram->Use();
        glBindTexture(GL_TEXTURE_2D, m_outputTexture);

        glDrawArrays(GL_TRIANGLES, 0 , 6);

        glBindTexture(GL_TEXTURE_2D, 0);
        m_outputProgram->Unset();

        outputVao->Unbind();
    }
}

void ShaderManager::RenderFloor(const glm::mat4 &p_modelMatrix, const glm::mat4 &p_projectionMatrix)
{
    std::shared_ptr<ShaderProgram> floorShader = m_floorProgram;
    floorShader->Use();
    glActiveTexture(GL_TEXTURE0);
    floorShader->SetUniform("modelMatrix", glm::transpose(p_modelMatrix));
    floorShader->SetUniform("projectionMatrix", glm::transpose(p_projectionMatrix));

    std::shared_ptr<Ogl::Vao> floor = Ogl->GetVao("floor");
    floor->Bind();
    glBindTexture(GL_TEXTURE_2D, m_floorLightTexture);

    glDrawArrays(GL_TRIANGLES, 0 , 6);

    floor->Unbind();
    glBindTexture(GL_TEXTURE_2D, 0);
    floorShader->Unset();

//    glDrawElements(GL_TRIANGLES, (MAX_XDIM - 1)*(yDimVisible - 1)*3*2, GL_UNSIGNED_INT, 
//        BUFFER_OFFSET(sizeof(GLuint)*3*2*(MAX_XDIM - 1)*(MAX_YDIM - 1)));
}

void ShaderManager::RenderSurface(const ShadingMode p_shadingMode, Domain &domain,
    const glm::mat4 &modelMatrix, const glm::mat4 &projectionMatrix)
{
    std::shared_ptr<ShaderProgram> shader = GetShaderProgram();
    shader->Use();
    glActiveTexture(GL_TEXTURE0);
    shader->SetUniform("modelMatrix", glm::transpose(modelMatrix));
    shader->SetUniform("projectionMatrix", glm::transpose(projectionMatrix));

    std::shared_ptr<Ogl::Vao> surface = Ogl->GetVao("surface");
    surface->Bind();

    if (p_shadingMode != ShadingMode::RayTracing && p_shadingMode != ShadingMode::SimplifiedRayTracing)
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    const int yDimVisible = domain.GetYDimVisible();
    glDrawElements(GL_TRIANGLES, (MAX_XDIM - 1)*(yDimVisible - 1)*3*2 , GL_UNSIGNED_INT, (GLvoid*)0);
    surface->Unbind();

    shader->Unset();   
}