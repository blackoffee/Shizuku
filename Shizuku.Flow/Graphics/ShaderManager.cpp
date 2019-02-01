#include "ShaderManager.h"
#include "GraphicsManager.h"
#include "Floor.h"
#include "ObstDefinition.h"
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
    m_surfaceRayTrace = std::make_shared<ShaderProgram>();
    m_surfaceContour = std::make_shared<ShaderProgram>();
    m_lightingProgram = std::make_shared<ShaderProgram>();
    m_obstProgram = std::make_shared<ShaderProgram>();
    //m_causticsProgram = std::make_shared<ShaderProgram>();
    m_outputProgram = std::make_shared<ShaderProgram>();
    //m_floorProgram = std::make_shared<ShaderProgram>();
    //m_lightRayProgram = std::make_shared<ShaderProgram>();

    Ogl = std::make_shared < Shizuku::Core::Ogl >();

    //m_pillars = std::map<const int, std::shared_ptr<Pillar>>();
    m_cameraDatum = std::make_shared<Pillar>(Ogl);

	m_floor = std::make_shared<Floor>(Ogl);
}

void ShaderManager::CreateCudaLbm()
{
    m_cudaLbm = std::make_shared<CudaLbm>();
}

std::shared_ptr<CudaLbm> ShaderManager::GetCudaLbm()
{
    return m_cudaLbm;
}

cudaGraphicsResource* ShaderManager::GetCudaPosColorResource()
{
    return m_cudaPosColorResource;
}

cudaGraphicsResource* ShaderManager::GetCudaNormalResource()
{
    return m_cudaNormalResource;
}

cudaGraphicsResource* ShaderManager::GetCudaFloorLightTextureResource()
{
    return m_cudaFloorLightTextureResource;
}

cudaGraphicsResource* ShaderManager::GetCudaEnvTextureResource()
{
    return m_cudaEnvTextureResource;
}


void ShaderManager::CreateVboForCudaInterop()
{
    unsigned int solutionMemorySize = MAX_XDIM*MAX_YDIM * 4 * sizeof(float);
    unsigned int floorSize = MAX_XDIM*MAX_YDIM * 4 * sizeof(float);
    const unsigned int size = solutionMemorySize + floorSize;
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
    std::shared_ptr<Ogl::Buffer> posColor = Ogl->CreateBuffer<float>(GL_ARRAY_BUFFER, 0, size, "surface", GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&m_cudaPosColorResource, posColor->GetId(), cudaGraphicsMapFlagsWriteDiscard);
    std::shared_ptr<Ogl::Buffer> normals = Ogl->CreateBuffer<float>(GL_ARRAY_BUFFER, 0, size, "surface_normals", GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&m_cudaNormalResource, normals->GetId(), cudaGraphicsMapFlagsWriteDiscard);
    CreateElementArrayBuffer();

	m_floor->SetVbo(posColor);
}

void ShaderManager::CreateElementArrayBuffer()
{
    const int numberOfElements = (MAX_XDIM - 1)*(MAX_YDIM - 1);
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
//    for (int j = 0; j < MAX_YDIM-1; j++){
//        for (int i = 0; i < MAX_XDIM-1; i++){
//            //going clockwise, since y orientation will be flipped when rendered
//            elementIndices[numberOfElements*3+j*(MAX_XDIM-1)*6+i*6+0] = numberOfNodes+(i)+(j)*MAX_XDIM;
//            elementIndices[numberOfElements*3+j*(MAX_XDIM-1)*6+i*6+1] = numberOfNodes+(i+1)+(j)*MAX_XDIM;
//            elementIndices[numberOfElements*3+j*(MAX_XDIM-1)*6+i*6+2] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;
//
//            elementIndices[numberOfElements*3+j*(MAX_XDIM-1)*6+i*6+3] = numberOfNodes+(i)+(j)*MAX_XDIM;
//            elementIndices[numberOfElements*3+j*(MAX_XDIM-1)*6+i*6+4] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;
//            elementIndices[numberOfElements*3+j*(MAX_XDIM-1)*6+i*6+5] = numberOfNodes+(i)+(j+1)*MAX_XDIM;
//        }
//    }
//
//    GLuint* elemEdgeIndices = new GLuint[numberOfElements * 12];
//
//    for (int j = 0; j < MAX_YDIM-1; j++){
//        for (int i = 0; i < MAX_XDIM-1; i++){
//            //going clockwise, since y orientation will be flipped when rendered
//            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+0] = numberOfNodes+(i)+(j)*MAX_XDIM;
//            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+1] = numberOfNodes+(i+1)+(j)*MAX_XDIM;
//            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+2] = numberOfNodes+(i+1)+(j)*MAX_XDIM;
//            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+3] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;
//            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+4] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;
//            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+5] = numberOfNodes+(i)+(j)*MAX_XDIM;
//
//            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+6] = numberOfNodes+(i)+(j)*MAX_XDIM;
//            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+7] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;
//            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+8] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;
//            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+9] = numberOfNodes+(i)+(j+1)*MAX_XDIM;
//            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+10] = numberOfNodes+(i)+(j+1)*MAX_XDIM;
//            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+11] = numberOfNodes+(i)+(j)*MAX_XDIM;
//        }
//    }
//
//	GLuint* lightPathIndices = new GLuint[numberOfNodes * 2];
//    for (int j = 0; j < MAX_YDIM; j++){
//        for (int i = 0; i < MAX_XDIM; i++){
//            //going clockwise, since y orientation will be flipped when rendered
//            lightPathIndices[j*MAX_XDIM*2+i*2+0] = i+j*MAX_XDIM;
//            lightPathIndices[j*MAX_XDIM*2+i*2+1] = numberOfNodes+i+j*MAX_XDIM;
//        }
//    }

    Ogl->CreateBuffer(GL_ELEMENT_ARRAY_BUFFER, elementIndices, numberOfElements * 3 * 2, "surface_indices", GL_DYNAMIC_DRAW);
    free(elementIndices);
//    Ogl->CreateBuffer(GL_ELEMENT_ARRAY_BUFFER, elemEdgeIndices, numberOfElements * 6 * 2, "surfaceMesh_indices", GL_DYNAMIC_DRAW);
//    free(elemEdgeIndices);
//    Ogl->CreateBuffer(GL_ELEMENT_ARRAY_BUFFER, lightPathIndices, numberOfNodes * 2, "lightPaths_indices", GL_DYNAMIC_DRAW);
//    free(lightPathIndices);
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

void ShaderManager::CompileShaders()
{
    m_surfaceRayTrace->Initialize("SurfaceRayTrace");
    m_surfaceRayTrace->CreateShader("Assets/SurfaceShader.vert.glsl", GL_VERTEX_SHADER);
    m_surfaceRayTrace->CreateShader("Assets/SurfaceShader.frag.glsl", GL_FRAGMENT_SHADER);
    m_surfaceContour->Initialize("SurfaceContour");
    m_surfaceContour->CreateShader("Assets/SurfaceContour.vert.glsl", GL_VERTEX_SHADER);
    m_surfaceContour->CreateShader("Assets/SurfaceContour.frag.glsl", GL_FRAGMENT_SHADER);
    m_lightingProgram->Initialize("Lighting");
    m_lightingProgram->CreateShader("Assets/SurfaceShader.comp.glsl", GL_COMPUTE_SHADER);
    m_obstProgram->Initialize("Obstructions");
    m_obstProgram->CreateShader("Assets/Obstructions.comp.glsl", GL_COMPUTE_SHADER);
//    m_causticsProgram->Initialize("Caustics");
//    m_causticsProgram->CreateShader("Assets/Caustics.vert.glsl", GL_VERTEX_SHADER);
//    m_causticsProgram->CreateShader("Assets/Caustics.frag.glsl", GL_FRAGMENT_SHADER);
    m_outputProgram->Initialize("Output");
    m_outputProgram->CreateShader("Assets/Output.vert.glsl", GL_VERTEX_SHADER);
    m_outputProgram->CreateShader("Assets/Output.frag.glsl", GL_FRAGMENT_SHADER);
//    m_floorProgram->Initialize("Floor");
//    m_floorProgram->CreateShader("Assets/Floor.vert.glsl", GL_VERTEX_SHADER);
//    m_floorProgram->CreateShader("Assets/Floor.frag.glsl", GL_FRAGMENT_SHADER);
//    m_lightRayProgram->Initialize("LightRay");
//    m_lightRayProgram->CreateShader("Assets/LightRay.vert.glsl", GL_VERTEX_SHADER);
//    m_lightRayProgram->CreateShader("Assets/LightRay.geom.glsl", GL_GEOMETRY_SHADER);
//    m_lightRayProgram->CreateShader("Assets/LightRay.frag.glsl", GL_FRAGMENT_SHADER);
}

void ShaderManager::AllocateStorageBuffers()
{

    CreateShaderStorageBuffer(GLfloat(0), MAX_XDIM*MAX_YDIM*9, "LbmA");
    CreateShaderStorageBuffer(GLfloat(0), MAX_XDIM*MAX_YDIM*9, "LbmB");
    CreateShaderStorageBuffer(GLint(0), MAX_XDIM*MAX_YDIM, "Floor");
    CreateShaderStorageBuffer(ObstDefinition{}, MAXOBSTS, "Obstructions");
    CreateShaderStorageBuffer(float4{0,0,0,1e6}, 1, "RayIntersection");
}

void ShaderManager::SetUpFloorTexture()
{
//    int width, height;
//    unsigned char* image = SOIL_load_image("Assets/Floor.png", &width, &height, 0, SOIL_LOAD_RGB);
//    //std::cout << SOIL_last_result() << std::endl;
//    assert(image != NULL);
//    float* tex = new float[4 * width*height];
//    for (int i = 0; i < width*height; ++i)
//    {
//        tex[4 * i] = image[3 * i]/255.f;
//        tex[4 * i + 1] = image[3 * i + 1]/255.f;
//        tex[4 * i + 2] = image[3 * i + 2]/255.f;
//        tex[4 * i + 3] = 1.f;// color[3];
//    }
//
//    glGenTextures(1, &m_poolFloorTexture);
//    glBindTexture(GL_TEXTURE_2D, m_poolFloorTexture);
//
//    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex);
//    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, tex);
//
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
//
//    glGenerateMipmap(GL_TEXTURE_2D);
//
//    SOIL_free_image_data(image);
//
//    glBindTexture(GL_TEXTURE_2D, 0);
}

void ShaderManager::SetUpEnvironmentTexture()
{
    int width, height;
    unsigned char* image = SOIL_load_image("Assets/Environment.png", &width, &height, 0, SOIL_LOAD_RGB);
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
	m_floor->Initialize();
//    // set up FBO and texture to render to 
//    glGenFramebuffers(1, &m_floorFbo);
//    glBindFramebuffer(GL_FRAMEBUFFER, m_floorFbo);
//
//    glGenTextures(1, &m_floorLightTexture);
//    glBindTexture(GL_TEXTURE_2D, m_floorLightTexture);
//
//    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, CAUSTICS_TEX_SIZE, CAUSTICS_TEX_SIZE, 0, GL_RGBA, GL_FLOAT, 0);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
//    const GLfloat color[4] = { 0, 0, 0, 0 };
//    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color);
//
//    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_floorLightTexture, 0);
//
//    GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
//    glDrawBuffers(1, drawBuffers);
//
//    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
//        throw "Framebuffer creation failed";

    cudaGraphicsGLRegisterImage(&m_cudaFloorLightTextureResource, m_floor->CausticsTex(), GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);

//    glBindTexture(GL_TEXTURE_2D, 0);
//    glBindFramebuffer(GL_FRAMEBUFFER, 0);
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
    ObstDefinition* obst_h = cudaLbm->GetHostObst();

    std::shared_ptr<Ogl::Buffer> obstSsbo = Ogl->GetBuffer("Obstructions");
    Ogl->BindSSBO(0, *obstSsbo, GL_SHADER_STORAGE_BUFFER);
    glBufferData(GL_SHADER_STORAGE_BUFFER, MAXOBSTS*sizeof(ObstDefinition), obst_h, GL_STATIC_DRAW);
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

    std::shared_ptr<ShaderProgram> const shader = m_lightingProgram;
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

    Ogl->BindBO(GL_ARRAY_BUFFER, *Ogl->GetBuffer("surface_normals"));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 16, 0);

    surface->Unbind();

//    std::shared_ptr<Ogl::Vao> surfaceMesh = Ogl->CreateVao("surfaceMesh");
//    surfaceMesh->Bind();
//
//    Ogl->BindBO(GL_ARRAY_BUFFER, *Ogl->GetBuffer("surface"));
//    Ogl->BindBO(GL_ELEMENT_ARRAY_BUFFER, *Ogl->GetBuffer("surfaceMesh_indices"));
//
//    glEnableVertexAttribArray(0);
//    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, 0);
//    glEnableVertexAttribArray(1);
//    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 16, (GLvoid*)(3 * sizeof(GLfloat)));
//
//    surfaceMesh->Unbind();
//
//    std::shared_ptr<Ogl::Vao> lightPaths = Ogl->CreateVao("lightPaths");
//    lightPaths->Bind();
//
//    Ogl->BindBO(GL_ARRAY_BUFFER, *Ogl->GetBuffer("surface"));
//    Ogl->BindBO(GL_ELEMENT_ARRAY_BUFFER, *Ogl->GetBuffer("lightPaths_indices"));
//
//    glEnableVertexAttribArray(0);
//    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, 0);
//    glEnableVertexAttribArray(1);
//    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 16, (GLvoid*)(3 * sizeof(GLfloat)));
//
//    lightPaths->Unbind();
}

void ShaderManager::SetUpFloorVao()
{
//    std::shared_ptr<Ogl::Vao> floor = Ogl->CreateVao("floor");
//    floor->Bind();
//
//    const GLfloat quadVertices[] = {
//        -1.0f, -1.0f, -1.0f,
//         1.0f, -1.0f, -1.0f,
//        -1.0f,  1.0f, -1.0f,
//        -1.0f,  1.0f, -1.0f,
//         1.0f, -1.0f, -1.0f,
//         1.0f,  1.0f, -1.0f,
//    };
//
//    Ogl->CreateBuffer(GL_ARRAY_BUFFER, quadVertices, 18, "floor", GL_STATIC_DRAW);
//
//    Ogl->BindBO(GL_ARRAY_BUFFER, *Ogl->GetBuffer("floor"));
//
//    glEnableVertexAttribArray(0);
//    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
//
//    floor->Unbind();
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
	m_floor->RenderCausticsToTexture(domain, p_viewSize);
//    std::shared_ptr<Ogl::Vao> surface = Ogl->GetVao("surface");
//    surface->Bind();
//
//    glBindFramebuffer(GL_FRAMEBUFFER, m_floorFbo);
//    glBindTexture(GL_TEXTURE_2D, m_poolFloorTexture);
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//    m_causticsProgram->Use();
//
//    m_causticsProgram->SetUniform("texCoordScale", 3.f);
//    glViewport(0, 0, CAUSTICS_TEX_SIZE, CAUSTICS_TEX_SIZE);
//    glEnable(GL_BLEND);
//    glBlendFunc(GL_ONE, GL_ONE);
//    glBlendEquation(GL_FUNC_ADD);
//
//    //Draw floor
//    const int xDimVisible = domain.GetXDimVisible();
//    const int yDimVisible = domain.GetYDimVisible();
//    const int offsetToFloorIndices = 3 * 2 * (MAX_XDIM - 1)*(MAX_YDIM - 1);
//
//	glDrawElements(GL_TRIANGLES, 3 * (2 * (MAX_XDIM - 1))*(yDimVisible-1), GL_UNSIGNED_INT,
//		BUFFER_OFFSET(sizeof(GLuint)*(offsetToFloorIndices)));
//
//    m_causticsProgram->Unset();
//
//    glBindTexture(GL_TEXTURE_2D, 0);
//    glBindFramebuffer(GL_FRAMEBUFFER, 0);
//
//    glDisable(GL_BLEND);
//
//    surface->Unbind();
//    glViewport(0, 0, p_viewSize.Width, p_viewSize.Height);
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
    std::shared_ptr<ShaderProgram> const shader = m_lightingProgram;

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

void ShaderManager::UpdateObstructionsUsingComputeShader(const int obstId, ObstDefinition &newObst, const float scaleFactor)
{
    std::shared_ptr<Ogl::Buffer> ssbo_obsts = Ogl->GetBuffer("Obstructions");
    Ogl->BindSSBO(0, *ssbo_obsts);
    std::shared_ptr<ShaderProgram> const shader = m_obstProgram;
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

    std::shared_ptr<ShaderProgram> const shader = m_lightingProgram;

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


void ShaderManager::Render(const ContourVariable p_contour , Domain &p_domain, const RenderParams& p_params,
        const bool p_drawFloorWireframe, const Rect<int>& p_viewSize, const float p_obstHeight, const int obstCount)
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

    RenderFloor(p_domain, p_params, p_drawFloorWireframe);

	if (p_drawFloorWireframe)
		m_floor->RenderCausticsMesh(p_domain, p_params);

	if (p_contour == ContourVariable::WATER_RENDERING)
		RenderSurface(p_domain, p_params, p_viewSize, p_obstHeight, obstCount);
	else
		RenderSurfaceContour(p_contour, p_domain, p_params);


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

void ShaderManager::RenderFloor(Domain &p_domain, const RenderParams& p_params, const bool p_drawWireframe)
{
	m_floor->Render(p_domain, p_params);

//    std::shared_ptr<ShaderProgram> floorShader = m_floorProgram;
//    floorShader->Use();
//    glActiveTexture(GL_TEXTURE0);
//    floorShader->SetUniform("modelMatrix", p_params.ModelView);
//    floorShader->SetUniform("projectionMatrix", p_params.Projection);
//
//    std::shared_ptr<Ogl::Vao> floor = Ogl->GetVao("floor");
//    floor->Bind();
//    glBindTexture(GL_TEXTURE_2D, m_floorLightTexture);
//
//    glDrawArrays(GL_TRIANGLES, 0 , 6);
//
//    floor->Unbind();
//    glBindTexture(GL_TEXTURE_2D, 0);
//    floorShader->Unset();
//
//    if (p_drawWireframe)
//    {
//		m_lightRayProgram->Use();
//	    m_lightRayProgram->SetUniform("modelMatrix", p_params.ModelView);
//	    m_lightRayProgram->SetUniform("projectionMatrix", p_params.Projection);
//	    m_lightRayProgram->SetUniform("Filter", false);
//        std::shared_ptr<Ogl::Vao> surface = Ogl->GetVao("surfaceMesh");
//        surface->Bind();
//
//        const int xDimVisible = p_domain.GetXDimVisible();
//        const int yDimVisible = p_domain.GetYDimVisible();
//        for (int i = 0; i < yDimVisible - 1; ++i)
//        {
//            glDrawElements(GL_LINES, (xDimVisible - 1) * 12, GL_UNSIGNED_INT,
//                BUFFER_OFFSET(sizeof(GLuint)*(MAX_XDIM - 1)*i * 12));
//        }
//
//        surface->Unbind();
//
//        std::shared_ptr<Ogl::Vao> paths = Ogl->GetVao("lightPaths");
//        paths->Bind();
//
//	    m_lightRayProgram->SetUniform("Filter", true);
//		glDisable(GL_DEPTH_TEST);
//		for (int j = 0; j < yDimVisible; ++j)
//		{
//			glDrawElements(GL_LINES, xDimVisible * 2, GL_UNSIGNED_INT,
//				BUFFER_OFFSET(sizeof(GLuint)*(MAX_XDIM*j) * 2));
//		}
//
//
//
//
////		Rect<int> area(xDimVisible / 5.f, yDimVisible / 5.f);
////		Types::Point<int> pos(xDimVisible / 2.f, yDimVisible / 3.f);
////		glDisable(GL_DEPTH_TEST);
////        for (int j = pos.Y-area.Height; j < pos.Y+area.Height; ++j)
////        {
////			for (int i = pos.X-area.Width; i < pos.X+area.Width; ++i)
////			{
////				glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT,
////					BUFFER_OFFSET(sizeof(GLuint)*(MAX_XDIM*i+j) * 2));
////			}
////        }
////		glEnable(GL_DEPTH_TEST);
//
//        paths->Unbind();
//		m_lightRayProgram->Unset();
//    }

}

void ShaderManager::RenderSurface(Domain &domain, const RenderParams& p_params, const Rect<int>& p_viewSize,
	const float p_obstHeight, const int p_obstCount)
{
	glActiveTexture(GL_TEXTURE0);
	m_surfaceRayTrace->Use();
	m_surfaceRayTrace->SetUniform("modelMatrix", p_params.ModelView);
	m_surfaceRayTrace->SetUniform("projectionMatrix", p_params.Projection);
	m_surfaceRayTrace->SetUniform("cameraPos", p_params.Camera);
	m_surfaceRayTrace->SetUniform("obstHeight", p_obstHeight);
	m_surfaceRayTrace->SetUniform("obstCount", p_obstCount);
	m_surfaceRayTrace->SetUniform("obstColor", p_params.Schema.Obst.Value());
	m_surfaceRayTrace->SetUniform("obstColorHighlight", p_params.Schema.ObstHighlight.Value());
	m_surfaceRayTrace->SetUniform("viewSize", glm::vec2((float)p_viewSize.Width, (float)p_viewSize.Height));
	
	std::shared_ptr<Ogl::Vao> surface = Ogl->GetVao("surface");
	surface->Bind();
	glBindTexture(GL_TEXTURE_2D, m_floor->CausticsTex());
	
	std::shared_ptr<Ogl::Buffer> obstSsbo = Ogl->GetBuffer("managed_obsts");
	Ogl->BindSSBO(0, *obstSsbo, GL_SHADER_STORAGE_BUFFER);
	
	const int yDimVisible = domain.GetYDimVisible();
	glDrawElements(GL_TRIANGLES, (MAX_XDIM - 1)*(yDimVisible - 1)*3*2 , GL_UNSIGNED_INT, (GLvoid*)0);
	surface->Unbind();
	
	glBindTexture(GL_TEXTURE_2D, 0);
	Ogl->UnbindBO(GL_SHADER_STORAGE_BUFFER);
	m_surfaceRayTrace->Unset();   

#ifdef DRAW_CAMERA
	RenderCameraPos(p_shadingMode, domain, modelMatrix, projectionMatrix, p_cameraPos, p_viewSize, obstHeight);
#endif
}

void ShaderManager::RenderSurfaceContour(const ContourVariable p_contour, Domain &domain, const RenderParams& p_params)
{
	m_surfaceContour->Use();
	m_surfaceContour->SetUniform("modelMatrix", p_params.ModelView);
	m_surfaceContour->SetUniform("projectionMatrix", p_params.Projection);
	m_surfaceContour->SetUniform("cameraPos", p_params.Camera);
	
	std::shared_ptr<Ogl::Vao> surface = Ogl->GetVao("surface");
	surface->Bind();
	
	const int yDimVisible = domain.GetYDimVisible();
	glDrawElements(GL_TRIANGLES, (MAX_XDIM - 1)*(yDimVisible - 1)*3*2 , GL_UNSIGNED_INT, (GLvoid*)0);
	surface->Unbind();
	
	m_surfaceContour->Unset();   

#ifdef DRAW_CAMERA
	RenderCameraPos(p_shadingMode, domain, modelMatrix, projectionMatrix, p_cameraPos, p_viewSize, obstHeight);
#endif
}

void ShaderManager::RenderCameraPos(const RenderParams& p_params)
{
    if (m_cameraDatum->IsInitialized())
    {
        m_cameraDatum->Render(p_params);
    }
}
void ShaderManager::UpdateCameraDatum(const PillarDefinition& p_def)
{
    if (!m_cameraDatum->IsInitialized())
    {
        m_cameraDatum->Initialize();
    }

    m_cameraDatum->SetDefinition(p_def);
}