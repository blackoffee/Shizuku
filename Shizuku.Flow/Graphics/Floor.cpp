#include "Floor.h"
#include "Domain.h"
#include "Shizuku.Core/Ogl/Shader.h"
#include "common.h"

#include <soil.h>
#include "cuda_runtime.h"
#include <GLEW/glew.h>
#include "cuda_gl_interop.h"  // needs GLEW

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

using namespace Shizuku::Core;
using namespace Shizuku::Flow;


Floor::Floor(std::shared_ptr<Ogl> p_ogl)
{
    m_ogl = p_ogl;
    m_initialized = false;
    m_floorShader = std::make_shared<ShaderProgram>();
    m_causticsShader = std::make_shared<ShaderProgram>();
    m_lightRayShader = std::make_shared<ShaderProgram>();
    m_beamPathShader = std::make_shared<ShaderProgram>();
}

void Floor::SetVbo(std::shared_ptr<Ogl::Buffer> p_vbo)
{
    m_vbo = p_vbo;
}

void Floor::Initialize()
{
    CompileShaders();
    PrepareIndices();
    PrepareVaos();
    PrepareTextures();
    cudaGraphicsGLRegisterImage(&m_cudaFloorLightTextureResource, m_causticsTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);
    m_initialized = true;
}

bool Floor::IsInitialized()
{
    return m_initialized;
}

void Floor::CompileShaders()
{
    m_floorShader->Initialize("Floor");
    m_floorShader->CreateShader("Assets/Floor.vert.glsl", GL_VERTEX_SHADER);
    m_floorShader->CreateShader("Assets/Floor.frag.glsl", GL_FRAGMENT_SHADER);
    m_lightRayShader->Initialize("LightRay");
    m_lightRayShader->CreateShader("Assets/LightRay.vert.glsl", GL_VERTEX_SHADER);
    m_lightRayShader->CreateShader("Assets/LightRay.geom.glsl", GL_GEOMETRY_SHADER);
    m_lightRayShader->CreateShader("Assets/LightRay.frag.glsl", GL_FRAGMENT_SHADER);
    m_beamPathShader->Initialize("BeamPath");
    m_beamPathShader->CreateShader("Assets/BeamPath.vert.glsl", GL_VERTEX_SHADER);
    m_beamPathShader->CreateShader("Assets/BeamPath.geom.glsl", GL_GEOMETRY_SHADER);
    m_beamPathShader->CreateShader("Assets/BeamPath.frag.glsl", GL_FRAGMENT_SHADER);
    m_causticsShader->Initialize("Caustics");
    m_causticsShader->CreateShader("Assets/Caustics.vert.glsl", GL_VERTEX_SHADER);
    m_causticsShader->CreateShader("Assets/Caustics.frag.glsl", GL_FRAGMENT_SHADER);
}

void Floor::PrepareTextures()
{
    int width, height;
    unsigned char* image = SOIL_load_image("Assets/Floor.png", &width, &height, 0, SOIL_LOAD_RGB);
    //std::cout << SOIL_last_result() << std::endl;
    assert(image != NULL);
    float* tex = new float[4 * width*height];
    for (int i = 0; i < width*height; ++i)
    {
        tex[4 * i] = image[3 * i]/255.f;
        tex[4 * i + 1] = image[3 * i + 1]/255.f;
        tex[4 * i + 2] = image[3 * i + 2]/255.f;
        tex[4 * i + 3] = 1.f;// color[3];
    }

    glGenTextures(1, &m_floorTex);
    glBindTexture(GL_TEXTURE_2D, m_floorTex);

    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glGenerateMipmap(GL_TEXTURE_2D);

    SOIL_free_image_data(image);

    glBindTexture(GL_TEXTURE_2D, 0);



    // set up FBO and texture to render to 
    glGenFramebuffers(1, &m_floorFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_floorFbo);

    glGenTextures(1, &m_causticsTex);
    glBindTexture(GL_TEXTURE_2D, m_causticsTex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, CAUSTICS_TEX_SIZE, CAUSTICS_TEX_SIZE, 0, GL_RGBA, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    const GLfloat color[4] = { 0, 0, 0, 0 };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_causticsTex, 0);

    GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, drawBuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw "Framebuffer creation failed";

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLuint Floor::CausticsTex()
{
    return m_causticsTex;
}

cudaGraphicsResource * Shizuku::Flow::Floor::CudaFloorLightTextureResource()
{
    return m_cudaFloorLightTextureResource;
}

void Floor::SetProbeRegion(const ProbeRegion& p_region)
{
    m_region = p_region;
}

void Floor::PrepareIndices()
{
    const int numberOfElements = (MAX_XDIM - 1)*(MAX_YDIM - 1);
    const int numberOfNodes = MAX_XDIM * MAX_YDIM;
    GLuint* elementIndices = new GLuint[numberOfElements * 3 * 2];
    for (int j = 0; j < MAX_YDIM-1; j++){
        for (int i = 0; i < MAX_XDIM-1; i++){
            //going clockwise, since y orientation will be flipped when rendered
            elementIndices[j*(MAX_XDIM-1)*6+i*6+0] = numberOfNodes+(i)+(j)*MAX_XDIM;
            elementIndices[j*(MAX_XDIM-1)*6+i*6+1] = numberOfNodes+(i+1)+(j)*MAX_XDIM;
            elementIndices[j*(MAX_XDIM-1)*6+i*6+2] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;

            elementIndices[j*(MAX_XDIM-1)*6+i*6+3] = numberOfNodes+(i)+(j)*MAX_XDIM;
            elementIndices[j*(MAX_XDIM-1)*6+i*6+4] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;
            elementIndices[j*(MAX_XDIM-1)*6+i*6+5] = numberOfNodes+(i)+(j+1)*MAX_XDIM;
        }
    }

    GLuint* elemEdgeIndices = new GLuint[numberOfElements * 12];

    for (int j = 0; j < MAX_YDIM-1; j++){
        for (int i = 0; i < MAX_XDIM-1; i++){
            //going clockwise, since y orientation will be flipped when rendered
            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+0] = numberOfNodes+(i)+(j)*MAX_XDIM;
            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+1] = numberOfNodes+(i+1)+(j)*MAX_XDIM;
            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+2] = numberOfNodes+(i+1)+(j)*MAX_XDIM;
            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+3] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;
            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+4] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;
            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+5] = numberOfNodes+(i)+(j)*MAX_XDIM;

            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+6] = numberOfNodes+(i)+(j)*MAX_XDIM;
            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+7] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;
            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+8] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;
            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+9] = numberOfNodes+(i)+(j+1)*MAX_XDIM;
            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+10] = numberOfNodes+(i)+(j+1)*MAX_XDIM;
            elemEdgeIndices[j*(MAX_XDIM-1)*12+i*12+11] = numberOfNodes+(i)+(j)*MAX_XDIM;
        }
    }

    GLuint* lightPathIndices = new GLuint[numberOfNodes * 2];
    for (int j = 0; j < MAX_YDIM; j++){
        for (int i = 0; i < MAX_XDIM; i++){
            //going clockwise, since y orientation will be flipped when rendered
            lightPathIndices[j*MAX_XDIM*2+i*2+0] = i+j*MAX_XDIM;
            lightPathIndices[j*MAX_XDIM*2+i*2+1] = numberOfNodes+i+j*MAX_XDIM;
        }
    }

    GLuint* beamPathIndices = new GLuint[2 * 2 * 3 * numberOfElements];
    for (int j = 0; j < MAX_YDIM - 1; j++) {
        for (int i = 0; i < MAX_XDIM - 1; ++i)
        {
            beamPathIndices[j*(MAX_XDIM - 1) * 12 + i * 12 + 0] = i + j * MAX_XDIM;
            beamPathIndices[j*(MAX_XDIM - 1) * 12 + i * 12 + 1] = (i+1) + j * MAX_XDIM;
            beamPathIndices[j*(MAX_XDIM - 1) * 12 + i * 12 + 2] = i + (j+1) * MAX_XDIM;

            beamPathIndices[j*(MAX_XDIM - 1) * 12 + i * 12 + 3] = numberOfNodes + i + j * MAX_XDIM;
            beamPathIndices[j*(MAX_XDIM - 1) * 12 + i * 12 + 4] = numberOfNodes + (i+1) + j * MAX_XDIM;
            beamPathIndices[j*(MAX_XDIM - 1) * 12 + i * 12 + 5] = numberOfNodes + i + (j+1) * MAX_XDIM;

            beamPathIndices[j*(MAX_XDIM - 1) * 12 + i * 12 + 6] = (i+1) + j * MAX_XDIM;
            beamPathIndices[j*(MAX_XDIM - 1) * 12 + i * 12 + 7] = (i+1) + (j+1) * MAX_XDIM;
            beamPathIndices[j*(MAX_XDIM - 1) * 12 + i * 12 + 8] = i + (j+1) * MAX_XDIM;

            beamPathIndices[j*(MAX_XDIM - 1) * 12 + i * 12 + 9]  = numberOfNodes + (i+1) + j * MAX_XDIM;
            beamPathIndices[j*(MAX_XDIM - 1) * 12 + i * 12 + 10] = numberOfNodes + (i+1) + (j+1) * MAX_XDIM;
            beamPathIndices[j*(MAX_XDIM - 1) * 12 + i * 12 + 11] = numberOfNodes + i + (j+1) * MAX_XDIM;
        }
    }

    m_ogl->CreateBuffer(GL_ELEMENT_ARRAY_BUFFER, elementIndices, numberOfElements * 3 * 2, "DeformedFloor_indices", GL_DYNAMIC_DRAW);
    free(elementIndices);
    m_ogl->CreateBuffer(GL_ELEMENT_ARRAY_BUFFER, elemEdgeIndices, numberOfElements * 6 * 2, "DeformedFloorEdges_indices", GL_DYNAMIC_DRAW);
    free(elemEdgeIndices);
    m_ogl->CreateBuffer(GL_ELEMENT_ARRAY_BUFFER, lightPathIndices, numberOfNodes * 2, "lightPaths_indices", GL_DYNAMIC_DRAW);
    free(lightPathIndices);
    m_ogl->CreateBuffer(GL_ELEMENT_ARRAY_BUFFER, beamPathIndices, 2 * 2 * 3 * numberOfElements, "beamPaths_indices", GL_DYNAMIC_DRAW);
    free(beamPathIndices);
}

void Floor::PrepareVaos()
{
    std::shared_ptr<Ogl::Vao> deformedFloor = m_ogl->CreateVao("DeformedFloor");
    deformedFloor->Bind();

    m_ogl->BindBO(GL_ARRAY_BUFFER, *m_vbo);
    m_ogl->BindBO(GL_ELEMENT_ARRAY_BUFFER, *m_ogl->GetBuffer("DeformedFloor_indices"));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 16, (GLvoid*)(3 * sizeof(GLfloat)));

    deformedFloor->Unbind();

    std::shared_ptr<Ogl::Vao> deformedFloorEdges = m_ogl->CreateVao("DeformedFloorEdges");
    deformedFloorEdges->Bind();

    m_ogl->BindBO(GL_ARRAY_BUFFER, *m_ogl->GetBuffer("surface"));
    m_ogl->BindBO(GL_ELEMENT_ARRAY_BUFFER, *m_ogl->GetBuffer("DeformedFloorEdges_indices"));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 16, (GLvoid*)(3 * sizeof(GLfloat)));

    deformedFloorEdges->Unbind();

    std::shared_ptr<Ogl::Vao> lightPaths = m_ogl->CreateVao("LightPaths");
    lightPaths->Bind();

    m_ogl->BindBO(GL_ARRAY_BUFFER, *m_vbo);
    m_ogl->BindBO(GL_ELEMENT_ARRAY_BUFFER, *m_ogl->GetBuffer("lightPaths_indices"));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 16, (GLvoid*)(3 * sizeof(GLfloat)));

    lightPaths->Unbind();

    std::shared_ptr<Ogl::Vao> beamPaths = m_ogl->CreateVao("BeamPaths");
    beamPaths->Bind();

    m_ogl->BindBO(GL_ARRAY_BUFFER, *m_vbo);
    m_ogl->BindBO(GL_ELEMENT_ARRAY_BUFFER, *m_ogl->GetBuffer("beamPaths_indices"));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 16, (GLvoid*)(3 * sizeof(GLfloat)));

    beamPaths->Unbind();

    std::shared_ptr<Ogl::Vao> floor = m_ogl->CreateVao("Floor");
    floor->Bind();

    const GLfloat quadVertices[] = {
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
    };

    const auto floorVbo = m_ogl->CreateBuffer(GL_ARRAY_BUFFER, quadVertices, 18, "Floor", GL_STATIC_DRAW);

    m_ogl->BindBO(GL_ARRAY_BUFFER, *floorVbo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    floor->Unbind();
}

void Floor::RenderCausticsToTexture(Domain &domain, const Rect<int>& p_viewSize)
{
    std::shared_ptr<Ogl::Vao> surface = m_ogl->GetVao("DeformedFloor");
    surface->Bind();

    glActiveTexture(GL_TEXTURE0);
    glBindFramebuffer(GL_FRAMEBUFFER, m_floorFbo);
    glBindTexture(GL_TEXTURE_2D, m_floorTex);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_causticsShader->Use();

    m_causticsShader->SetUniform("texCoordScale", 3.f);
    glViewport(0, 0, CAUSTICS_TEX_SIZE, CAUSTICS_TEX_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);

    //Draw floor
    const int xDimVisible = domain.GetXDimVisible();
    const int yDimVisible = domain.GetYDimVisible();
    const int offsetToFloorIndices = 0;// 3 * 2 * (MAX_XDIM - 1)*(MAX_YDIM - 1);

    glDrawElements(GL_TRIANGLES, 3 * (2 * (MAX_XDIM - 1))*(yDimVisible-1), GL_UNSIGNED_INT,
        BUFFER_OFFSET(sizeof(GLuint)*(offsetToFloorIndices)));

    m_causticsShader->Unset();

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDisable(GL_BLEND);

    surface->Unbind();
    glViewport(0, 0, p_viewSize.Width, p_viewSize.Height);
}

void Floor::Render(Domain &p_domain, const RenderParams& p_params)
{
    std::shared_ptr<ShaderProgram> floorShader = m_floorShader;
    floorShader->Use();
    glActiveTexture(GL_TEXTURE0);
    floorShader->SetUniform("modelMatrix", p_params.ModelView);
    floorShader->SetUniform("projectionMatrix", p_params.Projection);

    std::shared_ptr<Ogl::Vao> floor = m_ogl->GetVao("Floor");
    floor->Bind();
    glBindTexture(GL_TEXTURE_2D, m_causticsTex);

    glDrawArrays(GL_TRIANGLES, 0 , 6);

    floor->Unbind();
    glBindTexture(GL_TEXTURE_2D, 0);
    floorShader->Unset();
}

void Floor::RenderCausticsMesh(Domain &p_domain, const RenderParams& p_params)
{
    m_lightRayShader->Use();
    m_lightRayShader->SetUniform("modelMatrix", p_params.ModelView);
    m_lightRayShader->SetUniform("projectionMatrix", p_params.Projection);
    m_lightRayShader->SetUniform("Filter", false);
    std::shared_ptr<Ogl::Vao> surface = m_ogl->GetVao("DeformedFloorEdges");
    surface->Bind();

    const int xDimVisible = p_domain.GetXDimVisible();
    const int yDimVisible = p_domain.GetYDimVisible();
    for (int i = 0; i < yDimVisible - 1; ++i)
    {
        glDrawElements(GL_LINES, (xDimVisible - 1) * 12, GL_UNSIGNED_INT,
            BUFFER_OFFSET(sizeof(GLuint)*(MAX_XDIM - 1)*i * 12));
    }

    surface->Unbind();
    m_lightRayShader->Unset();
}

void Floor::RenderCausticsBeams(Domain &p_domain, const RenderParams& p_params)
{
    //glDisable(GL_DEPTH_TEST);
    m_beamPathShader->Use();
    std::shared_ptr<Ogl::Vao> paths = m_ogl->GetVao("BeamPaths");
    paths->Bind();

    m_beamPathShader->SetUniform("modelMatrix", p_params.ModelView);
    m_beamPathShader->SetUniform("projectionMatrix", p_params.Projection);
    m_beamPathShader->SetUniform("Filter", true);
    m_beamPathShader->SetUniform("Target", glm::vec2(m_region.Pos.X, m_region.Pos.Y));

    glDrawElements(GL_TRIANGLES_ADJACENCY, 2 * 2 * 3 * (MAX_XDIM - 1)*(MAX_YDIM - 1), GL_UNSIGNED_INT, (GLvoid*)0);

    paths->Unbind();
    m_beamPathShader->Unset();

    //glEnable(GL_DEPTH_TEST);
}
