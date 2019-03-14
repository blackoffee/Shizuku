#include "Pillar.h"
#include "Algorithms/Intersection.h"
#include "Shizuku.Core/Ogl/Ogl.h"
#include "Shizuku.Core/Ogl/Shader.h"
#include "Shizuku.Core/Types/Point3D.h"

using namespace Shizuku::Core;
using namespace Shizuku::Flow;

namespace
{
    void GetMouseRay(glm::vec3 &p_rayOrigin, glm::vec3 &p_rayDir, const HitParams& p_params)
    {
        glm::mat4 mvp = p_params.Projection*p_params.Modelview;
        glm::mat4 mvpInv = glm::inverse(mvp);
        glm::vec4 v1 = { (float)p_params.ScreenPos.X / (p_params.ViewSize.Width)*2.f - 1.f, (float)p_params.ScreenPos.Y / (p_params.ViewSize.Height)*2.f - 1.f, 0.0f*2.f - 1.f, 1.0f };
        glm::vec4 v2 = { (float)p_params.ScreenPos.X / (p_params.ViewSize.Width)*2.f - 1.f, (float)p_params.ScreenPos.Y / (p_params.ViewSize.Height)*2.f - 1.f, 1.0f*2.f - 1.f, 1.0f };
        glm::vec4 r1 = mvpInv * v1;
        glm::vec4 r2 = mvpInv * v2;
        p_rayOrigin.x = r1.x / r1.w;
        p_rayOrigin.y = r1.y / r1.w;
        p_rayOrigin.z = r1.z / r1.w;
        p_rayDir.x = r2.x / r2.w - p_rayOrigin.x;
        p_rayDir.y = r2.y / r2.w - p_rayOrigin.y;
        p_rayDir.z = r2.z / r2.w - p_rayOrigin.z;
        float mag = sqrt(p_rayDir.x*p_rayDir.x + p_rayDir.y*p_rayDir.y + p_rayDir.z*p_rayDir.z);
        p_rayDir.x /= mag;
        p_rayDir.y /= mag;
        p_rayDir.z /= mag;
    }
}

Pillar::Pillar(std::shared_ptr<Ogl> p_ogl)
{
    m_ogl = p_ogl;
    m_initialized = false;
    m_highlighted = false;
}

void Pillar::Initialize()
{
    PrepareBuffers();
    PrepareShader();
    m_initialized = true;
}

bool Pillar::IsInitialized()
{
    return m_initialized;
}

void Pillar::PrepareBuffers()
{
    std::shared_ptr<Ogl::Vao> pillar = m_ogl->CreateVao("pillar");
    pillar->Bind();

    const GLfloat quadVertices[] = {
        //left
        0.0f, 0.0f, 0.0f,
        0.0f,  1.0f, 0.0f,
        0.0f, 0.0f,  1.0f,
        0.0f,  1.0f,  1.0f,
        //right
         1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
         1.0f, 0.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        //front
        0.0f, 0.0f, 0.0f,
         1.0f, 0.0f, 0.0f,
        0.0f, 0.0f,  1.0f,
         1.0f, 0.0f,  1.0f,
        //back
        0.0f,  1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
        0.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        //top
        0.0f, 0.0f,  1.0f,
         1.0f, 0.0f,  1.0f,
        0.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        //bottom
        0.0f, 0.0f, 0.0f,
         1.0f, 0.0f, 0.0f,
        0.0f,  1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
    };

    const GLfloat quadNormals[] = {
        //left
         1.0f,  0.0f,  0.0f,
         1.0f,  0.0f,  0.0f,
         1.0f,  0.0f,  0.0f,
         1.0f,  0.0f,  0.0f,
        //right
        -1.0f,  0.0f,  0.0f,
        -1.0f,  0.0f,  0.0f,
        -1.0f,  0.0f,  0.0f,
        -1.0f,  0.0f,  0.0f,
        //front
         0.0f,  1.0f,  0.0f,
         0.0f,  1.0f,  0.0f,
         0.0f,  1.0f,  0.0f,
         0.0f,  1.0f,  0.0f,
        //back
         0.0f, -1.0f,  0.0f,
         0.0f, -1.0f,  0.0f,
         0.0f, -1.0f,  0.0f,
         0.0f, -1.0f,  0.0f,
        //top
         0.0f,  0.0f, -1.0f,
         0.0f,  0.0f, -1.0f,
         0.0f,  0.0f, -1.0f,
         0.0f,  0.0f, -1.0f,
        //bottom
         0.0f,  0.0f,  1.0f,
         0.0f,  0.0f,  1.0f,
         0.0f,  0.0f,  1.0f,
         0.0f,  0.0f,  1.0f,
    };

    const GLuint elemIndices[] = {
        //left 
        0, 1, 3,
        0, 3, 2,
        //right
        4+0, 4+2, 4+3,
        4+0, 4+3, 4+1,
        //front
        8+0, 8+2, 8+3,
        8+0, 8+3, 8+1,
        //back 
        12+0, 12+1, 12+3,
        12+0, 12+3, 12+2,
        //top
        16+0, 16+2, 16+3,
        16+0, 16+3, 16+1,
        //bottom 
        20+0, 20+1, 20+3,
        20+0, 20+3, 20+2
    };

    m_ogl->CreateBuffer(GL_ARRAY_BUFFER, quadVertices, 72, "pillar vert", GL_STATIC_DRAW);
    m_ogl->CreateBuffer(GL_ARRAY_BUFFER, quadNormals, 72, "pillar norm", GL_STATIC_DRAW);
    m_ogl->CreateBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndices, 12 * 3, "pillar indices", GL_STATIC_DRAW);

    m_ogl->BindBO(GL_ARRAY_BUFFER, *m_ogl->GetBuffer("pillar vert"));
    m_ogl->BindBO(GL_ELEMENT_ARRAY_BUFFER, *m_ogl->GetBuffer("pillar indices"));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    m_ogl->BindBO(GL_ARRAY_BUFFER, *m_ogl->GetBuffer("pillar norm"));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    pillar->Unbind();
}

void Pillar::PrepareShader()
{
    m_shaderProgram = std::make_shared<ShaderProgram>();
    m_shaderProgram->Initialize("Pillar");
    m_shaderProgram->CreateShader("Assets/Pillar.vert.glsl", GL_VERTEX_SHADER);
    m_shaderProgram->CreateShader("Assets/Pillar.frag.glsl", GL_FRAGMENT_SHADER);
}

const PillarDefinition& Pillar::Def()
{
    return m_def;
}

void Pillar::SetDefinition(const PillarDefinition& p_def)
{
    m_def = p_def;
}

void Pillar::SetPosition(const Types::Point<float>& p_pos)
{
    m_def.SetPosition(p_pos);
}

void Pillar::SetSize(const Types::Box<float>& p_size)
{
    m_def.SetSize(p_size);
}

void Pillar::Highlight(const bool p_highlight)
{
    m_highlighted = p_highlight;
}

HitResult Pillar::Hit(const HitParams& p_params)
{
    const Types::Box<float> bounds = Types::Box<float>(m_def.Size());
    const Types::Point3D<float> center = Types::Point3D<float>(m_def.Pos().X, m_def.Pos().Y, -1.f + 0.5f*m_def.Size().Depth);

    glm::vec3 rayOrigin, rayDir;
    GetMouseRay(rayOrigin, rayDir, p_params);

    float dist;
    return HitResult{
        Algorithms::Intersection::IntersectAABBWithRay(dist, rayOrigin, rayDir, center, bounds),
        boost::optional<float>(dist)
    };
}

void Pillar::Render(const RenderParams& p_params)
{
    glm::mat4 scale = glm::scale(glm::mat4(1), glm::vec3(m_def.Size().Width, m_def.Size().Height, m_def.Size().Depth));
    glm::mat4 trans = glm::translate(
        glm::mat4(1),
        glm::vec3(m_def.Pos().X-0.5f*m_def.Size().Width, m_def.Pos().Y-0.5f*m_def.Size().Height, -1.0f));
    glm::mat4 modelMat = trans * scale;
    glm::mat4 modelInvTrans = glm::transpose(glm::inverse(modelMat));

    m_shaderProgram->Use();
    glActiveTexture(GL_TEXTURE0);

    m_shaderProgram->SetUniform("modelMatrix", modelMat);
    m_shaderProgram->SetUniform("viewMatrix", p_params.ModelView);
    m_shaderProgram->SetUniform("projectionMatrix", p_params.Projection);
    m_shaderProgram->SetUniform("modelInvTrans", modelInvTrans);
    m_shaderProgram->SetUniform("cameraPos", p_params.Camera);
    const glm::vec4 col = m_highlighted? p_params.Schema.ObstHighlight.Value() : p_params.Schema.Obst.Value();
    m_shaderProgram->SetUniform("obstAlbedo", col);
    std::shared_ptr<Ogl::Vao> pillar = m_ogl->GetVao("pillar");
    pillar->Bind();

    glDrawElements(GL_TRIANGLES, 10*3, GL_UNSIGNED_INT, (GLvoid*)0);
    pillar->Unbind();

    m_shaderProgram->Unset();   
}

