#include "Pillar.h"
#include "Shizuku.Core/Ogl/Ogl.h"
#include "Shizuku.Core/Ogl/Shader.h"

using namespace Shizuku::Core;
using namespace Shizuku::Flow;

Pillar::Pillar(std::shared_ptr<Ogl> p_ogl)
{
    m_ogl = p_ogl;
}

void Pillar::Initialize()
{
    PrepareBuffers();
    PrepareShader();
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
    m_shaderProgram->CreateShader("Pillar.vert.glsl", GL_VERTEX_SHADER);
    m_shaderProgram->CreateShader("Pillar.frag.glsl", GL_FRAGMENT_SHADER);
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

void Pillar::Draw(const glm::mat4& p_model, const glm::mat4& p_proj)
{
    glm::mat4 modelMat;
    glm::mat4 scale = glm::scale(glm::mat4(1), glm::vec3(m_def.Size().Width, m_def.Size().Height, m_def.Size().Depth));
    glm::mat4 trans = glm::translate(
        glm::mat4(1),
        glm::vec3(m_def.Pos().X-0.5f*m_def.Size().Width, m_def.Pos().Y-0.5f*m_def.Size().Height, -1.0f));
    modelMat = p_model * trans * scale;
    glm::mat4 modelInvTrans = glm::transpose(glm::inverse(modelMat));

    m_shaderProgram->Use();
    glActiveTexture(GL_TEXTURE0);

    m_shaderProgram->SetUniform("modelMatrix", modelMat);
    m_shaderProgram->SetUniform("projectionMatrix", p_proj);
    m_shaderProgram->SetUniform("modelInvTrans", modelInvTrans);

    std::shared_ptr<Ogl::Vao> pillar = m_ogl->GetVao("pillar");
    pillar->Bind();

    glDrawElements(GL_TRIANGLES, 10*3, GL_UNSIGNED_INT, (GLvoid*)0);
    pillar->Unbind();

    m_shaderProgram->Unset();   
}

