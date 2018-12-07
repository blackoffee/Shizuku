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
        //bottom
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        //top
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
    };

    const GLuint elemIndices[] = {
        //top
        4, 6, 5,
        5, 6, 7,
        //left
        2, 6, 0,
        0, 6, 4,
        //right
        1, 5, 3,
        3, 5, 7,
        //front
        0, 4, 1,
        1, 4, 5,
        //back
        3, 7, 2,
        2, 7, 6
    };

    m_ogl->CreateBuffer(GL_ARRAY_BUFFER, quadVertices, 24, "pillar", GL_STATIC_DRAW);
    m_ogl->CreateBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndices, 10 * 3, "pillar indices", GL_STATIC_DRAW);

    m_ogl->BindBO(GL_ARRAY_BUFFER, *m_ogl->GetBuffer("pillar"));
    m_ogl->BindBO(GL_ELEMENT_ARRAY_BUFFER, *m_ogl->GetBuffer("pillar indices"));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    pillar->Unbind();
}

void Pillar::PrepareShader()
{
    m_shaderProgram = std::make_shared<ShaderProgram>();
    m_shaderProgram->Initialize("Pillar");
    m_shaderProgram->CreateShader("Pillar.vert.glsl", GL_VERTEX_SHADER);
    m_shaderProgram->CreateShader("Pillar.frag.glsl", GL_FRAGMENT_SHADER);
}

void Pillar::SetPosition(const Types::Point<float>& p_pos)
{
    m_position = p_pos;
}

void Pillar::SetSize(const Types::Point<float>& p_size)
{
    m_size = p_size;
}

void Pillar::Draw(const glm::mat4& p_model, const glm::mat4& p_proj)
{
    glm::mat4 modelMat;
    glm::mat4 scale = glm::scale(glm::mat4(1), glm::vec3(0.5f*m_size.X, 0.5f*m_size.Y, 0.5f));
    glm::mat4 trans = glm::translate(glm::mat4(1), glm::vec3(m_position.X, m_position.Y, -0.5f));
    modelMat = p_model * trans * scale;

    m_shaderProgram->Use();
    glActiveTexture(GL_TEXTURE0);

    m_shaderProgram->SetUniform("modelMatrix", modelMat);
    m_shaderProgram->SetUniform("projectionMatrix", p_proj);

    std::shared_ptr<Ogl::Vao> pillar = m_ogl->GetVao("pillar");
    pillar->Bind();

    glDrawElements(GL_TRIANGLES, 10*3, GL_UNSIGNED_INT, (GLvoid*)0);
    pillar->Unbind();

    m_shaderProgram->Unset();   
}

