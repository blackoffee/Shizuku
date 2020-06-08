#include "Ogl.h"
#include "Shader.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace Shizuku{
    namespace Core{
        Ogl::Ogl()
        {
        }

        Ogl::Buffer::Buffer()
        {
        }

        Ogl::Buffer::Buffer(GLuint id, std::string name)
        {
            m_id = id;
            m_name = name;
        }

        GLuint Ogl::Buffer::GetId()
        {
            return m_id;
        }

        std::string Ogl::Buffer::GetName()
        {
            return m_name;
        }


        Ogl::Vao::Vao(GLuint id, std::string name)
        {
            m_id = id;
            m_name = name;
        }

        void Ogl::BindBO(const GLenum target, Buffer &buffer)
        {
            glBindBuffer(target, buffer.GetId());
        }

        void Ogl::BindSSBO(const GLuint base, Buffer &buffer, const GLenum target)
        {
            glBindBufferBase(target, base, buffer.GetId());
        }

        void Ogl::UnbindBO(const GLenum target)
        {
            glBindBuffer(target, 0);
        }

        void Ogl::Vao::Bind()
        {
            glBindVertexArray(m_id);
        }

        void Ogl::Vao::Unbind()
        {
            glBindVertexArray(0);
        }

        Ogl::Buffer::~Buffer()
        {
            glDeleteBuffers(1, &m_id);
        }

        Ogl::Vao::~Vao()
        {
            glDeleteVertexArrays(1, &m_id);
        }
        std::shared_ptr<Ogl::Vao> Ogl::CreateVao(const std::string name)
        {
            GLuint temp;
            glGenVertexArrays(1, &temp);
            std::shared_ptr<Ogl::Vao> vao(new Vao{ temp, name });
            m_vaos.push_back(vao);
            return vao;
        }

        std::shared_ptr<Ogl::Buffer> Ogl::GetBuffer(const std::string name)
        {
            for (std::vector<std::shared_ptr<Buffer>>::iterator it = m_buffers.begin(); it != m_buffers.end(); ++it)
            {
                if ((*it)->m_name == name)
                {
                    return *it;
                }
            }
            return NULL;
        }


        std::shared_ptr<Ogl::Vao> Ogl::GetVao(const std::string name)
        {
            for (std::vector<std::shared_ptr<Vao>>::iterator it = m_vaos.begin(); it != m_vaos.end(); ++it)
            {
                if ((*it)->m_name == name)
                {
                    return *it;
                }
            }
            return NULL;
        }

        std::shared_ptr<ShaderProgram> Ogl::CreateShaderProgram(const std::string name)
        {
            std::shared_ptr<ShaderProgram> shaderProgram(new ShaderProgram);
            shaderProgram->Initialize(name);
            m_shaderPrograms.push_back(shaderProgram);
            return shaderProgram;
        }

        std::shared_ptr<ShaderProgram> Ogl::GetShaderProgram(const std::string name)
        {
            for (std::vector<std::shared_ptr<ShaderProgram>>::iterator it = m_shaderPrograms.begin(); it != m_shaderPrograms.end(); ++it)
            {
                if ((*it)->GetName() == name)
                {
                    return *it;
                }
            }
            return NULL;
        }
    }
}
