#pragma once

#ifdef SHIZUKU_CORE_EXPORTS  
#define CORE_API __declspec(dllexport)   
#else  
#define CORE_API __declspec(dllimport)   
#endif  

#include <GLEW/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <string>
#include <memory>

namespace Shizuku{
    namespace Core{
        class ShaderProgram;

        class CORE_API Ogl
        {
        public:
            struct Buffer
            {
                GLuint m_id;
                std::string m_name;
                Buffer();
                Buffer(GLuint id, std::string name);
                ~Buffer();
                GLuint GetId();
                std::string GetName();
            };
            void BindBO(GLenum target, Buffer &buffer);
            void BindSSBO(GLuint base, Buffer &buffer, GLenum target = GL_SHADER_STORAGE_BUFFER);
            void UnbindBO(GLenum target);
            struct CORE_API Vao
            {
                Vao();
                Vao(GLuint id, std::string name);
                GLuint m_id;
                std::string m_name;
                void Bind();
                void Unbind();
                ~Vao();
            };
        private:
            std::vector<std::shared_ptr<ShaderProgram>> m_shaderPrograms;
            std::vector<std::shared_ptr<Buffer>> m_buffers;
            std::vector<std::shared_ptr<Vao>> m_vaos;
        public:
            Ogl();

            template <typename T> std::shared_ptr<Buffer> CreateBuffer(const GLenum target, T* data, const unsigned int numberOfElements,
                const std::string name, const GLuint drawMode, const GLuint base = -1);

            std::shared_ptr<Vao> CreateVao(const std::string name);

            std::shared_ptr<Buffer> GetBuffer(const std::string name);

            std::shared_ptr<Vao> GetVao(const std::string name);

            std::shared_ptr<ShaderProgram> CreateShaderProgram(const std::string name);

            std::shared_ptr<ShaderProgram> GetShaderProgram(const std::string name);

        };


        template <typename T>
        std::shared_ptr<Ogl::Buffer> Ogl::CreateBuffer(const GLenum target, T* data, const unsigned int numberOfElements, const std::string name, const GLuint drawMode, const GLuint base)
        {
            GLuint temp;
            glGenBuffers(1, &temp);
            if (target == GL_SHADER_STORAGE_BUFFER)
            {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, base, temp);
            }
            else
            {
                glBindBuffer(target, temp);
            }
            glBufferData(target, numberOfElements*sizeof(T), data, drawMode);
            glBindBuffer(target, 0);
            std::shared_ptr<Ogl::Buffer> buffer = std::make_shared<Ogl::Buffer>(temp, name);
            m_buffers.push_back(buffer);
            return buffer;
        }

    }
}


