#include "Shader.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <glm/gtc/type_ptr.hpp>

namespace Shizuku{
    namespace Core{
        Shader::Shader(const GLchar* filePath, const GLenum shaderType, const GLuint Program)
        {
            // 1. Retrieve the vertex/fragment source code from filePath
            std::string vertexCode;
            std::ifstream vShaderFile;
            // ensures ifstream objects can throw exceptions:
            vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            try
            {
                // Open files
                vShaderFile.open(filePath);
                if (!vShaderFile.is_open())
                    throw std::runtime_error("Failed to open file.");

                std::stringstream vShaderStream;
                // Read file's buffer contents into streams
                vShaderStream << vShaderFile.rdbuf();
                // close file handlers
                vShaderFile.close();
                // Convert stream into string
                vertexCode = vShaderStream.str();
            }
			catch (std::system_error ex)
			{
				std::cout<< ex.code().message();

			}
            catch (std::ifstream::failure e)
            {
                std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << e.what() << std::endl;
            }
            const GLchar* vShaderCode = vertexCode.c_str();
            // 2. Compile shaders
            GLint success;
            GLchar infoLog[512];
            // Vertex Shader
            shaderID = glCreateShader(shaderType);
            glShaderSource(shaderID, 1, &vShaderCode, NULL);
            glCompileShader(shaderID);
            // Print compile errors if any
            glGetShaderiv(shaderID, GL_COMPILE_STATUS, &success);
            if (!success)
            {
                glGetShaderInfoLog(shaderID, 512, NULL, infoLog);
                std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
            }

            // Shader Program
            glAttachShader(Program, shaderID);
            glLinkProgram(Program);
            // Print linking errors if any
            glGetProgramiv(Program, GL_LINK_STATUS, &success);
            if (!success)
            {
                glGetProgramInfoLog(Program, 512, NULL, infoLog);
                std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
            }
            // Delete the shaders as they're linked into our program now and no longer necessery
            glDeleteShader(shaderID);
        }

        GLuint Shader::GetId()
        {
            return shaderID;
        }

        void ShaderProgram::Initialize(std::string name)
        {
            m_name = name;
            ProgramID = glCreateProgram();
        }

        GLuint ShaderProgram::GetId()
        {
            return ProgramID;
        }

        void ShaderProgram::CreateShader(const GLchar* filePath, const GLenum shaderType)
        {
            if (shaderType == GL_VERTEX_SHADER)
            {
                this->vertexShader = &Shader(filePath, shaderType, ProgramID);
            }
            else if (shaderType == GL_FRAGMENT_SHADER)
            {
                this->fragmentShader = &Shader(filePath, shaderType, ProgramID);
            }
            else if (shaderType == GL_GEOMETRY_SHADER)
            {
                this->geometryShader = &Shader(filePath, shaderType, ProgramID);
            }
            else if (shaderType == GL_COMPUTE_SHADER)
            {
                this->computeShader = &Shader(filePath, shaderType, ProgramID);
            }
        }

        void ShaderProgram::Use()
        {
            glUseProgram(ProgramID);
        }

        void ShaderProgram::Unset()
        {
            glUseProgram(0);
        }

        std::string ShaderProgram::GetName()
        {
            return m_name;
        }

        void ShaderProgram::SetUniform(const GLchar* varName, const int varValue)
        {
            const GLint targetLocation = glGetUniformLocation(ProgramID, varName);
            glUniform1i(targetLocation, varValue);
        }

        void ShaderProgram::SetUniform(const GLchar* varName, const float varValue)
        {
            const GLint targetLocation = glGetUniformLocation(ProgramID, varName);
            glUniform1f(targetLocation, varValue);
        }

        void ShaderProgram::SetUniform(const GLchar* varName, const bool varValue)
        {
            const GLint targetLocation = glGetUniformLocation(ProgramID, varName);
            glUniform1i(targetLocation, varValue);
        }

        void ShaderProgram::SetUniform(const GLchar* varName, const glm::vec2& varValue)
        {
            const GLint targetLocation = glGetUniformLocation(ProgramID, varName);
            glUniform2f(targetLocation, varValue.x, varValue.y);
        }

        void ShaderProgram::SetUniform(const GLchar* varName, const glm::vec3& varValue)
        {
            const GLint targetLocation = glGetUniformLocation(ProgramID, varName);
            glUniform3f(targetLocation, varValue.x, varValue.y, varValue.z);
        }

        void ShaderProgram::SetUniform(const GLchar* varName, const glm::vec4& varValue)
        {
            const GLint targetLocation = glGetUniformLocation(ProgramID, varName);
            glUniform4f(targetLocation, varValue.x, varValue.y, varValue.z, varValue.w);
        }

        void ShaderProgram::SetUniform(const GLchar* varName, const glm::mat4& varValue)
        {
            const GLint targetLocation = glGetUniformLocation(ProgramID, varName);
            glUniformMatrix4fv(targetLocation, 1, GL_FALSE, glm::value_ptr(varValue));
        }

        void ShaderProgram::RunSubroutine(const GLchar* subroutineName, const glm::ivec3& workGroupSize)
        {
            const GLuint subroutine = glGetSubroutineIndex(ProgramID, GL_COMPUTE_SHADER,
                subroutineName);
            glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &subroutine);
            glDispatchCompute(workGroupSize.x, workGroupSize.y, workGroupSize.z);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }
}


