#pragma once

#ifdef SHIZUKU_CORE_EXPORTS  
#define CORE_API __declspec(dllexport)   
#else  
#define CORE_API __declspec(dllimport)   
#endif  

#include <GLEW/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>


namespace Shizuku{
    namespace Core{
        class CORE_API Shader
        {
            GLuint shaderID;
        public:
            // Constructor generates the shader on the fly
            Shader(const GLchar* filePath, const GLenum shaderType, const GLuint Program);
            GLuint GetId();
        };

        class ShaderProgram
        {
            GLuint ProgramID;
            std::string m_name;
        public:
            Shader *vertexShader;
            Shader *fragmentShader;
            Shader *geometryShader;
            Shader *computeShader;

            ShaderProgram()
                : vertexShader(NULL), fragmentShader(NULL), geometryShader(NULL), computeShader(NULL)
            {}
            void Initialize(std::string name);
            GLuint GetId();
            void CreateShader(const GLchar* filePath, const GLenum shaderType);
            void Use();
            void Unset();
            std::string GetName();

            void SetUniform(const GLchar* varName, const int varValue);
            void SetUniform(const GLchar* varName, const float varValue);
            void SetUniform(const GLchar* varName, const bool varValue);
            void SetUniform(const GLchar* varName, const glm::vec3 varValue);
            void SetUniform(const GLchar* varName, const glm::mat4 varValue);
            void RunSubroutine(const GLchar* subroutineName,
                const glm::ivec3 workGroupSize);
        };
    }
}

