#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <GLEW/glew.h>
#include <GLFW/glfw3.h>

class Shader
{
public:
    GLuint shaderID;
    // Constructor generates the shader on the fly
    Shader(const GLchar* vertexPath, const GLenum shaderType, const GLuint Program)
    {
        // 1. Retrieve the vertex/fragment source code from filePath
        std::string vertexCode;
        std::ifstream vShaderFile;
        // ensures ifstream objects can throw exceptions:
        vShaderFile.exceptions(std::ifstream::badbit);
        try
        {
            // Open files
            vShaderFile.open(vertexPath);
            std::stringstream vShaderStream;
            // Read file's buffer contents into streams
            vShaderStream << vShaderFile.rdbuf();
            // close file handlers
            vShaderFile.close();
            // Convert stream into string
            vertexCode = vShaderStream.str();
        }
        catch (std::ifstream::failure e)
        {
            std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
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
};


class ShaderProgram
{
public:
    GLuint ProgramID;
    Shader *vertexShader;
    Shader *fragmentShader;
    Shader *geometryShader;
    Shader *computeShader;

    ShaderProgram()
    {
        ProgramID = glCreateProgram();
    }

    void CreateShader(const GLchar* vertexPath, const GLenum shaderType)
    {
        if (shaderType == GL_VERTEX_SHADER)
        {
            this->vertexShader = &Shader(vertexPath, shaderType, ProgramID);
        }
        else if (shaderType == GL_FRAGMENT_SHADER)
        {
            this->fragmentShader = &Shader(vertexPath, shaderType, ProgramID);
        }
        else if (shaderType == GL_GEOMETRY_SHADER)
        {
            this->geometryShader = &Shader(vertexPath, shaderType, ProgramID);
        }
        else if (shaderType == GL_COMPUTE_SHADER)
        {
            this->computeShader = &Shader(vertexPath, shaderType, ProgramID);
        }
    }

    void Use()
    {
        glUseProgram(ProgramID);
    }
};


