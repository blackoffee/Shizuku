#pragma once
#include <GLEW/glew.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>


class Shader
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
public:
    Shader *vertexShader;
    Shader *fragmentShader;
    Shader *geometryShader;
    Shader *computeShader;

    ShaderProgram()
        : vertexShader(NULL), fragmentShader(NULL), geometryShader(NULL), computeShader(NULL)
    {}
    void Initialize();
    GLuint GetId();
    void CreateShader(const GLchar* filePath, const GLenum shaderType);
    void Use();
    void Unset();
};


