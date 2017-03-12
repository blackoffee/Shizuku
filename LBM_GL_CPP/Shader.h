#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <GLEW/glew.h>

class Shader
{
    GLuint shaderID;
public:
    // Constructor generates the shader on the fly
    Shader(const GLchar* vertexPath, const GLenum shaderType, const GLuint Program);
    GLuint GetId();
};


class ShaderProgram
{
public:
    GLuint ProgramID;
    Shader *vertexShader;
    Shader *fragmentShader;
    Shader *geometryShader;
    Shader *computeShader;

    ShaderProgram();
    void CreateShader(const GLchar* vertexPath, const GLenum shaderType);
    void Use();
    void Unset();
};


