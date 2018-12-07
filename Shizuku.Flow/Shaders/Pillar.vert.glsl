#version 430 core
layout(location = 0) in vec3 position;

out vec4 fColor;

uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;


void main()
{
    gl_Position = projectionMatrix*modelMatrix*vec4(position, 1.f);
    fColor = vec4(0,1,1,1);
}