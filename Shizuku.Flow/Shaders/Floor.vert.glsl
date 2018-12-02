#version 430 core
layout(location = 0) in vec3 position;

out vec2 texCoord;

uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;

void main()
{
    gl_Position = projectionMatrix*modelMatrix*vec4(position, 1.f);
    texCoord = vec2(0.5f*(position.x+1.0f), 0.5f*(position.y+1.0f));
}