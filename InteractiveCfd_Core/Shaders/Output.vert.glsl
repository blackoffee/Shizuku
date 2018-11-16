#version 430 core
layout(location = 0) in vec3 position;

out vec2 texCoord;

void main()
{
    gl_Position = vec4(position,1.0f);
    texCoord = vec2(0.5f*(position.x+1.0f), 0.5f*(position.y+1.0f));
}