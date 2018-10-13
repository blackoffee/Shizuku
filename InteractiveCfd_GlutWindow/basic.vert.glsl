#version 430 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

out vec3 fColor;

uniform mat4 Transform;
uniform mat4 Projection;

void main()
{
    gl_Position = vec4(position,1.0f);
    //gl_Position = Projection*Transform*vec4(position,1.0f);
    fColor = color;
}