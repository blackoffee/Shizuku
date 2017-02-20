#version 430 core
in vec3 fColor;

out vec4 color;

void main()
{
    color = vec4(fColor, 0.5f);
}