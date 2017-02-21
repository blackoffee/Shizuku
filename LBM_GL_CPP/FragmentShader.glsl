#version 430 core
in vec4 fColor;

out vec4 color;

void main()
{
    color = vec4(fColor);
}