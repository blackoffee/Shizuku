#version 430 core
in vec4 fColor;

out vec4 color;

void main()
{
    color = vec4(fColor);
    //color = vec4(0.5f,0.0f,0.0f,1.f);
}