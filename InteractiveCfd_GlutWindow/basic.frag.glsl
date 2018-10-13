#version 430 core

in vec3 fColor;

out vec4 color;

void main()
{
    color = vec4(fColor, 1.0f);
    //Positions[i*6+1] = y+0.1f;
}