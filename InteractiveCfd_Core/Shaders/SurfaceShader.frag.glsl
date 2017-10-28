#version 430 core
in vec4 fColor;
in vec3 texCoords;

out vec4 color;

uniform sampler2D renderedTexture;


void main()
{
    color = vec4(fColor);
}