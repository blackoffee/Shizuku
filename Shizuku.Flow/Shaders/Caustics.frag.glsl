#version 430 core
in vec4 fColor;
in vec2 texCoord;

out vec4 color;

uniform sampler2D renderedTexture;

void main()
{
    color = texture(renderedTexture, texCoord) * vec4(fColor).rgba;
}