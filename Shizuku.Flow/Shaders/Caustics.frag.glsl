#version 430 core
in vec4 fColor;
in vec2 texCoord;

out vec4 color;

uniform sampler2D renderedTexture;

void main()
{
    //! NOTE - When writing to a floating format image, output colors are not clamped to [0,1];
    color = texture(renderedTexture, texCoord) * vec4(fColor).rgba;
}