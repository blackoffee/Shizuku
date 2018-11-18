#version 430 core
in vec4 fColor;

out vec4 color;

void main()
{
    if (fColor.a < 0.1f)
        discard;
    color = vec4(fColor);
}