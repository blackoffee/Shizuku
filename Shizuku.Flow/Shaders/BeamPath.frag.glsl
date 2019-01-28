#version 430 core
in vec4 fColor;
out vec4 color;

void main()
{
	color = fColor;
	//color = vec4(0, 0, 1, 1);
}
