#version 430 core
out vec4 color;

uniform vec4 MeshColor = vec4(0, 0.6, 0, 1);

void main()
{
	color = MeshColor;
}
