#version 430 core
layout(location = 0) in vec3 position;

uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;

out vec2 modelPos;

void main()
{
	modelPos = position.xy;
    gl_Position = projectionMatrix*modelMatrix*vec4(position, 1.f);
}
