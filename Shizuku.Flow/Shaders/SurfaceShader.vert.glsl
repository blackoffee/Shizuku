#version 430 core
layout(location = 0) in vec3 position;
layout(location = 1) in float color;
layout(location = 2) in vec3 normal;

out vec3 fNormal;
out float fWaterDepth;
out vec4 posInModel;

uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;


void main()
{
    posInModel = vec4(position, 1.f);
    gl_Position = projectionMatrix*modelMatrix*posInModel;

    fNormal = normal;
    fWaterDepth = position.z+1.f;
}