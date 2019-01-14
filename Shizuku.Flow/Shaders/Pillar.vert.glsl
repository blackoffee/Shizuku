#version 430 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec4 fPositionInModel;
out vec4 fNormalInModel;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 modelInvTrans;

void main()
{
    fNormalInModel = modelInvTrans*vec4(normal, 1.f);
    fPositionInModel = modelMatrix*vec4(position,1.f);
    gl_Position = projectionMatrix*viewMatrix*fPositionInModel;
}