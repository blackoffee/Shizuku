#version 430 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec4 fColor;
out vec4 fPositionEyeSpace;
out vec4 fNormalEyeSpace;
out vec4 eyeDirectionEyeSpace;

uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;
uniform mat4 modelInvTrans;

void main()
{
    fColor = vec4(0.8, 0.8, 0.8 ,1);
    fNormalEyeSpace = modelInvTrans*vec4(normal, 1.f);
    fPositionEyeSpace = modelMatrix*vec4(position,1.f);
    eyeDirectionEyeSpace = vec4(0) - fPositionEyeSpace;
    gl_Position = projectionMatrix*fPositionEyeSpace;
}