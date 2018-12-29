#version 430 core

#define OBST_ALBEDO vec3(0.8f)

in vec4 fPositionInModel;
in vec4 fNormalInModel;

out vec4 color;

uniform vec3 cameraPosition;

vec3 PhongLighting(vec3 posInModel, vec3 eyeDir, vec3 n)
{
    vec3 diffuseLightDirection1 = vec3(0.577367, 0.577367, -0.577367 );
    vec3 diffuseLightDirection2 = vec3( -0.577367, 0.577367, -0.577367 );
    vec3 diffuseLightColor1 = vec3(0.5f, 0.5f, 0.5f);
    vec3 diffuseLightColor2 = vec3(0.5f, 0.5f, 0.5f);
    vec3 specularLightColor1 = vec3(0.5f, 0.5f, 0.5f);

    float cosTheta1 = -dot(n,diffuseLightDirection1);
    cosTheta1 = cosTheta1 < 0 ? 0 : cosTheta1;
    float cosTheta2 = -dot(n, diffuseLightDirection2);
    cosTheta2 = cosTheta2 < 0 ? 0 : cosTheta2;

    vec3 specularLightPosition1 = vec3(-1.5f, -1.5f, 1.5f);
    vec3 specularLight1 = posInModel - specularLightPosition1;
    vec3 specularRefection1 = normalize(specularLight1 - 2.f*(dot(specularLight1, n)*n));
    float cosAlpha = -dot(eyeDir, specularRefection1);
    cosAlpha = cosAlpha < 0 ? 0 : cosAlpha;
    cosAlpha = pow(cosAlpha, 5.f);

    float lightAmbient = 0.3f;
    
    vec3 diffuse1  = 0.1f*cosTheta1*diffuseLightColor1;
    vec3 diffuse2  = 0.1f*cosTheta2*diffuseLightColor2;
    vec3 specular1 = cosAlpha*specularLightColor1;
    
    return min(vec3(1.f), (diffuse1.xyz + diffuse2.xyz + specular1.xyz + lightAmbient));
}

void main()
{
    vec3 n = normalize(fNormalInModel.xyz);
    vec3 eyeRayInModel = fPositionInModel.xyz - cameraPosition;
    vec3 lightFactor = PhongLighting(fPositionInModel.xyz, normalize(eyeRayInModel), n);
    color.xyz = lightFactor * OBST_ALBEDO;
}