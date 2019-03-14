#version 430 core

#define WATER_REFRACTIVE_INDEX 1.33f
#define CAUSTICS_TEX_SIZE 1024.f

in vec3 fNormal;
in vec4 posInModel;
in vec4 fColor;

out vec4 color;

uniform vec3 cameraPos;

vec3 RefractRay(vec3 inRay, vec3 n)
{
    const float r = 1.0 / WATER_REFRACTIVE_INDEX;
    const float c = -(dot(n, inRay));
    return r*inRay + (r*c - sqrt(1.f - r*r*(1.f - c*c)))*n;
}

vec3 ReflectRay(vec3 inRay, vec3 n)
{
    return 2.f*dot(inRay, -1.f*n)*n + inRay;
}

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

    float lightAmbient = 0.5f;
    
    vec3 diffuse1  = 0.1f*cosTheta1*diffuseLightColor1;
    vec3 diffuse2  = 0.1f*cosTheta2*diffuseLightColor2;
    vec3 specular1 = cosAlpha*specularLightColor1;
    
    return min(vec3(1.f), (diffuse1.xyz + diffuse2.xyz + specular1.xyz + lightAmbient));
}

void main()
{
    if (posInModel.x > 1.f)
        discard;

    vec3 eyeRayInModel = posInModel.xyz - cameraPos;

    //const vec3 reflectedRay = ReflectRay(eyeRayInModel, fNormal);
    const float cosTheta = clamp(dot(eyeRayInModel, -1.f*fNormal),0,1);
    const float nu = 1.f / WATER_REFRACTIVE_INDEX;
    const float r0 = (nu - 1.f)*(nu - 1.f) / ((nu + 1.f)*(nu + 1.f));
    const float reflectedRayIntensity = r0 + (1.f - cosTheta)*(1.f - cosTheta)*(1.f - cosTheta)*(1.f - cosTheta)*(1.f - cosTheta)*(1.f - r0);
    
    const vec3 envColor = vec3(0.8);
    color.xyz = mix(fColor.xyz, envColor, reflectedRayIntensity);
}