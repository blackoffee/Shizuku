#version 430 core
in vec4 fColor;
in vec4 fPositionEyeSpace;
in vec4 fNormalEyeSpace;
in vec4 eyeDirectionEyeSpace;

out vec4 color;

void main()
{
    vec3 n = normalize(fNormalEyeSpace.xyz);

    vec3 diffuseLightDirection1 = vec3(0.577367f, 0.577367f, -0.577367f );
    vec3 diffuseLightDirection2 = vec3( -0.577367f, 0.577367f, -0.577367f );
    vec3 eyeDirection = vec3(0) - fPositionEyeSpace.xyz;
    vec3 diffuseLightColor1 = vec3(0.5f, 0.5f, 0.5f);
    vec3 diffuseLightColor2 = vec3(0.5f, 0.5f, 0.5f);
    vec3 specularLightColor1 = vec3(0.5f, 0.5f, 0.5f);

    float cosTheta1 = clamp(dot(n,diffuseLightDirection1),0,1);
    float cosTheta2 = clamp(dot(n,diffuseLightDirection2),0,1);

    vec3 specularLightPosition1 = vec3(-1.5f, -1.5f, 1.5f);
    vec3 specularLight1 = fPositionEyeSpace.xyz - specularLightPosition1;
    vec3 specularRefection1 = specularLight1 - 2.f*(dot(specularLight1, n)*n);
    float cosAlpha = clamp(dot(normalize(eyeDirection), normalize(specularRefection1)),0,1);
    cosAlpha = pow(cosAlpha, 5.f);

    float lightAmbient = 0.3f;
    
    vec3 diffuse1  = 0.1f*cosTheta1*diffuseLightColor1;
    vec3 diffuse2  = 0.1f*cosTheta2*diffuseLightColor2;
    vec3 specular1 = cosAlpha*specularLightColor1;
    
    vec4 lightFactor;
    lightFactor.xyz = min(vec3(1.f,1.f,1.f), (diffuse1.xyz + diffuse2.xyz + specular1.xyz + lightAmbient));
    lightFactor.w = 1.f;

    color = lightFactor * fColor;

}