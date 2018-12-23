#version 430 core

#define WATER_REFRACTIVE_INDEX 1.33f
#define CAUSTICS_TEX_SIZE 1024.f

in vec3 fNormal;
in vec4 posInModel;
in float fWaterDepth;

out vec4 color;

uniform vec3 cameraPos;
uniform vec2 viewSize;

uniform sampler2D causticsTex;

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

void main()
{
    if (posInModel.x > 1.f)
        discard;

    //vec3 screenPos = float(gl_FragCoord.xyz)/vec3(viewSize.xy,1.f);
    //color = vec4(float(gl_FragCoord.x)/viewSize.x, float(gl_FragCoord.y)/viewSize.y, 0,1);

    vec3 eyeRayInModel = posInModel.xyz - cameraPos;

    const vec3 refractedRay = RefractRay(eyeRayInModel, fNormal);
    const vec3 reflectedRay = ReflectRay(eyeRayInModel, fNormal);
    const float cosTheta = clamp(dot(eyeRayInModel, -1.f*fNormal),0,1);
    const float nu = 1.f / WATER_REFRACTIVE_INDEX;
    const float r0 = (nu - 1.f)*(nu - 1.f) / ((nu + 1.f)*(nu + 1.f));
    const float reflectedRayIntensity = r0 + (1.f - cosTheta)*(1.f - cosTheta)*(1.f - cosTheta)*(1.f - cosTheta)*(1.f - cosTheta)*(1.f - r0);
    
    const vec2 delta = -refractedRay.xy*fWaterDepth / refractedRay.z;
    const vec2 floorPos = posInModel.xy+delta;
    const vec2 texCoord = 0.5f*(floorPos+vec2(1.f));

    color = texture(causticsTex, 0.5f*(floorPos+vec2(1.f)));
    const vec3 envColor = vec3(1);
    color.xyz = mix(color.xyz, envColor, reflectedRayIntensity);

    if (color.a == 0.f)
        discard;
}