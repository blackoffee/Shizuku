#version 430 core

#define WATER_REFRACTIVE_INDEX 1.33f
#define CAUSTICS_TEX_SIZE 1024.f
#define MAX_OBST 20
#define OBST_ALBEDO vec3(0.8f)

struct Obstruction
{
    int shape; // {SQUARE,CIRCLE,HORIZONTAL_LINE,VERTICAL_LINE};
    float x;
    float y;
    float r1;
    float r2;
    float u;
    float v;
    int state; // {ACTIVE,INACTIVE,NEW,REMOVED};
};

layout(binding = 0) buffer ssbo_obsts
{
    Obstruction obsts[];
};

in vec3 fNormal;
in vec4 posInModel;
in float fWaterDepth;

out vec4 color;

uniform vec3 cameraPos;
uniform vec2 viewSize;
uniform float obstHeight;

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

void Swap(inout float a, inout float b)
{
    const float temp = a;
    a = b;
    b = temp;
}

bool RayIntersectsWithBox(vec3 rayOrigin, vec3 rayDir, Obstruction obst, float boxHeight, out vec3 normal)
{
    const vec3 boxMin = vec3(obst.x-obst.r1, obst.y-obst.r1, -1.f);
    const vec3 boxMax = vec3(obst.x+obst.r1, obst.y+obst.r1, boxHeight-1.f);

    const vec3 t0 = (boxMin-rayOrigin)/rayDir;
    const vec3 t1 = (boxMax-rayOrigin)/rayDir;

    vec3 tMin = t0;
    vec3 tMax = t1;

    bool xFlag = false;
    bool yFlag = false;
    
    if (t0.x > t1.x)
    {
        Swap(tMin.x, tMax.x);
        xFlag = true;
    }
    if (t0.y > t1.y)
    {
        Swap(tMin.y, tMax.y);
        yFlag = true;
    }
    if (t0.z > t1.z)
    {
        Swap(tMin.z, tMax.z);
    }

    if (tMax.x < 0 || tMax.y < 0 || tMax.z < 0)
        return false;

    if (tMin.x > tMax.y || tMin.x > tMax.z || tMin.y > tMax.x || tMin.y > tMax.z || tMin.z > tMax.x || tMin.z > tMax.y)
        return false;

    if (tMin.x < tMin.y)
    {
        float xComp = xFlag? 1.f : -1.f;
        normal = vec3(xComp,0,0);
    }
    else
    {
        float yComp = yFlag? 1.f : -1.f;
        normal = vec3(0,yComp,0);
    }

    return true;
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

    float lightAmbient = 0.3f;
    
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

    for (int i = 0; i < MAX_OBST; ++i)
    {
        if (obsts[i].state == 0)
        {
            vec3 n;
            if (RayIntersectsWithBox(posInModel.xyz, refractedRay, obsts[i], obstHeight, n))
            {
                vec3 lightFactor = PhongLighting(posInModel.xyz, normalize(eyeRayInModel), n);
                color.xyz = lightFactor*vec3(0.8f);
            }
        }
    }

    if (color.a == 0.f)
        discard;
}