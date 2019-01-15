#version 430 core
layout(location = 0) in vec3 position;
layout(location = 1) in float color;
layout(location = 2) in vec3 normal;

out vec3 fNormal;
out float fWaterDepth;
out vec4 posInModel;
out vec4 fColor;

uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;

vec4 unpackColor(float f)
{
    uint f2 = floatBitsToUint(f);

    uint r = (f2 & uint(0x000000FF));
    uint g = (f2 & uint(0x0000FF00)) >> 8;
    uint b = (f2 & uint(0x00FF0000)) >> 16;
    uint a = (f2 & uint(0xFF000000)) >> 24;

    float rf = float(r);
    float gf = float(g);
    float bf = float(b);
    float af = float(a);
    vec4 color;
    color.x = rf/255.f;
    color.y = gf/255.f;
    color.z = bf/255.f;
    color.w = af/255.f;

    return color;
}

void main()
{
    posInModel = vec4(position, 1.f);
    gl_Position = projectionMatrix*modelMatrix*posInModel;
	fColor = unpackColor(color);

    fNormal = normal;
    fWaterDepth = position.z+1.f;
}
