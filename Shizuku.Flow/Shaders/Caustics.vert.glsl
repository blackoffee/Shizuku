#version 430 core
layout(location = 0) in vec3 position;
layout(location = 1) in float color;

out vec4 fColor;
out vec2 texCoord;

uniform float texCoordScale;

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
    fColor = unpackColor(color);

    gl_Position = vec4(position,1.0f);

    texCoord = texCoordScale * vec2(0.5f*(position.x+1.0f), 0.5f*(position.y+1.0f));
}