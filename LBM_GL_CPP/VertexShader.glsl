#version 430 core
layout(location = 0) in vec3 position;
layout(location = 1) in float color;

out vec3 fColor;

uniform vec4 viewportMatrix;
uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;

vec3 unpackColor(uint f)
{
//    char b[];
//    std::memcpy(&f, &b, sizeof(f));
//    return (b[0], b[1], b[2]);

    uint r = (f & uint(0xFF000000)) >> 24;
    uint g = (f & uint(0x00FF0000)) >> 16;
    uint b = (f & uint(0x0000FF00)) >> 8;
    uint a = (f & uint(0x000000FF));

 //   uint r = (f & uint(0x000000FF));
 //   uint g = (f & uint(0x0000FF00)) >> 8;
 //   uint b = (f & uint(0x00FF0000)) >> 16;
 //   uint a = (f & uint(0xFF000000));


    float rf = float(r);
    float gf = float(g);
    float bf = float(b);
    vec3 color;
    color.x = rf/256.0;
    color.y = gf/256.0;
    color.z = bf/256.0;
//    color.x = r/256.f;// floor(f / 256.0 / 256.0 / 256.0);
//    color.y = g/256.f;// floor((f - color.r * 256.0 * 256.0 * 256.0) / 256.0 / 256.f);
//    color.z = b/256.f;//floor(f - color.r * 255.0 * 256.0 * 256.0 - color.g * 256.0 * 256.0);

    return color;
}
vec3 unpackColor2(float f)
{
//    char b[];
//    std::memcpy(&f, &b, sizeof(f));
//    return (b[0], b[1], b[2]);

    uint f2 = floatBitsToUint(f);

    uint r = (f2 & uint(0xFF000000)) >> 24;
    uint g = (f2 & uint(0x00FF0000)) >> 16;
    uint b = (f2 & uint(0x0000FF00)) >> 8;
    uint a = (f2 & uint(0x000000FF));

 //   uint r = (f & uint(0x000000FF));
 //   uint g = (f & uint(0x0000FF00)) >> 8;
 //   uint b = (f & uint(0x00FF0000)) >> 16;
 //   uint a = (f & uint(0xFF000000));


    float rf = float(r);
    float gf = float(g);
    float bf = float(b);
    vec3 color;
    color.x = rf/256.0;
    color.y = gf/256.0;
    color.z = bf/256.0;
//    float f3 = float(f2);
//    color.x = floor(f3 / 256.0 / 256.0 / 256.0);
//    color.y = floor(f3 - color.x * 256.0 * 256.0 * 256.0) / 256.0 / 256.f;
//    color.z = floor(f3 - color.x * 255.0 * 256.0 * 256.0 - color.y * 256.0 * 256.0)/256.f;

    return color;
}

void main()
{
    
    gl_Position = projectionMatrix*modelMatrix*vec4(position, 1.f);

    //gl_Position = vec4(position, 1.0f);
    //gl_Position.y = position.y+0.3f*sin(3.14f*(position.x+1.f))*sin(time*0.3f);

    vec3 unpackedColor = unpackColor2(color);

    fColor.x = unpackedColor.x;
    fColor.y = unpackedColor.y;// color.y*sin((time + position.x + 1.f)*0.3f);
    fColor.z = unpackedColor.z;//color.z*sin((time + position.x + 1.f)*0.3f);
}