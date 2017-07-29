#version 430 core
layout(location = 0) in vec3 position;
layout(location = 1) in float color;

out vec4 fColor;

uniform int xDimVisible;
uniform int yDimVisible;

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
    color.x = rf/256.0;
    color.y = gf/256.0;
    color.z = bf/256.0;
    color.w = af/256.0;

    return color;
}



void main()
{
    fColor = unpackColor(color);

    //fColor = vec4(1.f);

    //if (position.z > -0.99f)
    //{
        gl_Position = vec4(position,1.0f);


    gl_Position.x = position.x;
    gl_Position.y = (position.y+1.0f)/yDimVisible*xDimVisible-1.0f;

//        gl_Position.xyz = vec3(0.f);
//        gl_Position.z = 0.2f;
//        gl_Position.w = 1.f;
    //}
    //else
    //{
        //gl_Position = vec4(position,1.0f);
        //fColor.w = 0.0f;
    //}
}