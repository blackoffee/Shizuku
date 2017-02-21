#version 430 core
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) buffer vbo
{
    vec4 positions[];
};

uniform int xDimVisible;
uniform int yDimVisible;
uniform int maxXDim;
uniform int maxyDim;
uniform vec3 cameraPosition;

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

float packColor(vec4 color)
{
    uint r = uint(floor(color.x*255));
    uint g = uint(floor(color.y*255))<<8;
    uint b = uint(floor(color.z*255))<<16;
    uint a = uint(floor(color.w*255))<<24;
    uint colorOut = r|g|b|a;

    return uintBitsToFloat(colorOut);
}

void Normalize(inout vec3 n)
{
    float mag = sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
    n /= mag;
}

float DotProduct(vec3 v1, vec3 v2)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

vec3 PhongLighting()
{
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint j = x + y * maxXDim;

    float slope_x = 0.f;
    float slope_y = 0.f;
    float cellSize = 2.f / 128;
    vec3 n;

    slope_x = (positions[(x + 1) + y*maxXDim].z - positions[(x - 1) + y*maxXDim].z) /
        (2.f*cellSize);
    slope_y = (positions[(x)+(y + 1)*maxXDim].z - positions[(x)+(y - 1)*maxXDim].z) /
        (2.f*cellSize);
    n.x = -slope_x*2.f*cellSize*2.f*cellSize;
    n.y = -slope_y*2.f*cellSize*2.f*cellSize;
    n.z = 2.f*cellSize*2.f*cellSize;

    Normalize(n);

    vec3 elementPosition = vec3(positions[j].x,positions[j].y,positions[j].z );
    vec3 diffuseLightDirection1 = vec3(0.577367, 0.577367, -0.577367 );
    vec3 diffuseLightDirection2 = vec3( -0.577367, 0.577367, -0.577367 );
    vec3 eyeDirection = elementPosition - cameraPosition;
    vec3 diffuseLightColor1 = vec3(0.5f, 0.5f, 0.5f);
    vec3 diffuseLightColor2 = vec3(0.5f, 0.5f, 0.5f);
    vec3 specularLightColor1 = vec3(0.5f, 0.5f, 0.5f);

    float cosTheta1 = -DotProduct(n,diffuseLightDirection1);
    cosTheta1 = cosTheta1 < 0 ? 0 : cosTheta1;
    float cosTheta2 = -DotProduct(n, diffuseLightDirection2);
    cosTheta2 = cosTheta2 < 0 ? 0 : cosTheta2;

    vec3 specularLightPosition1 = vec3(-1.5f, -1.5f, 1.5f);
    vec3 specularLight1 = elementPosition - specularLightPosition1;
    vec3 specularRefection1 = specularLight1 - 2.f*(DotProduct(specularLight1, n)*n);
    Normalize(specularRefection1);
    Normalize(eyeDirection);
    float cosAlpha = -DotProduct(eyeDirection, specularRefection1);
    cosAlpha = cosAlpha < 0 ? 0 : cosAlpha;
    cosAlpha = pow(cosAlpha, 5.f);

    float lightAmbient = 0.3f;
    
    vec3 diffuse1  = 0.1f*cosTheta1*diffuseLightColor1;
    vec3 diffuse2  = 0.1f*cosTheta2*diffuseLightColor2;
    vec3 specular1 = cosAlpha*specularLightColor1;
    
    vec3 lightFactor;
    lightFactor.x = min(1.f, (diffuse1.x + diffuse2.x + specular1.x + lightAmbient));
    lightFactor.y = min(1.f, (diffuse1.y + diffuse2.y + specular1.y + lightAmbient));
    lightFactor.z = min(1.f, (diffuse1.z + diffuse2.z + specular1.z + lightAmbient));

    return lightFactor;
}


void main()
{
    uint xi = gl_GlobalInvocationID.x;
    uint yi = gl_GlobalInvocationID.y;
    uint j = xi + yi*maxXDim;

    vec4 unpackedColor = unpackColor(positions[j].w);

    vec3 colorFactor = PhongLighting();

    vec4 finalColor;

    finalColor.x = unpackedColor.x*colorFactor.x;
    finalColor.y = unpackedColor.y*colorFactor.y;
    finalColor.z = unpackedColor.z*colorFactor.z;
    finalColor.w = unpackedColor.w;

    float x = positions[j].x;
    float y = positions[j].y;
    float z = positions[j].z;
    float w = packColor(finalColor);

    positions[j] = vec4(x,y,z,w);

}