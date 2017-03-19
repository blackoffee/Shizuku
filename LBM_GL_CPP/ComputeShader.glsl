#version 430 core
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) buffer vbo
{
    vec4 positions[];
};
layout(binding = 1) buffer ssbo_floor
{
	int floorLighting[];
};

uniform int xDimVisible;
uniform int yDimVisible;
uniform int maxXDim;
uniform int maxYDim;
uniform vec3 cameraPosition;

subroutine void VboUpdate_t(uvec3 workUnit);
subroutine void VboUpdate2_t();

subroutine uniform VboUpdate_t VboUpdate;
subroutine uniform VboUpdate2_t VboUpdate2;


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
    color.x = rf/255.0;
    color.y = gf/255.0;
    color.z = bf/255.0;
    color.w = af/255.0;

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
	n = normalize(n);
}

float DotProduct(vec3 v1, vec3 v2)
{
    return dot(v1,v2);
}

subroutine(VboUpdate_t) void PhongLighting(uvec3 workUnit)
{
	uint x = workUnit.x;
	uint y = workUnit.y;
	uint z = workUnit.z;
    uint j = x + y * maxXDim + z * maxXDim * maxYDim;

    float slope_x = 0.f;
    float slope_y = 0.f;
    float cellSize = 2.f / xDimVisible;
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
    
    vec4 lightFactor;
    lightFactor.xyz = min(vec3(1.f,1.f,1.f), (diffuse1.xyz + diffuse2.xyz + specular1.xyz + lightAmbient));
	lightFactor.w = 1.f;

    vec4 unpackedColor = unpackColor(positions[j].w);
    vec4 finalColor;
    finalColor.xyzw = unpackedColor.xyzw*lightFactor.xyzw;

    positions[j].w = packColor(finalColor);
}

float CrossProductArea(const vec2 u, const vec2 v)
{
    return 0.5f*sqrt((u.x*v.y-u.y*v.x)*(u.x*v.y-u.y*v.x));
}

float ComputeAreaFrom4Points(const vec2 nw, const vec2 ne,
    const vec2 sw, const vec2 se)
{
    vec2 vecN = ne - nw;
    vec2 vecS = se - sw;
    vec2 vecE = ne - se;
    vec2 vecW = nw - sw;
    return CrossProductArea(vecN, vecW) + CrossProductArea(vecE, vecS);
}

subroutine(VboUpdate_t) void ComputeFloorLightIntensitiesFromMeshDeformation(uvec3 workUnit)
{
    uint x = workUnit.x;
    uint y = workUnit.y;

	if (x < xDimVisible-1 && y < yDimVisible-1)
	{
		vec2 nw, ne, sw, se;
    	const int offset = maxXDim*maxYDim;
    	nw = vec2(positions[(x  )+(y+1)*maxXDim+offset].x, positions[(x  )+(y+1)*maxXDim+offset].y);
    	ne = vec2(positions[(x+1)+(y+1)*maxXDim+offset].x, positions[(x+1)+(y+1)*maxXDim+offset].y);
    	sw = vec2(positions[(x  )+(y  )*maxXDim+offset].x, positions[(x  )+(y  )*maxXDim+offset].y);
    	se = vec2(positions[(x+1)+(y  )*maxXDim+offset].x, positions[(x+1)+(y  )*maxXDim+offset].y);

    	const float areaOfLightMeshOnFloor = ComputeAreaFrom4Points(nw, ne, sw, se);
    	const float lightIntensity = 0.3f / areaOfLightMeshOnFloor;
		const int int_lightIntensity = int(lightIntensity*0.25f*1000000.f);
    	atomicAdd(floorLighting[x   + (y  )*maxXDim], int_lightIntensity);
    	atomicAdd(floorLighting[x+1 + (y  )*maxXDim], int_lightIntensity);
    	atomicAdd(floorLighting[x+1 + (y+1)*maxXDim], int_lightIntensity);
    	atomicAdd(floorLighting[x   + (y+1)*maxXDim], int_lightIntensity);
	}
}

void ChangeCoordinatesToScaledFloat(inout float xcoord, inout float ycoord,
	const uint x, const uint y, const int xDimVisible, const int yDimVisible)
{
    xcoord = float(x) / (float(xDimVisible) *0.5f);
    ycoord = float(y) / (float(xDimVisible) *0.5f);
    xcoord -= 1.0;
    ycoord -= 1.0;
}

subroutine(VboUpdate_t) void ApplyCausticLightingToFloor(uvec3 workUnit)
{
    uint x = workUnit.x;
    uint y = workUnit.y;
    uint j = maxXDim*maxYDim + x + y*maxXDim;
    float xcoord, ycoord, zcoord;

    xcoord = positions[j].x;
    ycoord = positions[j].y;
    zcoord = positions[j].z;

    float lightFactor = min(1.f,float(floorLighting[x + y*maxXDim])/1000000.f);
    floorLighting[x + y*maxXDim] = 0;

    float R = 50.f;
    float G = 120.f;
    float B = 255.f;
    float A = 255.f;

//    if (IsInsideObstruction(x, y, obstructions, 0.99f))
//    {
//        int obstID = FindOverlappingObstruction(x, y, obstructions,0.f);
//        if (obstID >= 0)
//        {
//            if (obstructions[obstID].state == Obstruction::NEW)
//            {
//                zcoord = dmin(-0.3f, zcoord + 0.075f);
//            }
//            else if (obstructions[obstID].state == Obstruction::REMOVED)
//            {
//                zcoord = dmax(-1.f, zcoord - 0.075f);
//            }
//            else if (obstructions[obstID].state == Obstruction::ACTIVE)
//            {
//                zcoord = -0.3f;
//            }
//            else
//            {
//                zcoord = -1.f;
//            }
//        }
//        else
//        {
//            zcoord = -1.f;
//        }
//        lightFactor = 0.8f;
//        R = 255.f;
//        G = 255.f;
//        B = 255.f;
//    }
//    else
//    {
        zcoord = -1.f;
//    }
    R *= lightFactor;
    G *= lightFactor;
    B *= lightFactor;

	vec4 color;
	color.x = R/255.f;
	color.y = G/255.f;
	color.z = B/255.f;
	color.w = A/255.f;

    ChangeCoordinatesToScaledFloat(xcoord, ycoord, x, y, xDimVisible, yDimVisible);
    positions[j].x = xcoord;
    positions[j].y = ycoord;
    positions[j].z = zcoord;
    positions[j].w = packColor(color);
}



subroutine(VboUpdate_t) void DoNothing(uvec3 workUnit)
{
}

subroutine(VboUpdate2_t) void DoNothing2()
{
}

void main()
{

	VboUpdate(gl_GlobalInvocationID);

}