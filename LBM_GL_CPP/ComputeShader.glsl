#version 430 core
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
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) buffer vbo
{
    vec4 positions[];
};
layout(binding = 1) buffer ssbo_floor
{
	int floorLighting[];
};
layout(binding = 2) buffer ssbo_obsts
{
	Obstruction obsts[];
};

uniform int xDimVisible;
uniform int yDimVisible;
uniform int maxXDim;
uniform int maxYDim;
uniform int maxObsts;
uniform vec3 cameraPosition;
uniform int targetObstId;
uniform Obstruction targetObst;
uniform bool isObstOp = false;

subroutine void VboUpdate_t(uvec3 workUnit);
subroutine void ObstUpdate_t();

subroutine uniform VboUpdate_t VboUpdate;
subroutine uniform ObstUpdate_t ObstUpdate;


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

int FindOverlappingObstruction(const float x, const float y,
    const float tolerance = 0.f)
{
    for (int i = 0; i < 3; i++){
        if (obsts[i].state != 1) 
        {
            const float r1 = obsts[i].r1;
            if (obsts[i].shape == 0){
                if (abs(x - obsts[i].x)<r1 + tolerance &&
                    abs(y - obsts[i].y)<r1 + tolerance)
                    return i;
            }
            else if (obsts[i].shape == 1){//shift by 0.5 cells for better looks
                const float distFromCenter = (x + 0.5f - obsts[i].x)*(x + 0.5f - obsts[i].x)
                    + (y + 0.5f - obsts[i].y)*(y + 0.5f - obsts[i].y);
                if (distFromCenter<(r1+tolerance)*(r1+tolerance)+0.1f)
                    return i;
            }
            else if (obsts[i].shape == 2){
                if (abs(x - obsts[i].x)<r1*2+tolerance &&
                    abs(y - obsts[i].y)<1*0.501f+tolerance)
                    return i;
            }
            else if (obsts[i].shape == 3){
                if (abs(y - obsts[i].y)<r1*2+tolerance &&
                    abs(x - obsts[i].x)<1*0.501f+tolerance)
                    return i;
            }
        }
    }
    return -1;
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

    //if (FindOverlappingObstruction(x, y, 0.99f) > -1)
    //{
        const int obstID = FindOverlappingObstruction(float(x), float(y), 0.f);
        if (obstID >= 0)
        {
			lightFactor = 0.8f;
        	R = 255.f;
        	G = 255.f;
        	B = 255.f;
            if (obsts[obstID].state == 2)
            {
                zcoord = min(-0.3f, zcoord + 0.075f);
            }
            else if (obsts[obstID].state == 3)
            {
                zcoord = max(-1.f, zcoord - 0.075f);
            }
            else if (obsts[obstID].state == 0)
            {
                zcoord = -0.3f;
            }
            else
            {
                zcoord = -1.f;
            }
        }
        else
        {
            zcoord = -1.f;
        }

    //}
    //else
    //{
    //    zcoord = -1.f;
    //}
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

subroutine(VboUpdate_t) void UpdateObstructionTransientStates(uvec3 workUnit)
{
    const uint x = workUnit.x;
    const uint y = workUnit.y;
    const uint j = maxXDim*maxYDim + x + y*maxXDim;

    const float zcoord = positions[j].z;

    const int obstID = FindOverlappingObstruction(float(x), float(y));
    if (obstID >= 0)
    {
        if (zcoord > -0.29f)
        {
            obsts[obstID].state = 0;
        }
        if (zcoord < -0.99f)
        {
            obsts[obstID].state = 1;
        }
    }
}

subroutine(ObstUpdate_t) void UpdateObstruction()
{
	obsts[targetObstId].shape = targetObst.shape;
    obsts[targetObstId].r1 = targetObst.r1;
    obsts[targetObstId].x = targetObst.x;
    obsts[targetObstId].y = targetObst.y;
    obsts[targetObstId].u = targetObst.u;
    obsts[targetObstId].v = targetObst.v;
    obsts[targetObstId].state = targetObst.state;
}

subroutine(ObstUpdate_t) void DoNothing2()
{

}

subroutine(VboUpdate_t) void DoNothing(uvec3 workUnit)
{
}


void main()
{
	if (!isObstOp)
	{
		VboUpdate(gl_GlobalInvocationID);
	}
	else
	{
		//ObstUpdate();
		UpdateObstruction();
	}
	

}