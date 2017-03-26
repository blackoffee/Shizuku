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
layout(binding = 0) buffer ssbo_fSource
{
    float fSource[];
};
layout(binding = 1) buffer ssbo_fTarget
{
    float fTarget[];
};
layout(binding = 2) buffer vbo
{
    vec4 positions[];
};
layout(binding = 3) buffer ssbo_floor
{
	int floorLighting[];
};
layout(binding = 4) buffer ssbo_obsts
{
	Obstruction obsts[];
};
uniform int xDim;
uniform int yDim;
uniform int xDimVisible;
uniform int yDimVisible;
uniform int maxXDim;
uniform int maxYDim;
uniform int maxObsts;
uniform vec3 cameraPosition;
uniform int targetObstId;
uniform Obstruction targetObst;
uniform bool isObstOp = false;
uniform float uMax;
uniform float omega;

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

void Swap(inout float a, inout float b)
{
    const float c = a;
    a = b;
    b = c;
}

void Normalize(inout vec3 n)
{
	n = normalize(n);
}

float DotProduct(const vec3 v1, const vec3 v2)
{
    return dot(v1,v2);
}

uint fMemory(const uint fNumb, const uint x, const uint y)
{
    return x+y*maxXDim+fNumb*maxXDim*maxYDim;
}

void ChangeCoordinatesToScaledFloat(inout float xcoord, inout float ycoord,
	const uint x, const uint y)
{
    xcoord = float(x) / (float(xDimVisible) *0.5f);
    ycoord = float(y) / (float(xDimVisible) *0.5f);
    xcoord -= 1.0;
    ycoord -= 1.0;
}

uint ImageFcn(const uint x, const uint y)
{
    if (x == 0)
    {
        return 3;// west
    }
    else if (x == xDim-1)
    {
        return 2;// east
    }
    else if (y == 0)
    {
        return 12;
    }
    else if (y == yDim-1)
    {
        return 11;
    }
    return 0;
}

void ComputeFEqs(inout float f[9], const float rho, const float u, const float v)
{
    const float usqr = u*u+v*v;
    f[0] = 0.4444444444f*(rho - 1.5f*usqr);
    f[1] = 0.1111111111f*(rho + 3.0f*u + 4.5f*u*u - 1.5f*usqr);
    f[2] = 0.1111111111f*(rho + 3.0f*v + 4.5f*v*v - 1.5f*usqr);
    f[3] = 0.1111111111f*(rho - 3.0f*u + 4.5f*u*u - 1.5f*usqr);
    f[4] = 0.1111111111f*(rho - 3.0f*v + 4.5f*v*v - 1.5f*usqr);
    f[5] = 0.02777777778*(rho + 3.0f*(u + v) + 4.5f*(u + v)*(u + v) - 1.5f*usqr);
    f[6] = 0.02777777778*(rho + 3.0f*(-u + v) + 4.5f*(-u + v)*(-u + v) - 1.5f*usqr);
    f[7] = 0.02777777778*(rho + 3.0f*(-u - v) + 4.5f*(-u - v)*(-u - v) - 1.5f*usqr);
    f[8] = 0.02777777778*(rho + 3.0f*(u - v) + 4.5f*(u - v)*(u - v) - 1.5f*usqr);   
}

subroutine(VboUpdate_t) void InitializeDomain(uvec3 workUnit)
{
	uint x = workUnit.x;
	uint y = workUnit.y;
	uint z = workUnit.z;
 
    float fEq[9];
    float rho = 1.f;
    float u = uMax;
    float v = 0.f;
    ComputeFEqs(fEq, rho, u, v);
    for (uint i = 0; i < 9; i++)
    {
        fSource[fMemory(i, x, y)] = fEq[i];
        fTarget[fMemory(i, x, y)] = fEq[i];
    }
}

void DirichletWest(inout float f[9], const uint y)
{
    if (y == 0){
        f[2] = f[4];
        f[6] = f[7];
    }
    else if (y == yDim - 1){
        f[4] = f[2];
        f[7] = f[6];
    }
    float u, v;
    u = uMax;
    v = 0.0f;
    f[1] = f[3] + u*0.66666667f;
    f[5] = f[7] - 0.5f*(f[2] - f[4]) + v*0.5f + u*0.166666667f;
    f[8] = f[6] + 0.5f*(f[2] - f[4]) - v*0.5f + u*0.166666667f;
}

void NeumannEast(inout float f[9], const uint y)
{
    if (y == 0){
        f[2] = f[4];
        f[5] = f[8];
    }
    else if (y == yDim - 1){
        f[4] = f[2];
        f[8] = f[5];
    }
    float rho, u, v;
    v = 0.0f;
    rho = 1.f;
    u = -rho + ((f[0] + f[2] + f[4]) + 2.0f*f[1] + 2.0f*f[5] + 2.0f*f[8]);
    f[3] = f[1] - u*0.66666667f;
    f[7] = f[5] + 0.5f*(f[2] - f[4]) - v*0.5f - u*0.166666667f;
    f[6] = f[8] - 0.5f*(f[2] - f[4]) + v*0.5f - u*0.166666667f;
}

void ApplyBCs(inout float f[9], const uint x, const uint y)
{
    uint im = ImageFcn(x,y);
    if (im == 2)//NeumannEast
    {
        NeumannEast(f, y);
    }
    else if (im == 3)//DirichletWest
    {
        DirichletWest(f, y);
    }
    else if (im == 11)//xsymmetry
    {
        f[4] = f[2];
        f[7] = f[6];
        f[8] = f[5];
    }
    else if (im == 12)//xsymmetry
    {
        f[2] = f[4];
        f[6] = f[7];
        f[5] = f[8];
    }  
}

void BounceBackWall(inout float f[9])
{
    Swap(f[1], f[3]);
    Swap(f[2], f[4]);
    Swap(f[5], f[7]);
    Swap(f[6], f[8]);
}

void ReadDistributions(inout float f[9], const uint x, const uint y)
{
    for (uint i = 0; i < 9; i++)
    {
        f[i] = fSource[fMemory(i, x, y)];
    }
}

void ReadIncomingDistributions(inout float f[9], const uint x, const uint y)
{
    uint j = x + y*maxXDim;
    f[0] = fSource[j];
    f[1] = fSource[fMemory(1, max(x - 1, 0), y)];
    f[3] = fSource[fMemory(3, min(x + 1, maxXDim), y)];
    f[2] = fSource[fMemory(2, x, y - 1)];
    f[5] = fSource[fMemory(5, max(x - 1, 0), y - 1)];
    f[6] = fSource[fMemory(6, min(x + 1, maxXDim), y - 1)];
    f[4] = fSource[fMemory(4, x, y + 1)];
    f[7] = fSource[fMemory(7, min(x + 1, maxXDim), y + 1)];
    f[8] = fSource[fMemory(8, max(x - 1, 0), min(y + 1, maxXDim))];
}

float ComputeStrainRateMagnitude(float f[9])
{
    const float rho = f[0] + f[1] + f[2] + f[3]
        + f[4] + f[5] + f[6] + f[7] + f[8];
    const float u = f[1] - f[3] + f[5] - f[6]
        - f[7] + f[8];
    const float v = f[2] - f[4] + f[5] + f[6]
        - f[7] - f[8];
    float fEq[9];
    ComputeFEqs(fEq, rho, u, v);
    float qxx = (f[1]-fEq[1]) + (f[3]-fEq[3]) + (f[5]-fEq[5]) + (f[6]-fEq[6])
        + (f[7]-fEq[7]) + (f[8]-fEq[8]);
    float qxy = (f[5]-fEq[5]) - (f[6]-fEq[6]) + (f[7]-fEq[7]) - (f[8]-fEq[8]) ;
    float qyy = (f[5]-fEq[5]) + (f[2]-fEq[2]) + (f[6]-fEq[6]) + (f[7]-fEq[7])
        + (f[4]-fEq[4]) + (f[8]-fEq[8]);
    return sqrt(qxx*qxx + qxy*qxy * 2 + qyy*qyy);
}

void Collide(inout float f[9])
{
    const float Q = ComputeStrainRateMagnitude(f);
    const float tau0 = 1.f / omega;
    const float smagConst = 0.1f;
    const float tau = 0.5f*tau0 + 0.5f*sqrt(tau0*tau0 + 18.f*sqrt(2.f)*Q);
    const float omegaTurb = 1.f / tau;

    float m1, m2, m4, m6, m7, m8;

    const float rho = f[0] + f[1] + f[2] + f[3]
        + f[4] + f[5] + f[6] + f[7] + f[8];
    const float u = f[1] - f[3] + f[5] - f[6]
        - f[7] + f[8];
    const float v = f[2] - f[4] + f[5] + f[6]
        - f[7] - f[8];

    m1 = -2.f*f[0] + f[1] + f[2] + f[3] + f[4] + 4.f*f[5] + 4.f*f[6] + 4.f*f[7]
        + 4.f*f[8] - 3.0f*(u*u + v*v);
    m2 = 3.f*f[0] - 3.f*f[1] - 3.f*f[2] - 3.f*f[3] - 3.f*f[4] + 3.0f*(u*u + v*v); //ep
    m4 = -f[1] + f[3] + 2.f*f[5] - 2.f*f[6] - 2.f*f[7] + 2.f*f[8];;//qx_eq
    m6 = -f[2] + f[4] + 2.f*f[5] + 2.f*f[6] - 2.f*f[7] - 2.f*f[8];;//qy_eq
    m7 = f[1] - f[2] + f[3] - f[4] - (u*u - v*v);//pxx_eq
    m8 = f[5] - f[6] + f[7] - f[8] - (u*v);//pxy_eq

    f[0] = f[0] - (-m1 + m2)*0.11111111f;
    f[1] = f[1] - (-m1*0.027777777f - 0.05555555556f*m2 - 0.16666666667f*m4 + m7*omegaTurb*0.25f);
    f[2] = f[2] - (-m1*0.027777777f - 0.05555555556f*m2 - 0.16666666667f*m6 - m7*omegaTurb*0.25f);
    f[3] = f[3] - (-m1*0.027777777f - 0.05555555556f*m2 + 0.16666666667f*m4 + m7*omegaTurb*0.25f);
    f[4] = f[4] - (-m1*0.027777777f - 0.05555555556f*m2 + 0.16666666667f*m6 - m7*omegaTurb*0.25f);
    f[5] = f[5] - (0.05555555556f*m1 + m2*0.027777777f + 0.08333333333f*m4 + 0.08333333333f*m6
        + m8*omegaTurb*0.25f);
    f[6] = f[6] - (0.05555555556f*m1 + m2*0.027777777f - 0.08333333333f*m4 + 0.08333333333f*m6
        - m8*omegaTurb*0.25f);
    f[7] = f[7] - (0.05555555556f*m1 + m2*0.027777777f - 0.08333333333f*m4 - 0.08333333333f*m6
        + m8*omegaTurb*0.25f);
    f[8] = f[8] - (0.05555555556f*m1 + m2*0.027777777f + 0.08333333333f*m4 - 0.08333333333f*m6
        - m8*omegaTurb*0.25f);
}



subroutine(VboUpdate_t) void MarchLbm(uvec3 workUnit)
{
 	uint x = workUnit.x;
	uint y = workUnit.y;
	uint z = workUnit.z;
    uint j = x + y * maxXDim;
    int obstId = FindOverlappingObstruction(float(x), float(y), 0.f);
    float fTemp[9];
    ReadIncomingDistributions(fTemp, x, y); 
    if (obstId >= 0)
    {
        BounceBackWall(fTemp);
    }
    else
    {
        ApplyBCs(fTemp, x, y);
        Collide(fTemp);
    }

    for (uint i = 0; i < 9; i++)
    {
        fTarget[fMemory(i, x, y)] = fTemp[i];
    }

}

subroutine(VboUpdate_t) void UpdateFluidVbo(uvec3 workUnit)
{
 	uint x = workUnit.x;
	uint y = workUnit.y;
	uint z = workUnit.z;
    uint j = x + y * maxXDim;
    float fTemp[9];
    ReadDistributions(fTemp, x, y);
  
    const float rho = fTemp[0] + fTemp[1] + fTemp[2] + fTemp[3]
        + fTemp[4] + fTemp[5] + fTemp[6] + fTemp[7] + fTemp[8];
    const float u = fTemp[1] - fTemp[3] + fTemp[5] - fTemp[6]
        - fTemp[7] + fTemp[8];
    const float v = fTemp[2] - fTemp[4] + fTemp[5] + fTemp[6]
        - fTemp[7] - fTemp[8];
    float xcoord, ycoord;
    ChangeCoordinatesToScaledFloat(xcoord, ycoord, x, y);
    const float zcoord = (rho-1.f)-0.5f;

    positions[j].x = xcoord;
    positions[j].y = ycoord;
    positions[j].z = zcoord;
    vec4 color = vec4(100.f/255.f, 150.f/255.f, 1.f, 100.f/255.f);
    if (FindOverlappingObstruction(x, y, 1.f) >= 0)
    {
        positions[j].z = 0.f;
        color = vec4(0.f, 1.f, 0.f, 1.f);
    }
    positions[j].w = packColor(color);

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

vec2 ComputePositionOfLightOnFloor(vec3 incidentLight, const uint x, const uint y)
{
    vec3 n = vec3( 0.f, 0.f, 1.f );
    float slope_x = 0.f;
    float slope_y = 0.f;
    float cellSize = 2.f / xDimVisible;
    if (x > 0 && x < (xDimVisible - 1) && y > 0 && y < (yDimVisible - 1))
    {
        slope_x = (positions[(x + 1) + y*maxXDim].z - positions[(x - 1) + y*maxXDim].z) /
            (2.f*cellSize);
        slope_y = (positions[(x)+(y + 1)*maxXDim].z - positions[(x)+(y - 1)*maxXDim].z) /
            (2.f*cellSize);
        n.x = -slope_x*2.f*cellSize*2.f*cellSize;
        n.y = -slope_y*2.f*cellSize*2.f*cellSize;
        n.z = 2.f*cellSize*2.f*cellSize;
    }
    Normalize(n);

    Normalize(incidentLight);
    float waterDepth = 80.f;

    vec3 refractedLight;
    vec3 r = vec3(1.0f / 1.3f);
    vec3 c = vec3(-(DotProduct(n, incidentLight)));
    refractedLight = r*incidentLight + (r*c - sqrt(vec3(1.f) - r*r*(vec3(1.f) - c*c)))*n;

    float dx = -refractedLight.x*(positions[(x)+(y)*maxXDim].z + 1.f)*waterDepth 
        / refractedLight.z;
    float dy = -refractedLight.y*(positions[(x)+(y)*maxXDim].z + 1.f)*waterDepth 
        / refractedLight.z;

    return vec2( float(x) + dx, float(y) + dy );
}

subroutine(VboUpdate_t) void DeformFloorMeshUsingCausticRay(uvec3 workUnit)
{
    const uint x = workUnit.x;
    const uint y = workUnit.y;
    const uint j = x + y*maxXDim + maxXDim*maxYDim;
    vec3 incidentLight = vec3(-0.25f, -0.25f, -1.f);

    if (x < xDimVisible && y < yDimVisible)
    {
        vec2 lightPositionOnFloor;
        if (FindOverlappingObstruction(x, y, 1.f) >= 0)
        {
            lightPositionOnFloor = vec2(x, y);
        }
        else
        {
            lightPositionOnFloor = ComputePositionOfLightOnFloor(incidentLight, x, y);
        }

        positions[j].x = lightPositionOnFloor.x;
        positions[j].y = lightPositionOnFloor.y;
    }
}

subroutine(VboUpdate_t) void ComputeFloorLightIntensitiesFromMeshDeformation(uvec3 workUnit)
{
    const uint x = workUnit.x;
    const uint y = workUnit.y;

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

    ChangeCoordinatesToScaledFloat(xcoord, ycoord, x, y);
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