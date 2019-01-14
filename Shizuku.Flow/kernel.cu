#define OBST_HEIGHT 0.8f
#define WATER_REFRACTIVE_INDEX 1.33f

#include "kernel.h"
#include "LbmNode.h"
#include "CudaCheck.h"
#include "VectorUtils.h"
#include "Graphics/CudaLbm.h"
#include "Graphics/ObstDefinition.h"

using namespace Shizuku::Flow;

/*----------------------------------------------------------------------------------------
 *	Device functions
 */

__global__ void UpdateObstructions(ObstDefinition* obstructions, const int obstNumber,
    const ObstDefinition newObst)
{
    obstructions[obstNumber].shape = newObst.shape;
    obstructions[obstNumber].r1 = newObst.r1;
    obstructions[obstNumber].x = newObst.x;
    obstructions[obstNumber].y = newObst.y;
    obstructions[obstNumber].u = newObst.u;
    obstructions[obstNumber].v = newObst.v;
    obstructions[obstNumber].state = newObst.state;
}

__device__ bool GetCoordFromRayHitOnObst(float3 &intersect, const float3 rayOrigin, const float3 rayDest,
    ObstDefinition* obstructions, float obstHeight, const float tolerance = 0.f)
{
    float3 rayDir = rayDest - rayOrigin;
    bool hit = false;
    for (int i = 0; i < MAXOBSTS; i++){
        if (obstructions[i].state == State::NORMAL)
        {
            const float3 obstLineP1 = { obstructions[i].x, obstructions[i].y, -1.f };
            const float3 obstLineP2 = { obstructions[i].x, obstructions[i].y, obstHeight };
            const float dist = GetDistanceBetweenTwoLineSegments(rayOrigin, rayDest, obstLineP1, obstLineP2);
            if (dist < obstructions[i].r1*2.5f)
            {
                const float x =  obstructions[i].x;
                const float y =  obstructions[i].y;
                if (obstructions[i].shape == Shape::SQUARE)
                {
                    const float r1 = obstructions[i].r1;
                    const float3 swt = { x - r1, y - r1, obstHeight };//-0.3f*80.f
                    const float3 set = { x + r1, y - r1, obstHeight };//-0.3f*80.f
                    const float3 nwt = { x - r1, y + r1, obstHeight };//-0.3f*80.f
                    const float3 net = { x + r1, y + r1, obstHeight };//-0.3f*80.f
                    const float3 swb = { x - (r1), y - (r1), -1.f };//-1.f*80.f
                    const float3 seb = { x + (r1), y - (r1), -1.f };//-1.f*80.f
                    const float3 nwb = { x - (r1), y + (r1), -1.f };//-1.f*80.f
                    const float3 neb = { x + (r1), y + (r1), -1.f };//-1.f*80.f

                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, nwt, swt, swb, nwb);
                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, swt, set, seb, swb);
                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, set, net, neb, seb);
                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, net, nwt, nwb, neb);
                }
                else if (obstructions[i].shape == Shape::CIRCLE)
                {
                    if (dist < obstructions[i].r1)
                    {
                        float3 v = CrossProduct(rayDir, obstLineP1 - obstLineP2);
                        Normalize(v);
                        intersect = float3{ x, y, obstHeight*0.5f }+dist*v;
                        hit = true;
                    }
                }
                else if (obstructions[i].shape == Shape::VERTICAL_LINE)
                {
                    const float r1 = LINE_OBST_WIDTH*0.501f;
                    const float r2 = obstructions[i].r1*2.f;
                    const float3 swt = { x - r1, y - r2, obstHeight };
                    const float3 set = { x + r1, y - r2, obstHeight };
                    const float3 nwt = { x - r1, y + r2, obstHeight };
                    const float3 net = { x + r1, y + r2, obstHeight };
                    const float3 swb = { x - (r1), y - (r2), 0.f };
                    const float3 seb = { x + (r1), y - (r2), 0.f };
                    const float3 nwb = { x - (r1), y + (r2), 0.f };
                    const float3 neb = { x + (r1), y + (r2), 0.f };

                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, nwt, swt, swb, nwb);
                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, swt, set, seb, swb);
                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, set, net, neb, seb);
                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, net, nwt, nwb, neb);
                }
                else if (obstructions[i].shape == Shape::HORIZONTAL_LINE)
                {
                    const float r1 = obstructions[i].r1*2.f;
                    const float r2 = LINE_OBST_WIDTH*0.501f;
                    const float3 swt = { x - r1, y - r2, obstHeight };
                    const float3 set = { x + r1, y - r2, obstHeight };
                    const float3 nwt = { x - r1, y + r2, obstHeight };
                    const float3 net = { x + r1, y + r2, obstHeight };
                    const float3 swb = { x - (r1), y - (r2), 0.f };
                    const float3 seb = { x + (r1), y - (r2), 0.f };
                    const float3 nwb = { x - (r1), y + (r2), 0.f };
                    const float3 neb = { x + (r1), y + (r2), 0.f };

                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, nwt, swt, swb, nwb);
                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, swt, set, seb, swb);
                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, set, net, neb, seb);
                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, net, nwt, nwb, neb);
                }
            }
        }
    }
    return hit;
}

__device__	float ScaledLength(const int p_l, const int p_maxDim)
{
    return (float)p_l / (p_maxDim - 1) * 2.f;
}

__device__	float ScaledCoord(const int p_x, const int p_maxDim)
{
    return (float)p_x / (p_maxDim - 1) * 2.f - 1.f;
}

__device__	int IntCoord(const float p_x, const int p_maxDim)
{
    return (p_x + 1.f)*0.5f*(p_maxDim - 1);
}

__device__	float2 ScaledCoords(int p_x, int p_y, const int p_maxDim)
{
    return make_float2(
        ScaledCoord(p_x, p_maxDim),
        ScaledCoord(p_y, p_maxDim));
}

__device__ float ObstructionPickingTol(const int p_xDimVisible)
{
    return 1.5f*2.f / p_xDimVisible;
}

// Initialize domain using constant velocity
__global__ void InitializeLBM(float4* vbo, float *f, int *Im, float uMax,
    Domain simDomain)
{
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int y = threadIdx.y + blockIdx.y*blockDim.y;

    LbmNode lbm;
    lbm.Initialize(f, 1.f, uMax, 0.f);
    lbm.WriteDistributions(f, x, y);
}

// main LBM function including streaming and colliding
__global__ void MarchLBM(float* fA, float* fB, const float omega, int *Im,
    ObstDefinition *obstructions, const float uMax, Domain simDomain)
{
    const int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int j = x + y*MAX_XDIM;
    const int im = Im[j];
    const int xDim = simDomain.GetXDim();
    const int yDim = simDomain.GetYDim();

    LbmNode lbm;
    lbm.SetXDim(xDim);
    lbm.SetYDim(yDim);
    lbm.ReadIncomingDistributions(fA, x, y);

    if (im == 1 || im == 10){//bounce-back condition
        lbm.BounceBackWall();
    }
//    else if (im == 20)
//    {
//        float rho, u, v;
//        rho = 1.0f;
//        u = obstructions[obstId].u;
//        v = obstructions[obstId].v;
//        lbm.MovingWall(rho, u, v);
//    }
    else
    {
        lbm.ApplyBCs(y, im, xDim, yDim, uMax);
        lbm.Collide(omega);
    }
    lbm.WriteDistributions(fB, x, y);
}

__global__ void UpdateSurfaceVbo(float4* vbo, float* fA, int *Im,
    const int contourVar, const float contMin, const float contMax,
    const int viewMode, const float uMax, Domain simDomain, const float waterDepth)
{
    const int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int j = x + y*MAX_XDIM;
    const int im = Im[j];

    const int xDim = simDomain.GetXDim();
    const int yDim = simDomain.GetYDim();
    LbmNode lbm;
    lbm.ReadDistributions(fA, x, y);
    const float rho = (im == 1) ? 1.0 : lbm.ComputeRho();
    const float u = lbm.ComputeU();
    const float v = lbm.ComputeV();

    //Prepare data for visualization

    //need to change x,y,z coordinates to NDC (-1 to 1)
    const int xDimVisible = simDomain.GetXDimVisible();
    const int yDimVisible = simDomain.GetYDimVisible();
    const float2 coords = ScaledCoords(x, y, xDimVisible);

    float zcoord = -1.f + waterDepth + 1.5f*(rho - 1.0f);
    float color;

    if (contourVar != ContourVariable::WATER_RENDERING)
    {
        //for color, need to convert 4 bytes (RGBA) to float
        float variableValue = 0.f;

        //change min/max contour values based on contour variable
        if (contourVar == ContourVariable::VEL_MAG)
        {
            variableValue = sqrt(u*u + v*v);
        }
        else if (contourVar == ContourVariable::VEL_U)
        {
            variableValue = u;
        }
        else if (contourVar == ContourVariable::VEL_V)
        {
            variableValue = v;
        }
        else if (contourVar == ContourVariable::PRESSURE)
        {
            variableValue = rho;
        }
        else if (contourVar == ContourVariable::STRAIN_RATE)
        {
            variableValue = lbm.ComputeStrainRateMagnitude();
        }

        ////Blue to white color scheme
        unsigned char R = dmin(255.f, dmax(255 * ((variableValue - contMin) /
            (contMax - contMin))));
        unsigned char G = dmin(255.f, dmax(255 * ((variableValue - contMin) /
            (contMax - contMin))));
        unsigned char B = 255;
        unsigned char A = 255;

        if (im == 1 || im == 20){
            R = 204; G = 204; B = 204;
        }

        unsigned char b[] = { R, G, B, A };
        std::memcpy(&color, &b, sizeof(color));
    }
    else
    {
        unsigned char b[] = { 0, 255, 0, 255 };
        std::memcpy(&color, &b, sizeof(color));
    }

    //vbo aray to be displayed
    vbo[j] = make_float4(coords.x, coords.y, zcoord, color);
}

__global__ void UpdateSurfaceNormals(float4* vbo, float4* p_normals, Domain simDomain)
{
    const int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int j = x + y*MAX_XDIM;
    const int xDimVisible = simDomain.GetXDimVisible();
    const int yDimVisible = simDomain.GetYDimVisible();

    float3 n = { 0, 0, 1 };
    float slope_x = 0.f;
    float slope_y = 0.f;
    const float cellSize = 2.f / xDimVisible;
    if (x == 0)
    {
        n.x = 0.f;
    }
    else if (y == 0)
    {
        n.y = 0.f;
    }
    else if (x >= xDimVisible - 1)
    {
        n.x = 0.f;
    }
    else if (y >= yDimVisible - 1)
    {
        n.y = 0.f;
    }
    else if (x > 0 && x < (xDimVisible - 1) && y > 0 && y < (yDimVisible - 1))
    {
        slope_x = (vbo[(x + 1) + y*MAX_XDIM].z - vbo[(x - 1) + y*MAX_XDIM].z) /
            (2.f*cellSize);
        slope_y = (vbo[(x)+(y + 1)*MAX_XDIM].z - vbo[(x)+(y - 1)*MAX_XDIM].z) /
            (2.f*cellSize);
        n.x = -slope_x*2.f*cellSize*2.f*cellSize;
        n.y = -slope_y*2.f*cellSize*2.f*cellSize;
        n.z = 2.f*cellSize*2.f*cellSize;
    }
    Normalize(n);
    p_normals[j] = make_float4(n.x, n.y, n.z, 0.f);
}

__global__ void PhongLighting(float4* vbo, float4* p_normals, ObstDefinition *obstructions, 
    float3 cameraPosition, Domain simDomain)
{
    const int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
    unsigned char color[4];
    std::memcpy(color, &(vbo[j].w), sizeof(color));
    unsigned char A = color[3];

    float3 n = make_float3(p_normals[j].x, p_normals[j].y, p_normals[j].z);

    const float3 elementPosition = {vbo[j].x,vbo[j].y,vbo[j].z };
    const float3 diffuseLightDirection1 = {0.577367, 0.577367, -0.577367 };
    const float3 diffuseLightDirection2 = { -0.577367, 0.577367, -0.577367 };
    //float3 cameraPosition = { -1.5, -1.5, 1.5};
    float3 eyeDirection = elementPosition - cameraPosition;
    const float3 diffuseLightColor1 = {0.5f, 0.5f, 0.5f};
    const float3 diffuseLightColor2 = {0.5f, 0.5f, 0.5f};
    const float3 specularLightColor1 = {0.5f, 0.5f, 0.5f};

    float cosTheta1 = -DotProduct(n,diffuseLightDirection1);
    cosTheta1 = cosTheta1 < 0 ? 0 : cosTheta1;
    float cosTheta2 = -DotProduct(n, diffuseLightDirection2);
    cosTheta2 = cosTheta2 < 0 ? 0 : cosTheta2;

    const float3 specularLightPosition1 = {-1.5f, -1.5f, 1.5f};
    const float3 specularLight1 = elementPosition - specularLightPosition1;
    float3 specularRefection1 = specularLight1 - 2.f*(DotProduct(specularLight1, n)*n);
    Normalize(specularRefection1);
    Normalize(eyeDirection);
    float cosAlpha = -DotProduct(eyeDirection, specularRefection1);
    cosAlpha = cosAlpha < 0 ? 0 : cosAlpha;
    cosAlpha = pow(cosAlpha, 5.f);

    const float lightAmbient = 0.3f;
    
    const float3 diffuse1  = 0.3f*cosTheta1*diffuseLightColor1;
    const float3 diffuse2  = 0.3f*cosTheta2*diffuseLightColor2;
    const float3 specular1 = cosAlpha*specularLightColor1;

    color[0] = color[0]*dmin(1.f,(diffuse1.x+diffuse2.x+specular1.x+lightAmbient));
    color[1] = color[1]*dmin(1.f,(diffuse1.y+diffuse2.y+specular1.y+lightAmbient));
    color[2] = color[2]*dmin(1.f,(diffuse1.z+diffuse2.z+specular1.z+lightAmbient));
    color[3] = A;

    std::memcpy(&(vbo[j].w), color, sizeof(color));
}

__global__ void InitializeMesh(float4* vbo, Domain simDomain)
{
    const int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
    const int xDimVisible = simDomain.GetXDimVisible();
    const int yDimVisible = simDomain.GetYDimVisible();
    const float zcoord = -1.f;
    const float2 coords = ScaledCoords(x, y, xDimVisible);
    unsigned char R(255), G(255), B(255), A(0);

    unsigned char b[] = { R, G, B, A };
    float color;
    std::memcpy(&color, &b, sizeof(color));
    vbo[j] = make_float4(coords.x, coords.y, zcoord, color);
}

__device__ float3 ReflectRay(float3 incidentLight, float3 n)
{
    return 2.f*DotProduct(incidentLight, -1.f*n)*n + incidentLight;
}

__device__ float3 RefractRay(float3 incidentLight, float3 n)
{
    const float r = 1.0 / WATER_REFRACTIVE_INDEX;
    const float c = -(DotProduct(n, incidentLight));
    return r*incidentLight + (r*c - sqrt(1.f - r*r*(1.f - c*c)))*n;
}

//TODO: Make this in model space coords
__device__ float2 ComputePositionOfLightOnFloor(float4* vbo, float3 incidentLight, 
    const int x, const int y, Domain simDomain, const float waterDepth)
{
    const int xDimVisible = simDomain.GetXDimVisible();
    const int yDimVisible = simDomain.GetYDimVisible();
    float3 n = { 0, 0, 1 };
    float slope_x = 0.f;
    float slope_y = 0.f;
    const float cellSize = 2.f / xDimVisible;
    if (x > 0 && x < (xDimVisible - 1) && y > 0 && y < (yDimVisible - 1))
    {
        slope_x = (vbo[(x + 1) + y*MAX_XDIM].z - vbo[(x - 1) + y*MAX_XDIM].z) /
            (2.f*cellSize);
        slope_y = (vbo[(x)+(y + 1)*MAX_XDIM].z - vbo[(x)+(y - 1)*MAX_XDIM].z) /
            (2.f*cellSize);
        n.x = -slope_x*2.f*cellSize*2.f*cellSize;
        n.y = -slope_y*2.f*cellSize*2.f*cellSize;
        n.z = 2.f*cellSize*2.f*cellSize;
    }
    Normalize(n);

    Normalize(incidentLight);

    const float3 refractedLight = RefractRay(incidentLight, n);

    const float2 coords = ScaledCoords(x, y, xDimVisible);
    //const float xf = (float)x / xDimVisible*2.f - 1.f;
    //const float yf = (float)y / xDimVisible*2.f - 1.f;

    const float2 delta = make_float2(
        -refractedLight.x*waterDepth / refractedLight.z,
        -refractedLight.y*waterDepth / refractedLight.z);

    //const float dx = -refractedLight.x*waterDepth / refractedLight.z;
    //const float dy = -refractedLight.y*waterDepth/refractedLight.z;

    //return float2{ (xf + dx), (yf + dy)};

    return coords + delta;
}

__device__ float ComputeAreaFrom4Points(const float2 &nw, const float2 &ne,
    const float2 &sw, const float2 &se)
{
    const float2 vecN = ne - nw;
    const float2 vecS = se - sw;
    const float2 vecE = ne - se;
    const float2 vecW = nw - sw;
    return CrossProductArea(vecN, vecW) + CrossProductArea(vecE, vecS);
}

__global__ void DeformFloorMeshUsingCausticRay(float4* vbo, float3 incidentLight, 
    ObstDefinition* obstructions, Domain simDomain, const float waterDepth)
{
    const int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
    const int xDimVisible = simDomain.GetXDimVisible();
    const int yDimVisible = simDomain.GetYDimVisible();
    
    if (x < xDimVisible && y < yDimVisible)
    {
        const float2 lightPositionOnFloor = ComputePositionOfLightOnFloor(vbo, incidentLight,
            x, y, simDomain, waterDepth);

        vbo[j + MAX_XDIM*MAX_YDIM].x = lightPositionOnFloor.x;
        vbo[j + MAX_XDIM*MAX_YDIM].y = lightPositionOnFloor.y;
    }
}

__global__ void ComputeFloorLightIntensitiesFromMeshDeformation(float4* vbo, float* floor_d, 
    ObstDefinition* obstructions, Domain simDomain, int* p_image)
{
    const int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int xDimVisible = simDomain.GetXDimVisible();
    const int yDimVisible = simDomain.GetYDimVisible();

    const int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
    const int im = p_image[j];

    if (x < xDimVisible-2 && y < yDimVisible-2)
    {
		if (im != 1)
        {
            const int offset = MAX_XDIM*MAX_YDIM;
            const float2 nw = make_float2(vbo[(x)+(y + 1)*MAX_XDIM + offset].x, vbo[(x)+(y + 1)*MAX_XDIM + offset].y);
            const float2 ne = make_float2(vbo[(x + 1) + (y + 1)*MAX_XDIM + offset].x, vbo[(x + 1) + (y + 1)*MAX_XDIM + offset].y);
            const float2 sw = make_float2(vbo[(x)+(y)*MAX_XDIM + offset].x, vbo[(x)+(y)*MAX_XDIM + offset].y);
            const float2 se = make_float2(vbo[(x + 1) + (y)*MAX_XDIM + offset].x, vbo[(x + 1) + (y)*MAX_XDIM + offset].y);

            const float areaOfLightMeshOnFloor = ComputeAreaFrom4Points(nw, ne, sw, se);
            const float cellSize = ScaledLength(1, xDimVisible);
            const float incidentLightIntensity = 0.4f;
            const float lightIntensity = incidentLightIntensity*(cellSize*cellSize) / areaOfLightMeshOnFloor;
            atomicAdd(&floor_d[x + (y)*MAX_XDIM], lightIntensity*0.25f);
            atomicAdd(&floor_d[x + 1 + (y)*MAX_XDIM], lightIntensity*0.25f);
            atomicAdd(&floor_d[x + 1 + (y + 1)*MAX_XDIM], lightIntensity*0.25f);
            atomicAdd(&floor_d[x + (y + 1)*MAX_XDIM], lightIntensity*0.25f);
        }
    }
}


__global__ void ApplyCausticLightingToFloor(float4* vbo, float* floor_d, 
    ObstDefinition* obstructions, Domain simDomain, const float obstHeight)
{
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int j = MAX_XDIM*MAX_YDIM + x + y*MAX_XDIM;

    float lightFactor = dmin(1.f,floor_d[x + y*MAX_XDIM]);
    floor_d[x + y*MAX_XDIM] = 0.f;

    unsigned char R = 255;
    unsigned char G = 255;
    unsigned char B = 255;
    unsigned char A = 255;

    const int xDimVisible = simDomain.GetXDimVisible();
    const int yDimVisible = simDomain.GetYDimVisible();

    //! NOTE: Mesh deformation is disabled for rendering to avoid mesh folding
    float2 coords = ScaledCoords(x, y, xDimVisible);
    float zcoord = vbo[j].z;

    R *= lightFactor;
    G *= lightFactor;
    B *= lightFactor;

    R = dmin(255, R);
    G = dmin(255, G);
    B = dmin(255, B);

    unsigned char b[] = { R, G, B, A };
    float color;
    std::memcpy(&color, &b, sizeof(color));

    vbo[j].z = zcoord;
    vbo[j].w = color;
}

__device__ int GetIntersectWithCubeMap(float3 &intersect, const float3 &rayOrigin, const float3 &rayDir)
{
    float distance = 99999999;
    float temp;
    int side = -1;
    const float maxDim = 1.f;// dmax(MAX_XDIM, MAX_YDIM);
    const float minDim = -1.f;
    if (rayDir.x > 0)
    {
        temp = (maxDim - rayOrigin.x) / rayDir.x;
        if (temp < distance)
        {
            distance = temp;
            side = 3;
        }
    }
    else if (rayDir.x < 0)
    {
        temp = (minDim -rayOrigin.x) / rayDir.x;
        if (temp < distance)
        {
            distance = temp;
            side = 1;
        }
    }
    if (rayDir.y > 0)
    {
        temp = (maxDim - rayOrigin.y) / rayDir.y;
        if (temp < distance)
        {
            distance = temp;
            side = 2;
        }
    }
    else if (rayDir.y < 0)
    {
        temp = (minDim -rayOrigin.y) / rayDir.y;
        if (temp < distance)
        {
            distance = temp;
            side = 4;
        }
    }
    if (rayDir.z > 0)
    {
        temp = (maxDim - rayOrigin.z) / rayDir.z;
        if (temp < distance)
        {
            distance = temp;
            side = 0;
        }
    }
    else if (rayDir.z < 0)
    {
        temp = (minDim -rayOrigin.z) / rayDir.z;
        if (temp < distance)
        {
            distance = temp;
            side = 5;
        }
    }
    intersect = (rayOrigin + distance*rayDir) / maxDim;
    return side;

}

__device__ int GetCubeMapFace(const float3 &rayDir)
{
    const float3 absDir = { abs(rayDir.x), abs(rayDir.y), abs(rayDir.z) };
    //if (absDir.z > absDir.x && absDir.z > absDir.y)
    if (absDir.z*absDir.z > absDir.x*absDir.x + absDir.y*absDir.y)
    {
        if (rayDir.z > 0)
            return 0;
        return 5;
    }
    //if (absDir.y > absDir.x && absDir.y > absDir.z)
    if (absDir.y*absDir.y > absDir.x*absDir.x + absDir.z*absDir.z)
    {
        if (rayDir.y > 0)
            return 2;
        return 4;
    }
    //if (absDir.x > absDir.y && absDir.x > absDir.z)
    if (absDir.x*absDir.x > absDir.y*absDir.y + absDir.z*absDir.z)
    {
        if (rayDir.x > 0)
            return 3;
        return 1;
    }
    return -1;
}


__device__ float2 GetUVCoordsForSkyMap(const float3 &rayOrigin, const float3 &rayDir)
{
    float2 uv;
    float3 intersect;
    if (GetIntersectWithCubeMap(intersect, rayOrigin, rayDir) == 0) //posz
    {
        uv.x = CAUSTICS_TEX_SIZE+IntCoord(intersect.x, CAUSTICS_TEX_SIZE)+0.5f;
        uv.y = 2 * CAUSTICS_TEX_SIZE + (CAUSTICS_TEX_SIZE - 1) - IntCoord(intersect.y, CAUSTICS_TEX_SIZE) + 0.5f;
    }
    else if (GetIntersectWithCubeMap(intersect, rayOrigin, rayDir) == 1) //negx
    {
        uv.x = IntCoord(intersect.y, CAUSTICS_TEX_SIZE)+0.5f;
        uv.y = CAUSTICS_TEX_SIZE+IntCoord(intersect.z, CAUSTICS_TEX_SIZE)+0.5f;
    }
    else if (GetIntersectWithCubeMap(intersect, rayOrigin, rayDir) == 3) //posx
    {
        uv.x = 2*CAUSTICS_TEX_SIZE+(CAUSTICS_TEX_SIZE - 1) - IntCoord(intersect.y, CAUSTICS_TEX_SIZE)+0.5f;
        uv.y = CAUSTICS_TEX_SIZE+IntCoord(intersect.z, CAUSTICS_TEX_SIZE)+0.5f;
    }
    else if (GetIntersectWithCubeMap(intersect, rayOrigin, rayDir) == 2) //posy
    {
        uv.x = CAUSTICS_TEX_SIZE+IntCoord(intersect.x, CAUSTICS_TEX_SIZE)+0.5f;
        uv.y = CAUSTICS_TEX_SIZE+IntCoord(intersect.z, CAUSTICS_TEX_SIZE)+0.5f;
    }
    else if (GetIntersectWithCubeMap(intersect, rayOrigin, rayDir) == 4) //negy
    {
        uv.x = 3*CAUSTICS_TEX_SIZE+(CAUSTICS_TEX_SIZE - 1) - IntCoord(intersect.x, CAUSTICS_TEX_SIZE)+0.5f;
        uv.y = CAUSTICS_TEX_SIZE+IntCoord(intersect.z, CAUSTICS_TEX_SIZE)+0.5f;
    }
    else //negz - this would be the floor
    {
        uv.x = CAUSTICS_TEX_SIZE+IntCoord(intersect.x, CAUSTICS_TEX_SIZE)+0.5f;
        uv.y = (CAUSTICS_TEX_SIZE - 1) - IntCoord(intersect.y, CAUSTICS_TEX_SIZE)+0.5f;
    }
    
    return uv;
}


texture<float4, 2, cudaReadModeElementType> floorTex;
texture<float4, 2, cudaReadModeElementType> envTex;

__global__ void SurfaceRefraction(float4* vbo, float4* p_normals, ObstDefinition *obstructions,
    float3 cameraPosition, Domain simDomain, const bool simplified, const float waterDepth, const float obstHeight)
{
    const int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)

    const int xDimVisible = simDomain.GetXDimVisible();

    const float3 n = make_float3(p_normals[j].x, p_normals[j].y, p_normals[j].z);

    const float waterDepthNormalized = (vbo[j].z + 1.f);
    const float xcoord = vbo[j].x;
    const float ycoord = vbo[j].y;
    const float3 elemPosNormalized = { xcoord, ycoord, vbo[j].z };
    const float3 viewingRay = elemPosNormalized - cameraPosition;  //normalized

    const float3 refractedRay = RefractRay(viewingRay, n);
    const float3 reflectedRay = ReflectRay(viewingRay, n);
    const float cosTheta = dmax(0.f,dmin(1.f,DotProduct(viewingRay, -1.f*n)));
    const float nu = 1.f / WATER_REFRACTIVE_INDEX;
    const float r0 = (nu - 1.f)*(nu - 1.f) / ((nu + 1.f)*(nu + 1.f));
    const float reflectedRayIntensity = r0 + (1.f - cosTheta)*(1.f - cosTheta)*(1.f - cosTheta)*(1.f - cosTheta)*(1.f - cosTheta)*(1.f - r0);
    
    const float dx = -refractedRay.x*waterDepthNormalized / refractedRay.z;
    const float dy = -refractedRay.y*waterDepthNormalized / refractedRay.z;

    const float xf = xcoord + dx;
    const float yf = ycoord + dy;
    const float xTex = IntCoord(xf, CAUSTICS_TEX_SIZE) + 0.5f;
    const float yTex = IntCoord(yf, CAUSTICS_TEX_SIZE) + 0.5f;

    const float2 uvSkyTex = GetUVCoordsForSkyMap(elemPosNormalized, reflectedRay);
    const float4 skyColor = tex2D(envTex, uvSkyTex.x, uvSkyTex.y);
    const float4 textureColor = tex2D(floorTex, xTex, yTex);

    const float3 refractedRayDest = { xf, yf, -1.f };

    unsigned char color[4];
    float3 refractionIntersect = { 99999, 99999, 99999 };
    float3 reflectionIntersect = { 99999, 99999, 99999 };
    if (xf > 1 || xf < -1 || yf > 1 || yf < -1)
    {
        color[0] = 0;
        color[1] = 0; 
        color[2] = 0; 
        color[3] = 0; 
    }
    else
    {
        if (simplified)
        {
            unsigned char refractedColor[4];
            refractedColor[0] = dmin((int)(textureColor.x*255.f), 255);
            refractedColor[1] = dmin((int)(textureColor.y*255.f), 255);
            refractedColor[2] = dmin((int)(textureColor.z*255.f), 255);
            refractedColor[3] = 255;

            unsigned char reflectedColor[4];
            reflectedColor[0] = skyColor.x;
            reflectedColor[1] = skyColor.y;
            reflectedColor[2] = skyColor.z;
            reflectedColor[3] = 255;

            if (textureColor.x > 1.f || textureColor.y > 1.f || textureColor.z > 1.f)
                printf("%f, %f, %f \n", textureColor.x, textureColor.y, textureColor.z);

            color[0] = (1.f - reflectedRayIntensity)*(float)refractedColor[0] + reflectedRayIntensity*(float)reflectedColor[0];
            color[1] = (1.f - reflectedRayIntensity)*(float)refractedColor[1] + reflectedRayIntensity*(float)reflectedColor[1];
            color[2] = (1.f - reflectedRayIntensity)*(float)refractedColor[2] + reflectedRayIntensity*(float)reflectedColor[2];
            color[3] = 255;
        }
        else
        {
            unsigned char refractedColor[4];
            if (GetCoordFromRayHitOnObst(refractionIntersect, elemPosNormalized, refractedRayDest, obstructions, obstHeight-1.f))
            {
                std::memcpy(refractedColor,
                    &(vbo[(int)(IntCoord(refractionIntersect.x, xDimVisible)+0.5f) +
                        (int)(IntCoord(refractionIntersect.y, xDimVisible)+0.5f)*MAX_XDIM + MAX_XDIM * MAX_YDIM].w),
                    sizeof(refractedColor));
            }
            else
            {
                refractedColor[0] = dmin((int)(textureColor.x*255.f), 255);
                refractedColor[1] = dmin((int)(textureColor.y*255.f), 255);
                refractedColor[2] = dmin((int)(textureColor.z*255.f), 255);
                refractedColor[3] = 255;
            }

            unsigned char reflectedColor[4];
            const float3 reflectedRayDest = elemPosNormalized + xDimVisible*reflectedRay;
            if (GetCoordFromRayHitOnObst(reflectionIntersect, elemPosNormalized, reflectedRayDest, obstructions, obstHeight-1.f))
            {
                std::memcpy(
                    reflectedColor,
                    &(vbo[(int)(IntCoord(reflectionIntersect.x, xDimVisible)+0.5f) +
                        (int)(IntCoord(reflectionIntersect.y, xDimVisible)+0.5f)*MAX_XDIM + MAX_XDIM * MAX_YDIM].w),
                    sizeof(reflectedColor));
            }
            else
            {
                reflectedColor[0] = skyColor.x;
                reflectedColor[1] = skyColor.y;
                reflectedColor[2] = skyColor.z;
                reflectedColor[3] = 255;
            }

            color[0] = (1.f-reflectedRayIntensity)*(float)refractedColor[0]+reflectedRayIntensity*(float)reflectedColor[0];
            color[1] = (1.f-reflectedRayIntensity)*(float)refractedColor[1]+reflectedRayIntensity*(float)reflectedColor[1];
            color[2] = (1.f-reflectedRayIntensity)*(float)refractedColor[2]+reflectedRayIntensity*(float)reflectedColor[2];
            color[3] = 255;
        }
    }
    std::memcpy(&(vbo[j].w), color, sizeof(color));
}


/*----------------------------------------------------------------------------------------
 * End of device functions
 */

void InitializeDomain(float4* vis, float* f_d, int* im_d, const float uMax,
    Domain &simDomain)
{
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(MAX_XDIM) / BLOCKSIZEX), MAX_YDIM / BLOCKSIZEY);
    InitializeLBM << <grid, threads >> >(vis, f_d, im_d, uMax, simDomain);
}

void SetObstructionVelocitiesToZero(ObstDefinition* obst_h, ObstDefinition* obst_d, Domain& simDomain)
{
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if ((abs(obst_h[i].u) > 0.f || abs(obst_h[i].v) > 0.f) &&
            obst_h[i].state != State::SELECTED)
        {
            UpdateObstructions << <1, 1 >> >(obst_d,i,obst_h[i]);
        }
    }
}

void MarchSolution(CudaLbm* cudaLbm)
{
    Domain* simDomain = cudaLbm->GetDomain();
    const int xDim = simDomain->GetXDim();
    const int yDim = simDomain->GetYDim();
    const int tStep = cudaLbm->GetTimeStepsPerFrame();
    float* fA_d = cudaLbm->GetFA();
    float* fB_d = cudaLbm->GetFB();
    int* im_d = cudaLbm->GetImage();
    ObstDefinition* obst_d = cudaLbm->GetDeviceObst();
    const float u = cudaLbm->GetInletVelocity();
    const float omega = cudaLbm->GetOmega();

    const dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    const dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    for (int i = 0; i < tStep; i+=2)
    {
        MarchLBM << <grid, threads >> >(fA_d, fB_d, omega, im_d, obst_d, u, *simDomain);
        MarchLBM << <grid, threads >> >(fB_d, fA_d, omega, im_d, obst_d, u, *simDomain);
    }
}

void UpdateSolutionVbo(float4* vis, float4* p_normals, CudaLbm* cudaLbm, const ContourVariable contVar,
    const float contMin, const float contMax, const ViewMode viewMode, const float waterDepth)
{
    Domain* simDomain = cudaLbm->GetDomain();
    const int xDim = simDomain->GetXDim();
    const int yDim = simDomain->GetYDim();
    float* f_d = cudaLbm->GetFA();
    int* im_d = cudaLbm->GetImage();
    const float u = cudaLbm->GetInletVelocity();

    const dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    const dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    UpdateSurfaceVbo << <grid, threads >> > (vis, f_d, im_d, contVar, contMin, contMax,
        viewMode, u, *simDomain, waterDepth);
    UpdateSurfaceNormals << <grid, threads >> > (vis, p_normals, *simDomain);
}

void UpdateDeviceObstructions(ObstDefinition* obst_d, const int targetObstID,
    const ObstDefinition &newObst, Domain& simDomain)
{
    UpdateObstructions << <1, 1 >> >(obst_d, targetObstID, newObst);
}

void SurfacePhongLighting(float4* vis, float4* p_normals, ObstDefinition* obst_d, const float3 cameraPosition,
    Domain &simDomain)
{
    const int xDim = simDomain.GetXDim();
    const int yDim = simDomain.GetYDim();
    const dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    const dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    PhongLighting << <grid, threads>> >(vis, p_normals, obst_d, cameraPosition, simDomain);
}

void InitializeSurface(float4* vis, Domain &simDomain)
{
    const dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    const dim3 grid(ceil(static_cast<float>(MAX_XDIM) / BLOCKSIZEX), MAX_YDIM / BLOCKSIZEY);
    InitializeMesh << <grid, threads >> >(vis, simDomain);
}

void InitializeFloor(float4* vis, Domain &simDomain)
{
    const dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    const dim3 grid(ceil(static_cast<float>(MAX_XDIM) / BLOCKSIZEX), MAX_YDIM / BLOCKSIZEY);
    InitializeMesh << <grid, threads >> >(&vis[MAX_XDIM*MAX_YDIM], simDomain);
}

void LightFloor(float4* vis, float4* p_normals, float* floor_d, ObstDefinition* obst_d,
    const float3 cameraPosition, Domain &simDomain, CudaLbm& p_lbm, const float waterDepth, const float obstHeight)
{
    const int xDim = simDomain.GetXDim();
    const int yDim = simDomain.GetYDim();
    const dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    const dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    const float3 incidentLight1 = { 0.f, 0.f, -1.f };
    DeformFloorMeshUsingCausticRay << <grid, threads >> >
        (vis, incidentLight1, obst_d, simDomain, waterDepth);
    ComputeFloorLightIntensitiesFromMeshDeformation << <grid, threads >> >
        (vis, floor_d, obst_d, simDomain, p_lbm.GetImage());

    ApplyCausticLightingToFloor << <grid, threads >> >(vis, floor_d, obst_d, simDomain, obstHeight);

    //phong lighting on floor mesh to shade obstructions
    PhongLighting << <grid, threads>> >(&vis[MAX_XDIM*MAX_YDIM], p_normals, obst_d, cameraPosition,
        simDomain);
}

void RefractSurface(float4* vis, float4* p_normals, cudaArray* floorLightTexture, cudaArray* envTexture, ObstDefinition* obst_d, const glm::vec4 cameraPos,
    Domain &simDomain, const float waterDepth, const float obstHeight, const bool simplified)
{
    const int xDim = simDomain.GetXDim();
    const int yDim = simDomain.GetYDim();
    const dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    const dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    gpuErrchk(cudaBindTextureToArray(floorTex, floorLightTexture));
    gpuErrchk(cudaBindTextureToArray(envTex, envTexture));
    const float3 f3CameraPos = make_float3(cameraPos.x, cameraPos.y, cameraPos.z);
    SurfaceRefraction << <grid, threads>> >(vis, p_normals, obst_d, f3CameraPos, simDomain, simplified, waterDepth, obstHeight);
}

