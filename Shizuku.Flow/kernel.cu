#define WATER_DEPTH_NORMALIZED 0.5f
#define OBST_HEIGHT 0.8f
#define WATER_REFRACTIVE_INDEX 1.33f

#include "kernel.h"
#include "LbmNode.h"
#include "CudaCheck.h"
#include "VectorUtils.h"
#include "Graphics/CudaLbm.h"

/*----------------------------------------------------------------------------------------
 *	Device functions
 */

__global__ void UpdateObstructions(Obstruction* obstructions, const int obstNumber,
    const Obstruction newObst)
{
    obstructions[obstNumber].shape = newObst.shape;
    obstructions[obstNumber].r1 = newObst.r1;
    obstructions[obstNumber].x = newObst.x;
    obstructions[obstNumber].y = newObst.y;
    obstructions[obstNumber].u = newObst.u;
    obstructions[obstNumber].v = newObst.v;
    obstructions[obstNumber].state = newObst.state;
}

inline __device__ bool IsInsideObstruction(const float x, const float y,
    Obstruction* obstructions, const float tolerance = 0.f)
{
    for (int i = 0; i < MAXOBSTS; i++){
        if (obstructions[i].state != State::INACTIVE)
        {
            float r1 = obstructions[i].r1;
            if (obstructions[i].shape == Shape::SQUARE){
                if (abs(x - obstructions[i].x)<r1 + tolerance &&
                    abs(y - obstructions[i].y)<r1 + tolerance)
                    return true;
            }
            else if (obstructions[i].shape == Shape::CIRCLE){//shift by 0.5 cells for better looks
                float distFromCenter = (x + 0.5f - obstructions[i].x)*(x + 0.5f - obstructions[i].x)
                    + (y + 0.5f - obstructions[i].y)*(y + 0.5f - obstructions[i].y);
                if (distFromCenter<(r1+tolerance)*(r1+tolerance)+0.1f)
                    return true;
            }
            else if (obstructions[i].shape == Shape::HORIZONTAL_LINE){
                if (abs(x - obstructions[i].x)<r1*2+tolerance &&
                    abs(y - obstructions[i].y)<LINE_OBST_WIDTH*0.501f+tolerance)
                    return true;
            }
            else if (obstructions[i].shape == Shape::VERTICAL_LINE){
                if (abs(y - obstructions[i].y)<r1*2+tolerance &&
                    abs(x - obstructions[i].x)<LINE_OBST_WIDTH*0.501f+tolerance)
                    return true;
            }
        }
    }
    return false;
}

inline __device__ int FindOverlappingObstruction(const float x, const float y,
    Obstruction* obstructions, const float tolerance = 0.f)
{
    for (int i = 0; i < MAXOBSTS; i++){
        if (obstructions[i].state != State::INACTIVE)
        {
            float r1 = obstructions[i].r1 + tolerance;
            if (obstructions[i].shape == Shape::SQUARE){
                if (abs(x - obstructions[i].x)<r1 && abs(y - obstructions[i].y)<r1)
                    return i;//10;
            }
            else if (obstructions[i].shape == Shape::CIRCLE){//shift by 0.5 cells for better looks
                float distFromCenter = (x + 0.5f - obstructions[i].x)*(x + 0.5f - obstructions[i].x)
                    + (y + 0.5f - obstructions[i].y)*(y + 0.5f - obstructions[i].y);
                if (distFromCenter<r1*r1+0.1f)
                    return i;//10;
            }
            else if (obstructions[i].shape == Shape::HORIZONTAL_LINE){
                if (abs(x - obstructions[i].x)<r1*2 &&
                    abs(y - obstructions[i].y)<LINE_OBST_WIDTH*0.501f+tolerance)
                    return i;//10;
            }
            else if (obstructions[i].shape == Shape::VERTICAL_LINE){
                if (abs(y - obstructions[i].y)<r1*2 &&
                    abs(x - obstructions[i].x)<LINE_OBST_WIDTH*0.501f+tolerance)
                    return i;//10;
            }
        }
    }
    return -1;
}

__device__ bool GetCoordFromRayHitOnObst(float3 &intersect, const float3 rayOrigin, const float3 rayDest,
    Obstruction* obstructions, float obstHeight, const float tolerance = 0.f)
{
    float3 rayDir = rayDest - rayOrigin;
    bool hit = false;
    for (int i = 0; i < MAXOBSTS; i++){
        if (obstructions[i].state == State::ACTIVE)
        {
            float3 obstLineP1 = { obstructions[i].x, obstructions[i].y, 0.f };
            float3 obstLineP2 = { obstructions[i].x, obstructions[i].y, obstHeight };
            float dist = GetDistanceBetweenTwoLineSegments(rayOrigin, rayDest, obstLineP1, obstLineP2);
            if (dist < obstructions[i].r1*2.5f)
            {
                float x =  obstructions[i].x;
                float y =  obstructions[i].y;
                if (obstructions[i].shape == Shape::SQUARE)
                {
                    float r1 = obstructions[i].r1;
                    float3 swt = { x - r1, y - r1, obstHeight };//-0.3f*80.f
                    float3 set = { x + r1, y - r1, obstHeight };//-0.3f*80.f
                    float3 nwt = { x - r1, y + r1, obstHeight };//-0.3f*80.f
                    float3 net = { x + r1, y + r1, obstHeight };//-0.3f*80.f
                    float3 swb = { x - (r1+0.5f), y - (r1+0.5f), 0.f };//-1.f*80.f
                    float3 seb = { x + (r1+0.5f), y - (r1+0.5f), 0.f };//-1.f*80.f
                    float3 nwb = { x - (r1+0.5f), y + (r1+0.5f), 0.f };//-1.f*80.f
                    float3 neb = { x + (r1+0.5f), y + (r1+0.5f), 0.f };//-1.f*80.f

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
                    float r1 = LINE_OBST_WIDTH*0.501f;
                    float r2 = obstructions[i].r1*2.f;
                    float3 swt = { x - r1, y - r2, obstHeight };
                    float3 set = { x + r1, y - r2, obstHeight };
                    float3 nwt = { x - r1, y + r2, obstHeight };
                    float3 net = { x + r1, y + r2, obstHeight };
                    float3 swb = { x - (r1), y - (r2), 0.f };
                    float3 seb = { x + (r1), y - (r2), 0.f };
                    float3 nwb = { x - (r1), y + (r2), 0.f };
                    float3 neb = { x + (r1), y + (r2), 0.f };

                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, nwt, swt, swb, nwb);
                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, swt, set, seb, swb);
                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, set, net, neb, seb);
                    hit = hit | IntersectLineSegmentWithRect(intersect, rayOrigin, rayDest, net, nwt, nwb, neb);
                }
                else if (obstructions[i].shape == Shape::HORIZONTAL_LINE)
                {
                    float r1 = obstructions[i].r1*2.f;
                    float r2 = LINE_OBST_WIDTH*0.501f;
                    float3 swt = { x - r1, y - r2, obstHeight };
                    float3 set = { x + r1, y - r2, obstHeight };
                    float3 nwt = { x - r1, y + r2, obstHeight };
                    float3 net = { x + r1, y + r2, obstHeight };
                    float3 swb = { x - (r1), y - (r2), 0.f };
                    float3 seb = { x + (r1), y - (r2), 0.f };
                    float3 nwb = { x - (r1), y + (r2), 0.f };
                    float3 neb = { x + (r1), y + (r2), 0.f };

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


__device__	void ChangeCoordinatesToNDC(float &xcoord,float &ycoord,
    const int xDimVisible, const int yDimVisible)
{
    xcoord = threadIdx.x + blockDim.x*blockIdx.x;
    ycoord = threadIdx.y + blockDim.y*blockIdx.y;
    xcoord /= xDimVisible *0.5f;
    ycoord /= yDimVisible *0.5f;//(float)(blockDim.y*gridDim.y);
    xcoord -= 1.0;// xdim / maxDim;
    ycoord -= 1.0;// ydim / maxDim;
}

__device__	void ChangeCoordinatesToScaledFloat(float &xcoord,float &ycoord,
    const int xDimVisible, const int yDimVisible)
{
    xcoord = threadIdx.x + blockDim.x*blockIdx.x;
    ycoord = threadIdx.y + blockDim.y*blockIdx.y;
    xcoord /= xDimVisible *0.5f;
    ycoord /= xDimVisible *0.5f;//(float)(blockDim.y*gridDim.y);
    xcoord -= 1.0;// xdim / maxDim;
    ycoord -= 1.0;// ydim / maxDim;
}

// Initialize domain using constant velocity
__global__ void InitializeLBM(float4* vbo, float *f, int *Im, float uMax,
    Domain simDomain)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    LbmNode lbm;
    lbm.Initialize(f, 1.f, uMax, 0.f);
    lbm.WriteDistributions(f, x, y);

    float xcoord, ycoord, zcoord;
    int xDimVisible = simDomain.GetXDimVisible();
    int yDimVisible = simDomain.GetYDimVisible();
    ChangeCoordinatesToScaledFloat(xcoord, ycoord, xDimVisible, yDimVisible);
    zcoord = 0.f;
    unsigned char R(255), G(255), B(255), A(255);
    unsigned char b[] = { R, G, B, A };
    float color;
    std::memcpy(&color, &b, sizeof(color));
    int j = x + y*MAX_XDIM;
    vbo[j] = make_float4(xcoord, ycoord, zcoord, color);
}

// main LBM function including streaming and colliding
__global__ void MarchLBM(float* fA, float* fB, const float omega, int *Im,
    Obstruction *obstructions, const float uMax, Domain simDomain)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = x + y*MAX_XDIM;
    int im = Im[j];
    int obstId = FindOverlappingObstruction(x, y, obstructions);
    if (obstId >= 0)
    {
        if (obstructions[obstId].u < 1e-2f && obstructions[obstId].v < 1e-2f)
        {
            im = 1; //bounce back
            Im[j] = im;
        }
        else
        {
            im = 20; //moving wall
            Im[j] = im;
        }
    }
    int xDim = simDomain.GetXDim();
    int yDim = simDomain.GetYDim();

    LbmNode lbm;
    lbm.SetXDim(xDim);
    lbm.SetYDim(yDim);
    lbm.ReadIncomingDistributions(fA, x, y);

    if (im == 1 || im == 10){//bounce-back condition
        lbm.BounceBackWall();
    }
    else if (im == 20)
    {
        float rho, u, v;
        rho = 1.0f;
        u = obstructions[FindOverlappingObstruction(x, y, obstructions)].u;
        v = obstructions[FindOverlappingObstruction(x, y, obstructions)].v;
        lbm.MovingWall(rho, u, v);
    }
    else{
        lbm.ApplyBCs(y, im, xDim, yDim, uMax);
        lbm.Collide(omega);
    }
    lbm.WriteDistributions(fB, x, y);
}

// main LBM function including streaming and colliding
__global__ void UpdateSurfaceVbo(float4* vbo, float* fA, int *Im,
    const int contourVar, const float contMin, const float contMax,
    const int viewMode, const float uMax, Domain simDomain, const float waterDepth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = x + y*MAX_XDIM;
    int im = Im[j];
    float u, v, rho;

    int xDim = simDomain.GetXDim();
    int yDim = simDomain.GetYDim();
    LbmNode lbm;
    lbm.ReadDistributions(fA, x, y);
    rho = lbm.ComputeRho();
    u = lbm.ComputeU();
    v = lbm.ComputeV();

    //Prepare data for visualization

    //need to change x,y,z coordinates to NDC (-1 to 1)
    float xcoord, ycoord, zcoord;
    int xDimVisible = simDomain.GetXDimVisible();
    int yDimVisible = simDomain.GetYDimVisible();
    ChangeCoordinatesToScaledFloat(xcoord, ycoord, xDimVisible, yDimVisible);

    if (im == 1) rho = 1.0;
    zcoord = -1.f + waterDepth + 1.5f*(rho - 1.0f);
    //zcoord =  (-1.f+waterDepth) + 1.5f*(rho - 1.0f);

    //for color, need to convert 4 bytes (RGBA) to float
    float variableValue = 0.f;

    float strainRate;

    //change min/max contour values based on contour variable
    if (contourVar == ContourVariable::VEL_MAG)
    {
        variableValue = sqrt(u*u+v*v);
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
        strainRate = lbm.ComputeStrainRateMagnitude();
        variableValue = strainRate;
    }

    ////Blue to white color scheme
    unsigned char R = dmin(255.f,dmax(255 * ((variableValue - contMin) /
        (contMax - contMin))));
    unsigned char G = dmin(255.f,dmax(255 * ((variableValue - contMin) /
        (contMax - contMin))));
    unsigned char B = 255;
    unsigned char A = 255;

    if (contourVar == ContourVariable::WATER_RENDERING)
    {
        R = 100; G = 150; B = 255;
        A = 100;
    }
    if (im == 1 || im == 20){
        R = 204; G = 204; B = 204;
    }
//    else if (im != 0 || x == xDimVisible-1)
//    {
//        zcoord = -1.f;
//    }

    float color;
    unsigned char b[] = { R, G, B, A };
    std::memcpy(&color, &b, sizeof(color));

    //vbo aray to be displayed
    vbo[j] = make_float4(xcoord, ycoord, zcoord, color);
}

__global__ void CleanUpVBO(float4* vbo, Domain simDomain)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
    float xcoord, ycoord;
    int xDimVisible = simDomain.GetXDimVisible();
    int yDimVisible = simDomain.GetYDimVisible();
    ChangeCoordinatesToScaledFloat(xcoord, ycoord, xDimVisible, yDimVisible);
    if (x >= xDimVisible || y >= yDimVisible)
    {
        float zcoord = -1.f;
        if (x == xDimVisible)
        {
            zcoord = vbo[(x - 1) + y*MAX_XDIM].z;
        }

        unsigned char b[] = { 0,0,0,0 };
        float color;
        std::memcpy(&color, &b, sizeof(color));
        //clean up surface mesh
        vbo[j] = make_float4(xcoord, ycoord, zcoord, color);
        //clean up floor mesh
        vbo[j+MAX_XDIM*MAX_YDIM] = make_float4(xcoord, ycoord, -1.f, color);
    }
}

__global__ void PhongLighting(float4* vbo, Obstruction *obstructions, 
    float3 cameraPosition, Domain simDomain)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
    unsigned char color[4];
    std::memcpy(color, &(vbo[j].w), sizeof(color));
    unsigned char A = color[3];

    int xDimVisible = simDomain.GetXDimVisible();
    int yDimVisible = simDomain.GetYDimVisible();
    float3 n = { 0, 0, 0 };
    float slope_x = 0.f;
    float slope_y = 0.f;
    float cellSize = 2.f / xDimVisible;
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
    float3 elementPosition = {vbo[j].x,vbo[j].y,vbo[j].z };
    float3 diffuseLightDirection1 = {0.577367, 0.577367, -0.577367 };
    float3 diffuseLightDirection2 = { -0.577367, 0.577367, -0.577367 };
    //float3 cameraPosition = { -1.5, -1.5, 1.5};
    float3 eyeDirection = elementPosition - cameraPosition;
    float3 diffuseLightColor1 = {0.5f, 0.5f, 0.5f};
    float3 diffuseLightColor2 = {0.5f, 0.5f, 0.5f};
    float3 specularLightColor1 = {0.5f, 0.5f, 0.5f};

    float cosTheta1 = -DotProduct(n,diffuseLightDirection1);
    cosTheta1 = cosTheta1 < 0 ? 0 : cosTheta1;
    float cosTheta2 = -DotProduct(n, diffuseLightDirection2);
    cosTheta2 = cosTheta2 < 0 ? 0 : cosTheta2;

    float3 specularLightPosition1 = {-1.5f, -1.5f, 1.5f};
    float3 specularLight1 = elementPosition - specularLightPosition1;
    float3 specularRefection1 = specularLight1 - 2.f*(DotProduct(specularLight1, n)*n);
    Normalize(specularRefection1);
    Normalize(eyeDirection);
    float cosAlpha = -DotProduct(eyeDirection, specularRefection1);
    cosAlpha = cosAlpha < 0 ? 0 : cosAlpha;
    cosAlpha = pow(cosAlpha, 5.f);

    float lightAmbient = 0.3f;
    
    float3 diffuse1  = 0.3f*cosTheta1*diffuseLightColor1;
    float3 diffuse2  = 0.3f*cosTheta2*diffuseLightColor2;
    float3 specular1 = cosAlpha*specularLightColor1;

    color[0] = color[0]*dmin(1.f,(diffuse1.x+diffuse2.x+specular1.x+lightAmbient));
    color[1] = color[1]*dmin(1.f,(diffuse1.y+diffuse2.y+specular1.y+lightAmbient));
    color[2] = color[2]*dmin(1.f,(diffuse1.z+diffuse2.z+specular1.z+lightAmbient));
    color[3] = A;

    std::memcpy(&(vbo[j].w), color, sizeof(color));
}



__global__ void InitializeFloorMesh(float4* vbo, float* floor_d, Domain simDomain)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = MAX_XDIM*MAX_YDIM + x + y*MAX_XDIM;//index on padded mem (pitch in elements)
    int xDimVisible = simDomain.GetXDimVisible();
    int yDimVisible = simDomain.GetYDimVisible();
    float xcoord, ycoord, zcoord;
    ChangeCoordinatesToScaledFloat(xcoord, ycoord, xDimVisible, yDimVisible);
    zcoord = -1.f;
    unsigned char R(255), G(255), B(255), A(255);

    unsigned char b[] = { R, G, B, A };
    float color;
    std::memcpy(&color, &b, sizeof(color));
    vbo[j] = make_float4(xcoord, ycoord, zcoord, color);
}

__device__ float3 ReflectRay(float3 incidentLight, float3 n)
{
    return 2.f*DotProduct(incidentLight, -1.f*n)*n + incidentLight;
}

__device__ float3 RefractRay(float3 incidentLight, float3 n)
{
    float r = 1.0 / WATER_REFRACTIVE_INDEX;
    float c = -(DotProduct(n, incidentLight));
    return r*incidentLight + (r*c - sqrt(1.f - r*r*(1.f - c*c)))*n;
}

__device__ float2 ComputePositionOfLightOnFloor(float4* vbo, float3 incidentLight, 
    const int x, const int y, Domain simDomain, const float waterDepth)
{
    int xDimVisible = simDomain.GetXDimVisible();
    int yDimVisible = simDomain.GetYDimVisible();
    float3 n = { 0, 0, 1 };
    float slope_x = 0.f;
    float slope_y = 0.f;
    float cellSize = 2.f / xDimVisible;
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

    float3 refractedLight = RefractRay(incidentLight, n);
//    float r = 1.0 / WATER_REFRACTIVE_INDEX;
//    float c = -(DotProduct(n, incidentLight));
//    refractedLight = r*incidentLight + (r*c - sqrt(1.f - r*r*(1.f - c*c)))*n;
    const float elemSpaceWaterDepth = (vbo[(x)+(y)*MAX_XDIM].z + 1.f)/2.f*xDimVisible*waterDepth;

    float dx = -refractedLight.x*elemSpaceWaterDepth/refractedLight.z;
    float dy = -refractedLight.y*elemSpaceWaterDepth/refractedLight.z;

    return float2{ (float)x + dx, (float)y + dy };
}

__device__ float ComputeAreaFrom4Points(const float2 &nw, const float2 &ne,
    const float2 &sw, const float2 &se)
{
    float2 vecN = ne - nw;
    float2 vecS = se - sw;
    float2 vecE = ne - se;
    float2 vecW = nw - sw;
    return CrossProductArea(vecN, vecW) + CrossProductArea(vecE, vecS);
}

__global__ void DeformFloorMeshUsingCausticRay(float4* vbo, float3 incidentLight, 
    Obstruction* obstructions, Domain simDomain, const float waterDepth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
    int xDimVisible = simDomain.GetXDimVisible();
    int yDimVisible = simDomain.GetYDimVisible();
    
    if (x < xDimVisible && y < yDimVisible)
    {
        float2 lightPositionOnFloor;
        if (IsInsideObstruction(x, y, obstructions,1.f))
        {
            lightPositionOnFloor = make_float2(x, y);
        }
        else
        {
            lightPositionOnFloor = ComputePositionOfLightOnFloor(vbo, incidentLight,
                x, y, simDomain, waterDepth);
        }

        vbo[j + MAX_XDIM*MAX_YDIM].x = lightPositionOnFloor.x;
        vbo[j + MAX_XDIM*MAX_YDIM].y = lightPositionOnFloor.y;
    }
}

__global__ void ComputeFloorLightIntensitiesFromMeshDeformation(float4* vbo, float* floor_d, 
    Obstruction* obstructions, Domain simDomain)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int xDimVisible = simDomain.GetXDimVisible();
    int yDimVisible = simDomain.GetYDimVisible();

    if (x < xDimVisible-2 && y < yDimVisible-2)
    {
        float2 nw, ne, sw, se;
        int offset = MAX_XDIM*MAX_YDIM;
        nw = make_float2(vbo[(x  )+(y+1)*MAX_XDIM+offset].x, vbo[(x  )+(y+1)*MAX_XDIM+offset].y);
        ne = make_float2(vbo[(x+1)+(y+1)*MAX_XDIM+offset].x, vbo[(x+1)+(y+1)*MAX_XDIM+offset].y);
        sw = make_float2(vbo[(x  )+(y  )*MAX_XDIM+offset].x, vbo[(x  )+(y  )*MAX_XDIM+offset].y);
        se = make_float2(vbo[(x+1)+(y  )*MAX_XDIM+offset].x, vbo[(x+1)+(y  )*MAX_XDIM+offset].y);

        float areaOfLightMeshOnFloor = ComputeAreaFrom4Points(nw, ne, sw, se);
        float lightIntensity = 0.6f / areaOfLightMeshOnFloor;
        atomicAdd(&floor_d[x   + (y  )*MAX_XDIM], lightIntensity*0.25f);
        atomicAdd(&floor_d[x+1 + (y  )*MAX_XDIM], lightIntensity*0.25f);
        atomicAdd(&floor_d[x+1 + (y+1)*MAX_XDIM], lightIntensity*0.25f);
        atomicAdd(&floor_d[x   + (y+1)*MAX_XDIM], lightIntensity*0.25f);
    }
}

__global__ void ApplyCausticLightingToFloor(float4* vbo, float* floor_d, 
    Obstruction* obstructions, Domain simDomain)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = MAX_XDIM*MAX_YDIM + x + y*MAX_XDIM;
    float xcoord, ycoord, zcoord;

    xcoord = vbo[j].x;
    ycoord = vbo[j].y;
    zcoord = vbo[j].z;

    float lightFactor = dmin(1.f,floor_d[x + y*MAX_XDIM]);
    floor_d[x + y*MAX_XDIM] = 0.f;

    unsigned char R = 255.0f;
    unsigned char G = 255.0f;
    unsigned char B = 255.0f;
    unsigned char A = 255.f;

    if (IsInsideObstruction(x, y, obstructions, 0.99f))
    {
        int obstID = FindOverlappingObstruction(x, y, obstructions,0.f);
        if (obstID >= 0)
        {
            float fullObstHeight = -1.f+OBST_HEIGHT;
            if (obstructions[obstID].state == State::NEW)
            {
                zcoord = dmin(fullObstHeight, zcoord + 0.15f);
            }
            else if (obstructions[obstID].state == State::REMOVED)
            {
                obstructions[obstID].u = 0.0f;
                obstructions[obstID].v = 0.0f;
                zcoord = dmax(-1.f, zcoord - 0.15f);
            }
            else if (obstructions[obstID].state == State::ACTIVE)
            {
                zcoord = fullObstHeight;
            }
            else
            {
                obstructions[obstID].u = 0.0f;
                obstructions[obstID].v = 0.0f;
                zcoord = -1.f;
            }
        }
        else
        {
            zcoord = -1.f;
        }
        lightFactor = 0.8f;
        R = 255.f;
        G = 255.f;
        B = 255.f;
    }
    else
    {
        zcoord = -1.f;
    }
    R *= lightFactor;
    G *= lightFactor;
    B *= lightFactor;

    unsigned char b[] = { R, G, B, A };
    float color;
    std::memcpy(&color, &b, sizeof(color));
    int xDimVisible = simDomain.GetXDimVisible();
    int yDimVisible = simDomain.GetYDimVisible();
    ChangeCoordinatesToScaledFloat(xcoord, ycoord, xDimVisible, yDimVisible);
    vbo[j].x = xcoord;
    vbo[j].y = ycoord;
    vbo[j].z = zcoord;
    vbo[j].w = color;
}

__global__ void UpdateObstructionTransientStates(float4* vbo, Obstruction* obstructions)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = MAX_XDIM*MAX_YDIM + x + y*MAX_XDIM;//index on padded mem (pitch in elements)
    float zcoord = vbo[j].z;

    if (IsInsideObstruction(x, y, obstructions, 1.f))
    {
        int obstID = FindOverlappingObstruction(x, y, obstructions);
        if (obstID >= 0)
        {
            if (zcoord > -1.f+OBST_HEIGHT-0.1f)
            {
                obstructions[obstID].state = State::ACTIVE;
            }
            if (zcoord < -1.f+0.1f)
            {
                obstructions[obstID].state = State::INACTIVE;
            }
        }
    }
}

__global__ void RayCast(float4* vbo, float4* rayCastIntersect, float3 rayOrigin,
    float3 rayDir, Obstruction* obstructions, Domain simDomain)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = MAX_XDIM*MAX_YDIM + x + y*MAX_XDIM;
    int xDim = simDomain.GetXDim();
    int yDim = simDomain.GetYDim();
    int xDimVisible = simDomain.GetXDimVisible();
    int yDimVisible = simDomain.GetYDimVisible();

    if (x > 1 && y > 1 && x < xDimVisible - 1 && y < yDimVisible - 1)
    {
        if (IsInsideObstruction(x, y, obstructions, 1.f))
        {
            float3 nw{ vbo[j+MAX_XDIM].x, vbo[j+MAX_XDIM].y, vbo[j+MAX_XDIM].z };
            float3 ne{ vbo[j+MAX_XDIM+1].x, vbo[j+MAX_XDIM+1].y, vbo[j+MAX_XDIM+1].z };
            float3 se{ vbo[j+1].x, vbo[j+1].y, vbo[j+1].z };
            float3 sw{ vbo[j].x, vbo[j].y, vbo[j].z };

            float3 intersection = GetIntersectionOfLineWithTriangle(rayOrigin, rayDir,
                nw, ne, se);
            if (IsPointInsideTriangle(nw, ne, se, intersection))
            {
                float distance = Distance(intersection, rayOrigin);
                if (distance < rayCastIntersect[0].w)
                {
                    rayCastIntersect[0] = { intersection.x, intersection.y,
                        intersection.z, distance };
                }
                //printf("distance in kernel: %f\n", distance);
            }
            else{
                intersection = GetIntersectionOfLineWithTriangle(rayOrigin, rayDir,
                    ne, se, sw);
                if (IsPointInsideTriangle(ne, se, sw, intersection))
                {
                    float distance = Distance(intersection, rayOrigin);
                    if (distance < rayCastIntersect[0].w)
                    {
                        rayCastIntersect[0] = { intersection.x, intersection.y,
                            intersection.z, distance };
                    }
                    //printf("distance in kernel: %f\n", distance);
                }
            }
        }
    }

}

__device__ int GetIntersectWithCubeMap(float3 &intersect, const float3 &rayOrigin, const float3 &rayDir)
{
    float distance = 99999999;
    float temp;
    int side = -1;
    const float maxDim = dmax(MAX_XDIM, MAX_YDIM);
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
        temp = -rayOrigin.x / rayDir.x;
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
        temp = -rayOrigin.y / rayDir.y;
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
        temp = -rayOrigin.z / rayDir.z;
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
    float3 absDir = { abs(rayDir.x), abs(rayDir.y), abs(rayDir.z) };
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
    const float maxDim = dmax(MAX_XDIM, MAX_YDIM);
    if (GetIntersectWithCubeMap(intersect, rayOrigin, rayDir) == 0) //posz
    {
        uv.x =   1024+intersect.x*1024+0.5f;
        uv.y = 2*1024+(1.f-intersect.y)*1024+0.5f;
    }
    else if (GetIntersectWithCubeMap(intersect, rayOrigin, rayDir) == 1) //negx
    {
        uv.x = intersect.y*1024+0.5f;
        uv.y = 1024+intersect.z*1024+0.5f;
    }
    else if (GetIntersectWithCubeMap(intersect, rayOrigin, rayDir) == 3) //posx
    {
        uv.x = 2*1024+(1.f-intersect.y)*1024+0.5f;
        uv.y = 1024+intersect.z*1024+0.5f;
    }
    else if (GetIntersectWithCubeMap(intersect, rayOrigin, rayDir) == 2) //posy
    {
        uv.x = 1024+intersect.x*1024+0.5f;
        uv.y = 1024+intersect.z*1024+0.5f;
    }
    else if (GetIntersectWithCubeMap(intersect, rayOrigin, rayDir) == 4) //negy
    {
        uv.x = 3*1024+(1.f-intersect.x)*1024+0.5f;
        uv.y = 1024+intersect.z*1024+0.5f;
    }
    else //negz - this would be the floor
    {
        uv.x = 1024+intersect.x*1024+0.5f;
        uv.y = (1.f-intersect.y)*1024+0.5f;
    }
    return uv;
}


texture<float4, 2, cudaReadModeElementType> floorTex;
texture<float4, 2, cudaReadModeElementType> envTex;

__global__ void SurfaceRefraction(float4* vbo, Obstruction *obstructions,
    float3 cameraPosition, Domain simDomain, const bool simplified, const float waterDepth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)

    int xDimVisible = simDomain.GetXDimVisible();
    int yDimVisible = simDomain.GetYDimVisible();

    unsigned char color[4];
    std::memcpy(color, &(vbo[j].w), sizeof(color));

    color[3] = 50;

    float3 n = { 0, 0, 1 };
    float slope_x = 0.f;
    float slope_y = 0.f;
    float cellSize = 2.f / xDimVisible;
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
        color[3] = 255;
    }
    Normalize(n);
    const float elemSpaceWaterDepth = (vbo[j].z + 1.f)/2.f*xDimVisible*waterDepth; //non-normalized
    float3 elementPosition = {(float)x,(float)y,elemSpaceWaterDepth }; //non-normalized
    //float3 eyeDirection = xDimVisible*cameraPosition;  //normalized, for ort
    float3 viewingRay = elementPosition/xDimVisible - cameraPosition;  //normalized
    //printf("%f,%f,%f\n", cameraPosition.x, cameraPosition.y, cameraPosition.z);

    Normalize(viewingRay);
    float3 refractedRay = RefractRay(viewingRay, n);
    float3 reflectedRay = ReflectRay(viewingRay, n);
    float cosTheta = dmax(0.f,dmin(1.f,DotProduct(viewingRay, -1.f*n)));
    //if (threadIdx.x == 10) printf("view: %f, %f, %f\n", viewingRay.x, viewingRay.y, viewingRay.z);
    //if (threadIdx.x == 10) printf("n: %f, %f, %f\n", n.x, n.y, n.z);
    float nu = 1.f / WATER_REFRACTIVE_INDEX;
    float r0 = (nu - 1.f)*(nu - 1.f) / ((nu + 1.f)*(nu + 1.f));
    float reflectedRayIntensity = r0 + (1.f - cosTheta)*(1.f - cosTheta)*(1.f - cosTheta)*(1.f - cosTheta)*(1.f - cosTheta)*(1.f - r0);
    
    //const float waterDepth = (vbo[(x)+(y)*MAX_XDIM].z + 1.f)/2.f*xDimVisible*waterDepth;
    float dx = -refractedRay.x*waterDepth/refractedRay.z;
    float dy = -refractedRay.y*waterDepth/refractedRay.z;

    float xf = (float)x + dx;
    float yf = (float)y + dy;
    float xTex = xf/xDimVisible*1024+0.5f;
    float yTex = yf/xDimVisible*1024+0.5f;

    //float envColor = tex2D(envTex, 3, 3);
    //unsigned char environment[4];
    //std::memcpy(&environment, &envColor, sizeof(envColor));
    //float3 reflectedColor;
    //reflectedColor.x = (float)environment[0];
    //reflectedColor.y = (float)environment[1];
    //reflectedColor.z = (float)environment[2];
    //float3 reflectedColor = { 255.f, 255.f, 255.f };

    float2 uvSkyTex = GetUVCoordsForSkyMap(elementPosition, reflectedRay);
    float4 skyColor = tex2D(envTex, uvSkyTex.x, uvSkyTex.y);
    float4 textureColor = tex2D(floorTex, xTex, yTex);

    float3 refractedRayDest = { (float)(x)+dx, (float)(y)+dy, 0 };

    float3 refractionIntersect = { 99999, 99999, 99999 };
    float3 reflectionIntersect = { 99999, 99999, 99999 };
    if (xf > xDimVisible - 1 || xf < 0 || yf > yDimVisible - 1 || yf < 0)
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
            refractedColor[0] = textureColor.x*255.f;
            refractedColor[1] = textureColor.y*255.f;
            refractedColor[2] = textureColor.z*255.f;
            refractedColor[3] = 255;

            unsigned char reflectedColor[4];
            reflectedColor[0] = skyColor.x;
            reflectedColor[1] = skyColor.y;
            reflectedColor[2] = skyColor.z;
            reflectedColor[3] = 255;

            color[0] = (1.f - reflectedRayIntensity)*(float)refractedColor[0] + reflectedRayIntensity*(float)reflectedColor[0];
            color[1] = (1.f - reflectedRayIntensity)*(float)refractedColor[1] + reflectedRayIntensity*(float)reflectedColor[1];
            color[2] = (1.f - reflectedRayIntensity)*(float)refractedColor[2] + reflectedRayIntensity*(float)reflectedColor[2];
            color[3] = 255;
        }
        else
        {
            unsigned char refractedColor[4];
            if (GetCoordFromRayHitOnObst(refractionIntersect, elementPosition, refractedRayDest, obstructions, OBST_HEIGHT / 2.f*xDimVisible))
            {
                std::memcpy(refractedColor, &(vbo[(int)(refractionIntersect.x+0.5f) + (int)(refractionIntersect.y+0.5f)*MAX_XDIM + MAX_XDIM * MAX_YDIM].w),
                    sizeof(refractedColor));
            }
            else
            {
                refractedColor[0] = textureColor.x*255.f;
                refractedColor[1] = textureColor.y*255.f;
                refractedColor[2] = textureColor.z*255.f;
                refractedColor[3] = 255;
            }

            unsigned char reflectedColor[4];
            float3 reflectedRayDest = elementPosition + xDimVisible*reflectedRay;
            if (GetCoordFromRayHitOnObst(reflectionIntersect, elementPosition, reflectedRayDest, obstructions, OBST_HEIGHT / 2.f*xDimVisible))
            {
                std::memcpy(reflectedColor, &(vbo[(int)(reflectionIntersect.x+0.5f) + (int)(reflectionIntersect.y+0.5f)*MAX_XDIM + MAX_XDIM * MAX_YDIM].w),
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

void SetObstructionVelocitiesToZero(Obstruction* obst_h, Obstruction* obst_d, const float scaleFactor)
{
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if ((abs(obst_h[i].u) > 0.f || abs(obst_h[i].v) > 0.f) &&
            obst_h[i].state != State::REMOVED && obst_h[i].state != State::INACTIVE)
        {
            Obstruction obst = obst_h[i];
            obst.x /= scaleFactor;
            obst.y /= scaleFactor;
            obst.r1 /= scaleFactor;
            obst.r2 /= scaleFactor;
            obst.u = 0.f;
            obst.v = 0.f;
            UpdateObstructions << <1, 1 >> >(obst_d,i,obst);
        }
    }
}

void MarchSolution(CudaLbm* cudaLbm)
{
    Domain* simDomain = cudaLbm->GetDomain();
    int xDim = simDomain->GetXDim();
    int yDim = simDomain->GetYDim();
    int tStep = cudaLbm->GetTimeStepsPerFrame();
    float* fA_d = cudaLbm->GetFA();
    float* fB_d = cudaLbm->GetFB();
    int* im_d = cudaLbm->GetImage();
    Obstruction* obst_d = cudaLbm->GetDeviceObst();
    float u = cudaLbm->GetInletVelocity();
    float omega = cudaLbm->GetOmega();

    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    for (int i = 0; i < tStep; i+=2)
    {
        MarchLBM << <grid, threads >> >(fA_d, fB_d, omega, im_d, obst_d, u, *simDomain);
        MarchLBM << <grid, threads >> >(fB_d, fA_d, omega, im_d, obst_d, u, *simDomain);
    }
}

void UpdateSolutionVbo(float4* vis, CudaLbm* cudaLbm, const ContourVariable contVar,
    const float contMin, const float contMax, const ViewMode viewMode, const float waterDepth)
{
    Domain* simDomain = cudaLbm->GetDomain();
    int xDim = simDomain->GetXDim();
    int yDim = simDomain->GetYDim();
    float* f_d = cudaLbm->GetFA();
    int* im_d = cudaLbm->GetImage();
    float u = cudaLbm->GetInletVelocity();

    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    UpdateSurfaceVbo << <grid, threads >> > (vis, f_d, im_d, contVar, contMin, contMax,
        viewMode, u, *simDomain, waterDepth);
}

// ! In order to maintain the same relative positions/sizes of obstructions when the simulation resolution
// ! is changed, host obstruction data is stored relative to the max resolution. When host data is passed
// ! to GPU, the positions and sizes are scaled down based on the current resolution's scaling factor.
void UpdateDeviceObstructions(Obstruction* obst_d, const int targetObstID,
    const Obstruction &newObst, const float scaleFactor)
{
    Obstruction obst = newObst;
    obst.x /= scaleFactor;
    obst.y /= scaleFactor;
    obst.r1 /= scaleFactor;
    obst.r2 /= scaleFactor;
    UpdateObstructions << <1, 1 >> >(obst_d,targetObstID,obst);
}

void CleanUpDeviceVBO(float4* vis, Domain &simDomain)
{
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(MAX_XDIM / BLOCKSIZEX, MAX_YDIM / BLOCKSIZEY);
    CleanUpVBO << <grid, threads>> >(vis, simDomain);
}

void SurfacePhongLighting(float4* vis, Obstruction* obst_d, const float3 cameraPosition,
    Domain &simDomain)
{
    int xDim = simDomain.GetXDim();
    int yDim = simDomain.GetYDim();
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    PhongLighting << <grid, threads>> >(vis, obst_d, cameraPosition, simDomain);
}

void InitializeFloor(float4* vis, float* floor_d, Domain &simDomain)
{
    int xDim = simDomain.GetXDim();
    int yDim = simDomain.GetYDim();
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    InitializeFloorMesh << <grid, threads >> >(vis, floor_d, simDomain);
}

void LightFloor(float4* vis, float* floor_d, Obstruction* obst_d,
    const float3 cameraPosition, Domain &simDomain, const float waterDepth)
{
    int xDim = simDomain.GetXDim();
    int yDim = simDomain.GetYDim();
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    float3 incidentLight1 = { -0.25f, -0.25f, -1.f };
    DeformFloorMeshUsingCausticRay << <grid, threads >> >
        (vis, incidentLight1, obst_d, simDomain, waterDepth);
    ComputeFloorLightIntensitiesFromMeshDeformation << <grid, threads >> >
        (vis, floor_d, obst_d, simDomain);

    ApplyCausticLightingToFloor << <grid, threads >> >(vis, floor_d, obst_d, simDomain);
    UpdateObstructionTransientStates <<<grid,threads>>> (vis, obst_d);

    //phong lighting on floor mesh to shade obstructions
    PhongLighting << <grid, threads>> >(&vis[MAX_XDIM*MAX_YDIM], obst_d, cameraPosition,
        simDomain);
}

int RayCastMouseClick(float3 &rayCastIntersectCoord, float4* vis, float4* rayCastIntersect_d, 
    const float3 &rayOrigin, const float3 &rayDir, Obstruction* obst_d, Domain &simDomain)
{
    int xDim = simDomain.GetXDim();
    int yDim = simDomain.GetYDim();
    float4 intersectionCoord{ 0, 0, 0, 1e6 };
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    RayCast << <grid, threads >> >(vis, rayCastIntersect_d, rayOrigin, rayDir,
        obst_d, simDomain);
    cudaMemcpy(&intersectionCoord, rayCastIntersect_d, sizeof(float4),
        cudaMemcpyDeviceToHost); 
    if (intersectionCoord.w > 1e5) //ray did not intersect with any objects
    {
        return 1;
    }
    else
    {
        cudaMemcpy(&intersectionCoord, rayCastIntersect_d, sizeof(float4),
            cudaMemcpyDeviceToHost); 
        float4 clearSelectedIndex[1];
        clearSelectedIndex[0] = { 0, 0, 0, 1e6 };
        cudaMemcpy(rayCastIntersect_d, &clearSelectedIndex[0], sizeof(float4),
            cudaMemcpyHostToDevice); 
        rayCastIntersectCoord.x = intersectionCoord.x;
        rayCastIntersectCoord.y = intersectionCoord.y;
        rayCastIntersectCoord.z = intersectionCoord.z;
        return 0;
    }
}

void RefractSurface(float4* vis, cudaArray* floorLightTexture, cudaArray* envTexture, Obstruction* obst_d, const glm::vec4 cameraPos,
    Domain &simDomain, const float waterDepth, const bool simplified)
{
    int xDim = simDomain.GetXDim();
    int yDim = simDomain.GetYDim();
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    gpuErrchk(cudaBindTextureToArray(floorTex, floorLightTexture));
    gpuErrchk(cudaBindTextureToArray(envTex, envTexture));
    float3 f3CameraPos = make_float3(cameraPos.x, cameraPos.y, cameraPos.z);
    SurfaceRefraction << <grid, threads>> >(vis, obst_d, f3CameraPos, simDomain, simplified, waterDepth);
}

