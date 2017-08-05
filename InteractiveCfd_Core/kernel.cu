#define WATER_DEPTH 40.0f
#define WATER_DEPTH_NORMALIZED 0.5f
#define OBST_HEIGHT 2.f

#include "kernel.h"
#include "LbmNode.h"
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
__device__ float3 operator+(const float3 &u, const float3 &v)
{
    return make_float3(u.x + v.x, u.y + v.y, u.z + v.z);
}

__device__ float2 operator+(const float2 &u, const float2 &v)
{
    return make_float2(u.x + v.x, u.y + v.y);
}

__device__ float3 operator-(const float3 &u, const float3 &v)
{
    return make_float3(u.x - v.x, u.y - v.y, u.z - v.z);
}

__device__ float2 operator-(const float2 &u, const float2 &v)

{
    return make_float2(u.x - v.x, u.y - v.y);
}

__device__ float3 operator*(const float3 &u, const float3 &v)
{
    return make_float3(u.x * v.x, u.y * v.y, u.z * v.z);
}

__device__ float3 operator/(const float3 &u, const float3 &v)
{
    return make_float3(u.x / v.x, u.y / v.y, u.z / v.z);
}

__device__ float3 operator*(const float a, const float3 &u)
{
    return make_float3(a*u.x, a*u.y, a*u.z);
}

__device__ float3 operator/(const float3 &u, const float a)
{
    return make_float3(u.x / a, u.y / a, u.z / a);
}


__device__ float DotProduct(const float3 &u, const float3 &v)
{
    return u.x*v.x + u.y*v.y + u.z*v.z;
}

__device__ float3 CrossProduct(const float3 &u, const float3 &v)
{
    return make_float3(u.y*v.z-u.z*v.y, -(u.x*v.z-u.z*v.x), u.x*v.y-u.y*v.x);
}

__device__ float CrossProductArea(const float2 &u, const float2 &v)
{
    return 0.5f*sqrt((u.x*v.y-u.y*v.x)*(u.x*v.y-u.y*v.x));
}

__device__ void Normalize(float3 &u)
{
    float mag = sqrt(DotProduct(u, u));
    u.x /= mag;
    u.y /= mag;
    u.z /= mag;
}

__device__ float Distance(const float3 &u, const float3 &v)
{
    return sqrt(DotProduct((u-v), (u-v)));
}

__device__ bool IsPointsOnSameSide(const float2 &p1, const float2 &p2,
    const float2 &a, const float2 &b)
{
    float cp1 = (b - a).x*(p1 - a).y - (b - a).y*(p1 - a).x;
    float cp2 = (b - a).x*(p2 - a).y - (b - a).y*(p2 - a).x;
    if (cp1*cp2 >= 0)
    {
        return true;
    }
    return false;
}

__device__ bool IsPointInsideTriangle(const float2 &p, const float2 &a,
    const float2 &b, const float2 &c)
{
    if (IsPointsOnSameSide(p, a, b, c) &&
        IsPointsOnSameSide(p, b, a, c) &&
        IsPointsOnSameSide(p, c, a, b))
    {
        return true;
    }
    return false;
}

__device__ bool IsPointInsideTriangle(const float3 &p1, const float3 &p2,
    const float3 &p3, const float3 &q)
{
    float3 n = CrossProduct((p2 - p1), (p3 - p1));

    if (DotProduct(CrossProduct(p2 - p1, q - p1), n) < 0) return false;
    if (DotProduct(CrossProduct(p3 - p2, q - p2), n) < 0) return false;
    if (DotProduct(CrossProduct(p1 - p3, q - p3), n) < 0) return false;

    return true;
}


__device__ float GetDistanceBetweenPointAndLine(const float3 &p1, const float3 &q1, const float3 &q2)
{
    float3 closestPoint = q1+DotProduct(q2 - q1, p1 - q1)/sqrt(DotProduct(q2-q1,q2-q1))*(q2-q1);
    return Distance(p1, closestPoint);
}


__device__ float GetDistanceBetweenTwoLines(const float3 &p1, const float3 &p2, const float3 &q1, const float3 &q2)
{
    float3 n = CrossProduct(p2 - p1, q2 - q1);
    Normalize(n);
    return abs(DotProduct(p1 - q1, n));
}


// ! geomalgorithms.com/a07-_distance.html
__device__ float GetDistanceBetweenTwoLineSegments(const float3 &p1, const float3 &p2, const float3 &q1, const float3 &q2)
{
    float3 u = p2 - p1;
    Normalize(u);
    float3 v = q2 - q1;
    Normalize(v);
    float3 w = p1 - q1;

    float a = DotProduct(u, u);
    float b = DotProduct(u, v);
    float c = DotProduct(v, v);
    float d = DotProduct(u, w);
    float e = DotProduct(v, w);

    float sc = (b*e - c*d) / (a*c - b*b);

    if (sqrt(DotProduct(p2 - p1, p2 - p1)) > sc && sc > 0)
    {
        float3 n = CrossProduct(p2 - p1, q2 - q1);
        Normalize(n);
        return abs(DotProduct(p1 - q1, n));
    }

    float d1, d2, d3, d4;
    d1 = GetDistanceBetweenPointAndLine(p1, q1, q2);
    d2 = GetDistanceBetweenPointAndLine(p2, q1, q2);
    d3 = GetDistanceBetweenPointAndLine(q1, p1, p2);
    d4 = GetDistanceBetweenPointAndLine(q2, p1, p2);
    return dmin(dmin(dmin(d1, d2), d3), d4);
}


//inline __device__ bool DoesRayHitObst(const float3 rayOrigin, const float3 rayDest,
//    Obstruction* obstructions, const float tolerance = 0.f)
//{
//    for (int i = 0; i < MAXOBSTS; i++){
//        if (obstructions[i].state != State::INACTIVE)
//        {
//            float3 obstLineP1 = { obstructions[i].x, obstructions[i].y, -40 };
//            float3 obstLineP2 = { obstructions[i].x, obstructions[i].y, -80 };
//            float dist = GetDistanceBetweenTwoLineSegments(rayOrigin, rayDest, obstLineP1, obstLineP2);
//            if (dist < obstructions[i].r1+0.5f)
//            {
//                return true;
//            }
//        }
//    }
//    return false;
//}

// Gets intersection of ray with plane created by triangle
//p1, p2, p3 should be in clockwise order
__device__ float3 GetIntersectionOfRayWithTriangle(const float3 &rayOrigin,
    float3 rayDir, const float3 &p1, const float3 &p2, const float3 &p3)
{
    //plane of triangle
    float3 n = CrossProduct((p2 - p1), (p3 - p1));
    Normalize(n);
    float d = DotProduct(n, p1); //coefficient "d" of plane equation (n.x = d)

    Normalize(rayDir);
    float t = (d-DotProduct(n,rayOrigin))/(DotProduct(n,rayDir));

    return rayOrigin + t*rayDir;
}

// Only update intersect reference if intersect is inside the rectangle, and is closer to rayOrigin than previous value
__device__ bool IntersectRayWithRect(float3 &intersect, float3 rayOrigin, float3 rayDir, 
    float3 topLeft, float3 topRight, float3 bottomRight, float3 bottomLeft)
{
    float3 temp;
    temp = GetIntersectionOfRayWithTriangle(rayOrigin, rayDir, topLeft, topRight, bottomRight);
    if (IsPointInsideTriangle(topLeft, topRight, bottomRight, temp))
    {
        if (Distance(temp, rayOrigin) < Distance(intersect, rayOrigin))
        {
            intersect = temp;
            return true;
        }
    }
    temp = GetIntersectionOfRayWithTriangle(rayOrigin, rayDir, bottomRight, bottomLeft, topLeft);
    if (IsPointInsideTriangle(bottomRight, bottomLeft, topLeft, temp))
    {
        if (Distance(temp, rayOrigin) < Distance(intersect, rayOrigin))
        {
            intersect = temp;
            return true;
        }
    }
    return false;
}

__device__ bool GetCoordFromRayHitOnObst(float3 &intersect, const float3 rayOrigin, const float3 rayDest,
    Obstruction* obstructions, const float tolerance = 0.f)
{
    float3 rayDir = rayDest - rayOrigin;
    bool hit = false;
    for (int i = 0; i < MAXOBSTS; i++){
        if (obstructions[i].state != State::INACTIVE)
        {
            float obstHeight = WATER_DEPTH*1.5f;
            float3 obstLineP1 = { obstructions[i].x, obstructions[i].y, 0 };
            float3 obstLineP2 = { obstructions[i].x, obstructions[i].y, obstHeight };
            float dist = GetDistanceBetweenTwoLineSegments(rayOrigin, rayDest, obstLineP1, obstLineP2);
            if (dist < obstructions[i].r1*2.f+0.5f)
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

                    hit = hit | IntersectRayWithRect(intersect, rayOrigin, rayDir, nwt, swt, swb, nwb);
                    hit = hit | IntersectRayWithRect(intersect, rayOrigin, rayDir, swt, set, seb, swb);
                    hit = hit | IntersectRayWithRect(intersect, rayOrigin, rayDir, set, net, neb, seb);
                    hit = hit | IntersectRayWithRect(intersect, rayOrigin, rayDir, net, nwt, nwb, neb);
                }
                else if (obstructions[i].shape == Shape::CIRCLE)
                {
                    
                }
                else if (obstructions[i].shape == Shape::VERTICAL_LINE)
                {
                    float r1 = LINE_OBST_WIDTH*0.501f;
                    float r2 = obstructions[i].r1*2.f;
                    float3 swt = { x - r1, y - r2, obstHeight };
                    float3 set = { x + r1, y - r2, obstHeight };
                    float3 nwt = { x - r1, y + r2, obstHeight };
                    float3 net = { x + r1, y + r2, obstHeight };
                    float3 swb = { x - (r1+0.5f), y - (r2+0.5f), 0.f };
                    float3 seb = { x + (r1+0.5f), y - (r2+0.5f), 0.f };
                    float3 nwb = { x - (r1+0.5f), y + (r2+0.5f), 0.f };
                    float3 neb = { x + (r1+0.5f), y + (r2+0.5f), 0.f };

                    hit = hit | IntersectRayWithRect(intersect, rayOrigin, rayDir, nwt, swt, swb, nwb);
                    hit = hit | IntersectRayWithRect(intersect, rayOrigin, rayDir, swt, set, seb, swb);
                    hit = hit | IntersectRayWithRect(intersect, rayOrigin, rayDir, set, net, neb, seb);
                    hit = hit | IntersectRayWithRect(intersect, rayOrigin, rayDir, net, nwt, nwb, neb);
                }
                else if (obstructions[i].shape == Shape::HORIZONTAL_LINE)
                {
                    float r1 = obstructions[i].r1*2.f;
                    float r2 = LINE_OBST_WIDTH*0.501f;
                    float3 swt = { x - r1, y - r2, obstHeight };
                    float3 set = { x + r1, y - r2, obstHeight };
                    float3 nwt = { x - r1, y + r2, obstHeight };
                    float3 net = { x + r1, y + r2, obstHeight };
                    float3 swb = { x - (r1+0.5f), y - (r2+0.5f), 0.f };
                    float3 seb = { x + (r1+0.5f), y - (r2+0.5f), 0.f };
                    float3 nwb = { x - (r1+0.5f), y + (r2+0.5f), 0.f };
                    float3 neb = { x + (r1+0.5f), y + (r2+0.5f), 0.f };

                    hit = hit | IntersectRayWithRect(intersect, rayOrigin, rayDir, nwt, swt, swb, nwb);
                    hit = hit | IntersectRayWithRect(intersect, rayOrigin, rayDir, swt, set, seb, swb);
                    hit = hit | IntersectRayWithRect(intersect, rayOrigin, rayDir, set, net, neb, seb);
                    hit = hit | IntersectRayWithRect(intersect, rayOrigin, rayDir, net, nwt, nwb, neb);
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
    float R(255.f), G(255.f), B(255.f), A(255.f);
    char b[] = { R, G, B, A };
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
        if (obstructions[obstId].u < 1e-5f && obstructions[obstId].v < 1e-5f)
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
    const int viewMode, const float uMax, Domain simDomain)
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
    zcoord =  (-1.f+WATER_DEPTH_NORMALIZED) + (rho - 1.0f);

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
    char b[] = { R, G, B, A };
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
        unsigned char b[] = { 0,0,0,255 };
        float color;
        std::memcpy(&color, &b, sizeof(color));
        //clean up surface mesh
        vbo[j] = make_float4(xcoord, ycoord, -1.f, color);
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
    float R, G, B, A;
    R = color[0];
    G = color[1];
    B = color[2];
    A = color[3];

    int xDimVisible = simDomain.GetXDimVisible();
    int yDimVisible = simDomain.GetYDimVisible();
    float3 n = { 0, 0, 0 };
    float slope_x = 0.f;
    float slope_y = 0.f;
    float cellSize = 2.f / xDimVisible;
    if (x == 0)
    {
        n.x = -1.f;
    }
    else if (y == 0)
    {
        n.y = -1.f;
    }
    else if (x >= xDimVisible - 1)
    {
        n.x = 1.f;
    }
    else if (y >= yDimVisible - 1)
    {
        n.y = 1.f;
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
    
    float3 diffuse1  = 0.1f*cosTheta1*diffuseLightColor1;
    float3 diffuse2  = 0.1f*cosTheta2*diffuseLightColor2;
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

    char b[] = { R, G, B, A };
    float color;
    std::memcpy(&color, &b, sizeof(color));
    vbo[j] = make_float4(xcoord, ycoord, zcoord, color);
}

__device__ float2 ComputePositionOfLightOnFloor(float4* vbo, float3 incidentLight, 
    const int x, const int y, Domain simDomain)
{
    int j = MAX_XDIM*MAX_YDIM + x + y*MAX_XDIM;//index on padded mem (pitch in elements)
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

    float3 refractedLight;
    float r = 1.0 / 1.3f;
    float c = -(DotProduct(n, incidentLight));
    refractedLight = r*incidentLight + (r*c - sqrt(1.f - r*r*(1.f - c*c)))*n;
    const float waterDepth = (vbo[(x)+(y)*MAX_XDIM].z + 1.f)/WATER_DEPTH_NORMALIZED*WATER_DEPTH;

    float dx = -refractedLight.x*waterDepth/refractedLight.z;
    float dy = -refractedLight.y*waterDepth/refractedLight.z;

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
    Obstruction* obstructions, Domain simDomain)
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
                x, y, simDomain);
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
        float lightIntensity = 0.3f / areaOfLightMeshOnFloor;
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

    unsigned char R = 50.f;
    unsigned char G = 120.f;
    unsigned char B = 255.f;
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
                zcoord = dmax(-1.f, zcoord - 0.15f);
            }
            else if (obstructions[obstID].state == State::ACTIVE)
            {
                zcoord = fullObstHeight;
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

    char b[] = { R, G, B, A };
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
    float xcoord, ycoord, zcoord;

    zcoord = vbo[j].z;

    if (IsInsideObstruction(x, y, obstructions, 1.f))
    {
        int obstID = FindOverlappingObstruction(x, y, obstructions);
        if (obstID >= 0)
        {
            if (zcoord > -1.f+OBST_HEIGHT)
            {
                obstructions[obstID].state = State::ACTIVE;
            }
            if (zcoord < -0.99f)
            {
                obstructions[obstID].state = State::INACTIVE;
            }
        }
    }
}

__device__ int findIntersectWithObst(float4* vbo, float4* rayCastIntersect, float3 rayOrigin,
    float3 rayDir, Obstruction* obstructions, Domain simDomain)
{

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

            float3 intersection = GetIntersectionOfRayWithTriangle(rayOrigin, rayDir,
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
                intersection = GetIntersectionOfRayWithTriangle(rayOrigin, rayDir,
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

texture<float4, 2, cudaReadModeElementType> floorTex;

__global__ void SurfaceRefraction(float4* vbo, Obstruction *obstructions,
    float3 cameraPosition, Domain simDomain)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)

    int xDimVisible = simDomain.GetXDimVisible();
    int yDimVisible = simDomain.GetYDimVisible();

    unsigned char color[4];
    std::memcpy(color, &(vbo[j].w), sizeof(color));

    color[3] = 50;

    float3 n = { 0, 0, 0 };
    float slope_x = 0.f;
    float slope_y = 0.f;
    float cellSize = 2.f / xDimVisible;
    if (x == 0)
    {
        n.x = -1.f;
    }
    else if (y == 0)
    {
        n.y = -1.f;
    }
    else if (x >= xDimVisible - 1)
    {
        n.x = 1.f;
    }
    else if (y >= yDimVisible - 1)
    {
        n.y = 1.f;
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
    float zPos = (vbo[j].z + 1.f)/WATER_DEPTH_NORMALIZED*WATER_DEPTH;
    float3 elementPosition = {x,y,zPos };
    elementPosition.x /= xDimVisible;
    elementPosition.y /= xDimVisible;
    elementPosition.z /= xDimVisible;
    float3 eyeDirection = elementPosition - cameraPosition;  //normalized

    float2 lightPositionOnFloor = ComputePositionOfLightOnFloor(vbo, eyeDirection, x, y, simDomain);  //non-normalized

    float xTex = lightPositionOnFloor.x/xDimVisible*1024+0.5f;
    float yTex = lightPositionOnFloor.y/xDimVisible*1024+0.5f;

    float4 textureColor = tex2D(floorTex, xTex, yTex);
    color[0] = textureColor.x*255.f;
    color[1] = textureColor.y*255.f;
    color[2] = textureColor.z*255.f;


    float3 refractedRayDest = { lightPositionOnFloor.x, lightPositionOnFloor.y, 0.f }; //non-normalized

    //bool hitsObst = DoesRayHitObst(xDimVisible*elementPosition, refractedRayDest, obstructions);

    float3 intersect = { 99999, 99999, 99999 };
    if (GetCoordFromRayHitOnObst(intersect, xDimVisible*elementPosition, refractedRayDest, obstructions))
    {
        //color[0] = 200;
        //color[1] = 0;
        //color[2] = 0;
        //color[3] = 255;
        //std::memcpy(&(vbo[j].w), color, sizeof(color));
        std::memcpy(&(vbo[j].w), &(vbo[(int)(intersect.x+0.5f) + (int)(intersect.y+0.5f)*MAX_XDIM + MAX_XDIM * MAX_YDIM].w), sizeof(color));
    }
    else
    {
        std::memcpy(&(vbo[j].w), color, sizeof(color));
    }


//    if (x > 1 && y > 1 && x < xDimVisible - 1 && y < yDimVisible - 1)
//    {
//        if (IsInsideObstruction(x, y, obstructions, 1.f))
//        {
//            float3 nw{ vbo[j+MAX_XDIM].x, vbo[j+MAX_XDIM].y, vbo[j+MAX_XDIM].z };
//            float3 ne{ vbo[j+MAX_XDIM+1].x, vbo[j+MAX_XDIM+1].y, vbo[j+MAX_XDIM+1].z };
//            float3 se{ vbo[j+1].x, vbo[j+1].y, vbo[j+1].z };
//            float3 sw{ vbo[j].x, vbo[j].y, vbo[j].z };

//            float3 intersection = GetIntersectionOfRayWithTriangle(cameraPosition, eyeDirection,
//                nw, ne, se);
//            if (IsPointInsideTriangle(nw, ne, se, intersection))
//            {
//                    hitsObst = true;
//                //printf("distance in kernel: %f\n", distance);
//            }
//            else{
//                intersection = GetIntersectionOfRayWithTriangle(cameraPosition, eyeDirection,
//                    ne, se, sw);
//                if (IsPointInsideTriangle(ne, se, sw, intersection))
//                {
//                        hitsObst = true;
//                    //printf("distance in kernel: %f\n", distance);
//                }
//            }
//        }
//    }

//    if (hitsObst == true)
//    {
//        color[0] = 204;
//        color[1] = 204;
//        color[2] = 204;
//    }













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

void SetObstructionVelocitiesToZero(Obstruction* obst_h, Obstruction* obst_d)
{
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if (obst_h[i].u > 0.f || obst_h[i].v > 0.f)
        {
            obst_h[i].u = 0.f;
            obst_h[i].v = 0.f;
            UpdateObstructions << <1, 1 >> >(obst_d,i,obst_h[i]);
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
    for (int i = 0; i < tStep; i++)
    {
        MarchLBM << <grid, threads >> >(fA_d, fB_d, omega, im_d, obst_d, u, *simDomain);
        if (!cudaLbm->IsPaused())
        {
            MarchLBM << <grid, threads >> >(fB_d, fA_d, omega, im_d, obst_d, u, *simDomain);
        }
    }
}

void UpdateSolutionVbo(float4* vis, CudaLbm* cudaLbm, const ContourVariable contVar,
    const float contMin, const float contMax, const ViewMode viewMode)
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
        viewMode, u, *simDomain);
}

void UpdateDeviceObstructions(Obstruction* obst_d, const int targetObstID,
    const Obstruction &newObst)
{
    UpdateObstructions << <1, 1 >> >(obst_d,targetObstID,newObst);
}

void CleanUpDeviceVBO(float4* vis, Domain &simDomain)
{
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(MAX_XDIM / BLOCKSIZEX, MAX_YDIM / BLOCKSIZEY);
    CleanUpVBO << <grid, threads>> >(vis, simDomain);
}

void LightSurface(float4* vis, Obstruction* obst_d, const float3 cameraPosition,
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
    const float3 cameraPosition, Domain &simDomain)
{
    int xDim = simDomain.GetXDim();
    int yDim = simDomain.GetYDim();
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    float3 incidentLight1 = { -0.25f, -0.25f, -1.f };
    DeformFloorMeshUsingCausticRay << <grid, threads >> >
        (vis, incidentLight1, obst_d, simDomain);
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

void RefractSurface(float4* vis, cudaArray* floorTexture, Obstruction* obst_d, const glm::vec4 cameraPos,
    Domain &simDomain)
{
    int xDim = simDomain.GetXDim();
    int yDim = simDomain.GetYDim();
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    cudaBindTextureToArray(floorTex, floorTexture);
    float3 f3CameraPos = make_float3(cameraPos.x, cameraPos.y, cameraPos.z);
    SurfaceRefraction << <grid, threads>> >(vis, obst_d, f3CameraPos, simDomain);
}

