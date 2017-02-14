#include <string.h>
#include "math.h"
#include "kernel.h"


/*----------------------------------------------------------------------------------------
 *	Device functions
 */

__global__ void UpdateObstructions(Obstruction* obstructions, const int obstNumber, const Obstruction newObst){
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
        if (obstructions[i].state != Obstruction::INACTIVE)
        {
            float r1 = obstructions[i].r1;
            if (obstructions[i].shape == Obstruction::SQUARE){
                if (abs(x - obstructions[i].x)<r1 + tolerance && abs(y - obstructions[i].y)<r1 + tolerance)
                    return true;
            }
            else if (obstructions[i].shape == Obstruction::CIRCLE){//shift by 0.5 cells for better looks
                float distanceFromCenter = (x + 0.5f - obstructions[i].x)*(x + 0.5f - obstructions[i].x)\
                    + (y + 0.5f - obstructions[i].y)*(y + 0.5f - obstructions[i].y);
                if (distanceFromCenter<(r1+tolerance)*(r1+tolerance)+0.1f)
                    return true;
            }
            else if (obstructions[i].shape == Obstruction::HORIZONTAL_LINE){
                if (abs(x - obstructions[i].x)<r1*2+tolerance &&\
                    abs(y - obstructions[i].y)<LINE_OBST_WIDTH*0.501f+tolerance)
                    return true;
            }
            else if (obstructions[i].shape == Obstruction::VERTICAL_LINE){
                if (abs(y - obstructions[i].y)<r1*2+tolerance &&\
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
        if (obstructions[i].state != Obstruction::INACTIVE)
        {
            float r1 = obstructions[i].r1 + tolerance;
            if (obstructions[i].shape == Obstruction::SQUARE){
                if (abs(x - obstructions[i].x)<r1 && abs(y - obstructions[i].y)<r1)
                    return i;//10;
            }
            else if (obstructions[i].shape == Obstruction::CIRCLE){//shift by 0.5 cells for better looks
                float distanceFromCenter = (x + 0.5f - obstructions[i].x)*(x + 0.5f - obstructions[i].x)\
                    + (y + 0.5f - obstructions[i].y)*(y + 0.5f - obstructions[i].y);
                if (distanceFromCenter<r1*r1+0.1f)
                    return i;//10;
            }
            else if (obstructions[i].shape == Obstruction::HORIZONTAL_LINE){
                if (abs(x - obstructions[i].x)<r1*2 &&\
                    abs(y - obstructions[i].y)<LINE_OBST_WIDTH*0.501f+tolerance)
                    return i;//10;
            }
            else if (obstructions[i].shape == Obstruction::VERTICAL_LINE){
                if (abs(y - obstructions[i].y)<r1*2 &&\
                    abs(x - obstructions[i].x)<LINE_OBST_WIDTH*0.501f+tolerance)
                    return i;//10;
            }
        }
    }
    return -1;
}

__device__ int dmin(const int a, const int b)
{
    if (a<b) return a;
    else return b - 1;
}
__device__ int dmax(const int a)
{
    if (a>-1) return a;
    else return 0;
}
__device__ int dmax(const int a, const int b)
{
    if (a>b) return a;
    else return b;
}
__device__ float dmin(const float a, const float b)
{
    if (a<b) return a;
    else return b;
}
__device__ float dmin(const float a, const float b, const float c, const float d)
{
    return dmin(dmin(a, b), dmin(c, d));
}
__device__ float dmax(const float a)
{
    if (a>0) return a;
    else return 0;
}
__device__ float dmax(const float a, const float b)
{
    if (a>b) return a;
    else return b;
}
__device__ float dmax(const float a, const float b, const float c, const float d)
{
    return dmax(dmax(a, b), dmax(c, d));
}

inline __device__ int f_mem(const int f_num, const int x, const int y,
    const size_t pitch, const int yDim)
{
    return (x + y*pitch) + f_num*pitch*yDim;
}

inline __device__ int f_mem(const int f_num, const int x, const int y)
{

    return (x + y*MAX_XDIM) + f_num*MAX_XDIM*MAX_YDIM;
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
    if (IsPointsOnSameSide(p, a, b, c) &&\
        IsPointsOnSameSide(p, b, a, c) &&\
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
    SimulationParameters simParams)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
    float u, v, rho, usqr;
    rho = 1.f;
    u = uMax;// u_max;// UMAX;
    v = 0.0f;
    usqr = u*u + v*v;
    int offset = MAX_XDIM*MAX_YDIM;

    f[j + 0 * offset] = 0.4444444444f*(rho - 1.5f*usqr);
    f[j + 1 * offset] = 0.1111111111f*(rho + 3.0f*u + 4.5f*u*u - 1.5f*usqr);
    f[j + 2 * offset] = 0.1111111111f*(rho + 3.0f*v + 4.5f*v*v - 1.5f*usqr);
    f[j + 3 * offset] = 0.1111111111f*(rho - 3.0f*u + 4.5f*u*u - 1.5f*usqr);
    f[j + 4 * offset] = 0.1111111111f*(rho - 3.0f*v + 4.5f*v*v - 1.5f*usqr);
    f[j + 5 * offset] = 0.02777777778*(rho + 3.0f*(u + v) + 4.5f*(u + v)*(u + v) - 1.5f*usqr);
    f[j + 6 * offset] = 0.02777777778*(rho + 3.0f*(-u + v) + 4.5f*(-u + v)*(-u + v) - 1.5f*usqr);
    f[j + 7 * offset] = 0.02777777778*(rho + 3.0f*(-u - v) + 4.5f*(-u - v)*(-u - v) - 1.5f*usqr);
    f[j + 8 * offset] = 0.02777777778*(rho + 3.0f*(u - v) + 4.5f*(u - v)*(u - v) - 1.5f*usqr);

    float xcoord, ycoord, zcoord;
    int xDimVisible = simParams.m_xDimVisible;
    int yDimVisible = simParams.m_yDimVisible;
    ChangeCoordinatesToScaledFloat(xcoord, ycoord, xDimVisible, yDimVisible);
    zcoord = 0.f;
    float R(255.f), G(255.f), B(255.f), A(255.f);
    char b[] = { R, G, B, A };
    float color;
    std::memcpy(&color, &b, sizeof(color));
    vbo[j] = make_float4(xcoord, ycoord, zcoord, color);
}

// rho=1.0 BC for east side
__device__ void NeumannEast(float &f0, float &f1, float &f2, float &f3, float &f4,
    float &f5, float &f6, float &f7, float &f8,
    const int y, const int xDim, const int yDim)
{
    if (y == 0){
        f2 = f4;
        f5 = f8;
    }
    else if (y == yDim - 1){
        f4 = f2;
        f8 = f5;
    }
    float u, v, rho;
    v = 0.0;
    rho = 1.0;
    u = -rho + ((f0 + f2 + f4) + 2.0f*f1 + 2.0f*f5 + 2.0f*f8);

    f3 = f1 - u*0.66666667f;
    f7 = f5 + 0.5f*(f2 - f4) - 0.5f*v - u*0.16666667f;
    f6 = f8 - 0.5f*(f2 - f4) + 0.5f*v - u*0.16666667f;
}

// u=uMax BC for east side
__device__ void DirichletWest(float &f0, float &f1, float &f2, float &f3, float &f4,
    float &f5, float &f6, float &f7, float &f8,
    const int y, const int xDim, const int yDim, const float uMax)
{
    if (y == 0){
        f2 = f4;
        f6 = f7;
    }
    else if (y == yDim - 1){
        f4 = f2;
        f7 = f6;
    }
    float u, v;//,rho;
    u = uMax;//*PoisProf(float(y));
    v = 0.0f;//0.0;
    f1 = f3 + u*0.66666667f;
    f5 = f7 - 0.5f*(f2 - f4) + v*0.5f + u*0.166666667f;
    f8 = f6 + 0.5f*(f2 - f4) - v*0.5f + u*0.166666667f;
}

// applies BCs
__device__ void ApplyBCs(float& f0, float& f1, float& f2, float& f3, float& f4,
    float& f5, float& f6, float& f7, float& f8,
    const int y, const int im, const int xDim, const int yDim, const float uMax)
{
    if (im == 2)//NeumannEast
    {
        NeumannEast(f0, f1, f2, f3, f4, f5, f6, f7, f8, y, xDim, yDim);
    }
    else if (im == 3)//DirichletWest
    {
        DirichletWest(f0, f1, f2, f3, f4, f5, f6, f7, f8, y, xDim, yDim, uMax);
    }
    else if (im == 11)//xsymmetry
    {
        f4 = f2;
        f7 = f6;
        f8 = f5;
    }
    else if (im == 12)//xsymmetry
    {
        f2 = f4;
        f6 = f7;
        f5 = f8;
    }
}

__device__ void ComputeFEqs(float *feq, const float rho, const float u, const float v)
{
    float usqr = u*u+v*v;
    feq[0] = 4.0f/9.0f*(rho-1.5f*usqr);
    feq[1] = 1.0f/9.0f*(rho+3.0f*u+4.5f*u*u-1.5f*usqr);
    feq[2] = 1.0f/9.0f*(rho+3.0f*v+4.5f*v*v-1.5f*usqr);
    feq[3] = 1.0f/9.0f*(rho-3.0f*u+4.5f*u*u-1.5f*usqr);
    feq[4] = 1.0f/9.0f*(rho-3.0f*v+4.5f*v*v-1.5f*usqr);
    feq[5] = 1.0f/36.0f*(rho+3.0f*(u+v)+4.5f*(u+v)*(u+v)-1.5f*usqr);
    feq[6] = 1.0f/36.0f*(rho+3.0f*(-u+v)+4.5f*(-u+v)*(-u+v)-1.5f*usqr);
    feq[7] = 1.0f/36.0f*(rho+3.0f*(-u-v)+4.5f*(-u-v)*(-u-v)-1.5f*usqr);
    feq[8] = 1.0f/36.0f*(rho+3.0f*(u-v)+4.5f*(u-v)*(u-v)-1.5f*usqr);
}

__device__ void MovingWall(float &f0, float &f1, float &f2,
    float &f3, float &f4, float &f5,
    float &f6, float &f7, float &f8,
    const float rho, const float u, const float v)
{
    float feq[9];
    ComputeFEqs(feq, rho, u, v);
    f0 = feq[0];
    f1 = feq[1];
    f2 = feq[2];
    f3 = feq[3];
    f4 = feq[4];
    f5 = feq[5];
    f6 = feq[6];
    f7 = feq[7];
    f8 = feq[8];
}


// LBM collision step 
__device__ void LbmCollide(float &f0, float &f1, float &f2,
    float &f3, float &f4, float &f5,
    float &f6, float &f7, float &f8, const float omega, float &Q)
{
    //float rho,u,v;	
    float u, v;
    //rho = f0+f1+f2+f3+f4+f5+f6+f7+f8;
    u = f1 - f3 + f5 - f6 - f7 + f8;
    v = f2 - f4 + f5 + f6 - f7 - f8;
    float m1, m2, m4, m6, m7, m8;

    //	m1 =-4.f*f0 -    f1 -    f2 -    f3 -    f4+ 2.f*f5+ 2.f*f6+ 2.f*f7+ 2.f*f8-(-2.0f*rho+3.0f*(u*u+v*v));
    m1 = -2.f*f0 + f1 + f2 + f3 + f4 + 4.f*f5 + 4.f*f6 + 4.f*f7 + 4.f*f8 - 3.0f*(u*u + v*v);
    //m2 = 4.f*f0 -2.f*f1 -2.f*f2 -2.f*f3 -2.f*f4+     f5+     f6+     f7+     f8-(rho-3.0f*(u*u+v*v)); //ep
    m2 = 3.f*f0 - 3.f*f1 - 3.f*f2 - 3.f*f3 - 3.f*f4 + 3.0f*(u*u + v*v); //ep
    //m4 =        -2.f*f1        + 2.f*f3        +     f5 -    f6 -    f7+     f8-(-u);//qx_eq
    m4 = -f1 + f3 + 2.f*f5 - 2.f*f6 - 2.f*f7 + 2.f*f8;//-(-u);//qx_eq
    m6 = -f2 + f4 + 2.f*f5 + 2.f*f6 - 2.f*f7 - 2.f*f8;//-(-v);//qy_eq
    m7 = f1 - f2 + f3 - f4 - (u*u - v*v);//pxx_eq
    m8 = f5 - f6 + f7 - f8 - (u*v);//pxy_eq

    //	m1 =-4.f*f0 -    f1 -    f2 -    f3 -    f4+ 2.f*f5+ 2.f*f6+ 2.f*f7+ 2.f*f8-(-2.0f*rho+3.0f*(u*u+v*v));
    //	m2 = 4.f*f0 -2.f*f1 -2.f*f2 -2.f*f3 -2.f*f4+     f5+     f6+     f7+     f8-(rho-3.0f*(u*u+v*v)); //ep
    //	m4 =        -2.f*f1        + 2.f*f3        +     f5 -    f6 -    f7+     f8-(-u);//qx_eq
    //	m6 =                -2.f*f2        + 2.f*f4+     f5+     f6 -    f7 -    f8-(-v);//qy_eq
    //	m7 =             f1 -    f2+     f3 -    f4                                -(u*u-v*v);//pxx_eq
    //	m8 =                                             f5 -    f6+     f7 -    f8-(u*v);//pxy_eq

    float usqr = u*u+v*v;
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float feq0 = 4.0f/9.0f*(rho-1.5f*usqr);
    float feq1 = 1.0f/9.0f*(rho+3.0f*u+4.5f*u*u-1.5f*usqr);
    float feq2 = 1.0f/9.0f*(rho+3.0f*v+4.5f*v*v-1.5f*usqr);
    float feq3 = 1.0f/9.0f*(rho-3.0f*u+4.5f*u*u-1.5f*usqr);
    float feq4 = 1.0f/9.0f*(rho-3.0f*v+4.5f*v*v-1.5f*usqr);
    float feq5 = 1.0f/36.0f*(rho+3.0f*(u+v)+4.5f*(u+v)*(u+v)-1.5f*usqr);
    float feq6 = 1.0f/36.0f*(rho+3.0f*(-u+v)+4.5f*(-u+v)*(-u+v)-1.5f*usqr);
    float feq7 = 1.0f/36.0f*(rho+3.0f*(-u-v)+4.5f*(-u-v)*(-u-v)-1.5f*usqr);
    float feq8 = 1.0f/36.0f*(rho+3.0f*(u-v)+4.5f*(u-v)*(u-v)-1.5f*usqr);
    
    
    float qxx = (f1-feq1) + (f3-feq3) + (f5-feq5) + (f6-feq6) + (f7-feq7) + (f8-feq8);
    float qxy = (f5-feq5) - (f6-feq6) + (f7-feq7) - (f8-feq8)                        ;
    float qyy = (f5-feq5) + (f2-feq2) + (f6-feq6) + (f7-feq7) + (f4-feq4) + (f8-feq8);
    Q = sqrt(qxx*qxx + qxy*qxy * 2 + qyy*qyy);
    float tau0 = 1.f / omega;
    float CS = SMAG_CONST;// 0.1f;
    //float tau = 0.5f*tau0 + 0.5f*sqrt(tau0*tau0 + 1000000.0*Q*Q*Q);
    float tau = 0.5f*tau0 + 0.5f*sqrt(tau0*tau0 + 18.f*CS*sqrt(2.f)*Q);
    float omegaTurb = 1.f / tau;

    f0 = f0 - (-m1 + m2)*0.11111111f;//(-4.f*(m1)/36.0f+4.f *(m2)/36.0f);
    f1 = f1 - (-m1*0.027777777f - 0.05555555556f*m2 - 0.16666666667f*m4 + m7*omegaTurb*0.25f);
    f2 = f2 - (-m1*0.027777777f - 0.05555555556f*m2 - 0.16666666667f*m6 - m7*omegaTurb*0.25f);
    f3 = f3 - (-m1*0.027777777f - 0.05555555556f*m2 + 0.16666666667f*m4 + m7*omegaTurb*0.25f);
    f4 = f4 - (-m1*0.027777777f - 0.05555555556f*m2 + 0.16666666667f*m6 - m7*omegaTurb*0.25f);
    f5 = f5 - (0.05555555556f*m1 + m2*0.027777777f + 0.08333333333f*m4 + 0.08333333333f*m6 + m8*omegaTurb*0.25f);
    f6 = f6 - (0.05555555556f*m1 + m2*0.027777777f - 0.08333333333f*m4 + 0.08333333333f*m6 - m8*omegaTurb*0.25f);
    f7 = f7 - (0.05555555556f*m1 + m2*0.027777777f - 0.08333333333f*m4 - 0.08333333333f*m6 + m8*omegaTurb*0.25f);
    f8 = f8 - (0.05555555556f*m1 + m2*0.027777777f + 0.08333333333f*m4 - 0.08333333333f*m6 - m8*omegaTurb*0.25f);

    //MRT Scheme. Might be slightly more stable, but needs more optimization
//	float m[9];
//	float meq[9];
//	float S[9] = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,omega,omega};
//		meq[1] = -2.0*rho+3.0*usqr;//e_eq (uses rho, Yu)
//		meq[2] = rho-3.0*usqr; //epsilon_eq (uses rho, Yu)
//		meq[4] = -u;//qx_eq
//		meq[6] = -v;//qy_eq
//		meq[7] = u*u-v*v;//pxx_eq
//		meq[8] = u*v;//pxy_eq

////		m[0] = 1*f0+ 1*f1+ 1*f2+ 1*f3+ 1*f4+ 1*f5+ 1*f6+ 1*f7+ 1*f8;
//		m[1] =-4*f0+-1*f1+-1*f2+-1*f3+-1*f4+ 2*f5+ 2*f6+ 2*f7+ 2*f8;
//		m[2] = 4*f0+-2*f1+-2*f2+-2*f3+-2*f4+ 1*f5+ 1*f6+ 1*f7+ 1*f8;
////		m[3] = 0*f0+ 1*f1+ 0*f2+-1*f3+ 0*f4+ 1*f5+-1*f6+-1*f7+ 1*f8;
//		m[4] = 0*f0+-2*f1+ 0*f2+ 2*f3+ 0*f4+ 1*f5+-1*f6+-1*f7+ 1*f8;
////		m[5] = 0*f0+ 0*f1+ 1*f2+ 0*f3+-1*f4+ 1*f5+ 1*f6+-1*f7+-1*f8;
//		m[6] = 0*f0+ 0*f1+-2*f2+ 0*f3+ 2*f4+ 1*f5+ 1*f6+-1*f7+-1*f8;
//		m[7] = 0*f0+ 1*f1+-1*f2+ 1*f3+-1*f4+ 0*f5+ 0*f6+ 0*f7+ 0*f8;
//		m[8] = 0*f0+ 0*f1+ 0*f2+ 0*f3+ 0*f4+ 1*f5+-1*f6+ 1*f7+-1*f8;
//		
//		f0-=-4*(m[1]-meq[1])*S[1]/36.0+4 *(m[2]-meq[2])*S[2]/36.0+0 *(m[4]-meq[4])*S[4]/12.0+0 *(m[6]-meq[6])*S[6]/12.0+0 *(m[7]-meq[7])*omega/4.0;//+0 *(m[8]-meq[8])*Omega/4.0;
//		f1-=-  (m[1]-meq[1])*S[1]/36.0+-2*(m[2]-meq[2])*S[2]/36.0+-2*(m[4]-meq[4])*S[4]/12.0+0 *(m[6]-meq[6])*S[6]/12.0+   (m[7]-meq[7])*omega/4.0;//+0 *(m[8]-meq[8])*Omega/4.0;
//		f2-=-  (m[1]-meq[1])*S[1]/36.0+-2*(m[2]-meq[2])*S[2]/36.0+0 *(m[4]-meq[4])*S[4]/12.0+-2*(m[6]-meq[6])*S[6]/12.0+-  (m[7]-meq[7])*omega/4.0;//+0 *(m[8]-meq[8])*Omega/4.0;
//		f3-=-  (m[1]-meq[1])*S[1]/36.0+-2*(m[2]-meq[2])*S[2]/36.0+2 *(m[4]-meq[4])*S[4]/12.0+0 *(m[6]-meq[6])*S[6]/12.0+   (m[7]-meq[7])*omega/4.0;//+0 *(m[8]-meq[8])*Omega/4.0;
//		f4-=-  (m[1]-meq[1])*S[1]/36.0+-2*(m[2]-meq[2])*S[2]/36.0+0 *(m[4]-meq[4])*S[4]/12.0+2 *(m[6]-meq[6])*S[6]/12.0+-  (m[7]-meq[7])*omega/4.0;//+0 *(m[8]-meq[8])*Omega/4.0;
//		f5-=2 *(m[1]-meq[1])*S[1]/36.0+   (m[2]-meq[2])*S[2]/36.0+   (m[4]-meq[4])*S[4]/12.0+   (m[6]-meq[6])*S[6]/12.0+0 *(m[7]-meq[7])*omega/4.0+   (m[8]-meq[8])*omega/4.0;
//		f6-=2 *(m[1]-meq[1])*S[1]/36.0+   (m[2]-meq[2])*S[2]/36.0+-  (m[4]-meq[4])*S[4]/12.0+   (m[6]-meq[6])*S[6]/12.0+0 *(m[7]-meq[7])*omega/4.0+-  (m[8]-meq[8])*omega/4.0;
//		f7-=2 *(m[1]-meq[1])*S[1]/36.0+   (m[2]-meq[2])*S[2]/36.0+-  (m[4]-meq[4])*S[4]/12.0+-  (m[6]-meq[6])*S[6]/12.0+0 *(m[7]-meq[7])*omega/4.0+   (m[8]-meq[8])*omega/4.0;
//		f8-=2 *(m[1]-meq[1])*S[1]/36.0+   (m[2]-meq[2])*S[2]/36.0+   (m[4]-meq[4])*S[4]/12.0+-  (m[6]-meq[6])*S[6]/12.0+0 *(m[7]-meq[7])*omega/4.0+-  (m[8]-meq[8])*omega/4.0;	
            


}


// main LBM function including streaming and colliding
__global__ void MarchLBM(float4* vbo, float* fA, float* fB, const float omega, int *Im,
    Obstruction *obstructions, const int contourVar, const float contMin, const float contMax,
    const int viewMode, const float uMax, SimulationParameters simParams)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = x + y*MAX_XDIM;
    int im = Im[j];
    float u, v, rho;
    int obstId = FindOverlappingObstruction(x, y, obstructions);
    if (obstId >= 0)
    {
        if (obstructions[obstId].u < 1e-5f && obstructions[obstId].v < 1e-5f)
        {
            im = 1; //bounce back
        }
        else
        {
            im = 20; //moving wall
        }
    }
    int xDim = simParams.m_xDim;
    int yDim = simParams.m_yDim;
    float f0, f1, f2, f3, f4, f5, f6, f7, f8;
    f0 = fA[j];
    f1 = fA[f_mem(1, dmax(x - 1), y)];
    f3 = fA[f_mem(3, dmin(x + 1, xDim), y)];
    f2 = fA[f_mem(2, x, y - 1)];
    f5 = fA[f_mem(5, dmax(x - 1), y - 1)];
    f6 = fA[f_mem(6, dmin(x + 1, xDim), y - 1)];
    f4 = fA[f_mem(4, x, y + 1)];
    f7 = fA[f_mem(7, dmin(x + 1, xDim), y + 1)];
    f8 = fA[f_mem(8, dmax(x - 1), dmin(y + 1, yDim))];

    float StrainRate = 0.f;

    if (im == 99)
    {
    //do nothing
    }
    else if (im == 1 || im == 10){//bounce-back condition
        //atomicAdd();   //will need this if force is to be computed
        fB[f_mem(1, x, y)] = f3;
        fB[f_mem(2, x, y)] = f4;
        fB[f_mem(3, x, y)] = f1;
        fB[f_mem(4, x, y)] = f2;
        fB[f_mem(5, x, y)] = f7;
        fB[f_mem(6, x, y)] = f8;
        fB[f_mem(7, x, y)] = f5;
        fB[f_mem(8, x, y)] = f6;
    }
    else if (im == 20)
    {
        u = 0.f; v = 0.f, rho = 1.0f;
        u = obstructions[FindOverlappingObstruction(x, y, obstructions)].u;
        v = obstructions[FindOverlappingObstruction(x, y, obstructions)].v;
        MovingWall(f0, f1, f2, f3, f4, f5, f6, f7, f8, rho, u, v);
        fB[f_mem(1, x, y)] = f3;
        fB[f_mem(2, x, y)] = f4;
        fB[f_mem(3, x, y)] = f1;
        fB[f_mem(4, x, y)] = f2;
        fB[f_mem(5, x, y)] = f7;
        fB[f_mem(6, x, y)] = f8;
        fB[f_mem(7, x, y)] = f5;
        fB[f_mem(8, x, y)] = f6;
    }
    else{
        ApplyBCs(f0, f1, f2, f3, f4, f5, f6, f7, f8, y, im, xDim, yDim, uMax);

        LbmCollide(f0, f1, f2, f3, f4, f5, f6, f7, f8, omega, StrainRate);

        fB[f_mem(0, x, y)] = f0;
        fB[f_mem(1, x, y)] = f1;
        fB[f_mem(2, x, y)] = f2;
        fB[f_mem(3, x, y)] = f3;
        fB[f_mem(4, x, y)] = f4;
        fB[f_mem(5, x, y)] = f5;
        fB[f_mem(6, x, y)] = f6;
        fB[f_mem(7, x, y)] = f7;
        fB[f_mem(8, x, y)] = f8;
    }


    rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    u = f1 - f3 + f5 - f6 - f7 + f8;
    v = f2 - f4 + f5 + f6 - f7 - f8;
    float usqr = u*u+v*v;


    //Prepare data for visualization

    //need to change x,y,z coordinates to NDC (-1 to 1)
    float xcoord, ycoord, zcoord;
    int xDimVisible = simParams.m_xDimVisible;
    int yDimVisible = simParams.m_yDimVisible;
    ChangeCoordinatesToScaledFloat(xcoord, ycoord, xDimVisible, yDimVisible);

    if (im == 1) rho = 1.0;
    zcoord =  (rho - 1.0f) - 0.5f;

    //for color, need to convert 4 bytes (RGBA) to float
    float variableValue = 0.f;

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
        variableValue = StrainRate;
    }


    ////Blue to white color scheme
    unsigned char R = dmin(255.f,dmax(255 * ((variableValue - contMin) / (contMax - contMin))));
    unsigned char G = dmin(255.f,dmax(255 * ((variableValue - contMin) / (contMax - contMin))));
    unsigned char B = 255;// 255 * ((maxValue - variableValue) / (maxValue - minValue));
    unsigned char A = 255;// 255;


    ////Rainbow color scheme
    //signed char R = 255 * ((variableValue - minValue) / (maxValue - minValue));
    //signed char G = 255 - 255 * abs(variableValue - 0.5f*(maxValue + minValue)) / (maxValue - 0.5f*(maxValue + minValue));
    //signed char B = 255 * ((maxValue - variableValue) / (maxValue - minValue));
    //signed char A = 255;

    if (contourVar == ContourVariable::WATER_RENDERING)
    {
        variableValue = StrainRate;
        R = 100; G = 150; B = 255;
        A = 100;
    }
//	if (viewMode == ViewMode::THREE_DIMENSIONAL)
//	{
//		A = 155;
//	}

//	if (x >= (xDimVisible))
//	{
//		zcoord = -1.f;
//		R = 0; G = 0; B = 0;
//	}
    if (im == 1 && viewMode == TWO_DIMENSIONAL){
        R = 204; G = 204; B = 204;
        //zcoord = 0.15f;
    }
    else if (im != 0 || x == xDimVisible-1)
    {
        zcoord = -1.f;
    }
    else
    {
    }

    float color;
    char b[] = { R, G, B, A };
    std::memcpy(&color, &b, sizeof(color));

    //vbo aray to be displayed
    vbo[j] = make_float4(xcoord, ycoord, zcoord, color);
}

__global__ void CleanUpVBO(float4* vbo, SimulationParameters simParams)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
    float xcoord, ycoord;
    int xDimVisible = simParams.m_xDimVisible;
    int yDimVisible = simParams.m_yDimVisible;
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
    float3 cameraPosition, SimulationParameters simParams)
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

    int xDimVisible = simParams.m_xDimVisible;
    int yDimVisible = simParams.m_yDimVisible;
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
        slope_x = (vbo[(x + 1) + y*MAX_XDIM].z - vbo[(x - 1) + y*MAX_XDIM].z) / (2.f*cellSize);
        slope_y = (vbo[(x)+(y + 1)*MAX_XDIM].z - vbo[(x)+(y - 1)*MAX_XDIM].z) / (2.f*cellSize);
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



__global__ void InitializeFloorMesh(float4* vbo, float* floor_d, SimulationParameters simParams)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = MAX_XDIM*MAX_YDIM + x + y*MAX_XDIM;//index on padded mem (pitch in elements)
    int xDimVisible = simParams.m_xDimVisible;
    int yDimVisible = simParams.m_yDimVisible;
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
    const int x, const int y, SimulationParameters simParams)
{
    int j = MAX_XDIM*MAX_YDIM + x + y*MAX_XDIM;//index on padded mem (pitch in elements)
    int xDimVisible = simParams.m_xDimVisible;
    int yDimVisible = simParams.m_yDimVisible;
    float3 n = { 0, 0, 1 };
    float slope_x = 0.f;
    float slope_y = 0.f;
    float cellSize = 2.f / xDimVisible;
    if (x > 0 && x < (xDimVisible - 1) && y > 0 && y < (yDimVisible - 1))
    {
        slope_x = (vbo[(x + 1) + y*MAX_XDIM].z - vbo[(x - 1) + y*MAX_XDIM].z) / (2.f*cellSize);
        slope_y = (vbo[(x)+(y + 1)*MAX_XDIM].z - vbo[(x)+(y - 1)*MAX_XDIM].z) / (2.f*cellSize);
        n.x = -slope_x*2.f*cellSize*2.f*cellSize;
        n.y = -slope_y*2.f*cellSize*2.f*cellSize;
        n.z = 2.f*cellSize*2.f*cellSize;
    }
    Normalize(n);

    Normalize(incidentLight);
    float waterDepth = 80.f;

    float3 refractedLight;
    float r = 1.0 / 1.3f;
    float c = -(DotProduct(n, incidentLight));
    refractedLight = r*incidentLight + (r*c - sqrt(1.f - r*r*(1.f - c*c)))*n;

    float dx = -refractedLight.x*(vbo[(x)+(y)*MAX_XDIM].z + 1.f)*waterDepth / refractedLight.z;
    float dy = -refractedLight.y*(vbo[(x)+(y)*MAX_XDIM].z + 1.f)*waterDepth / refractedLight.z;

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
    Obstruction* obstructions, SimulationParameters simParams)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
    int xDimVisible = simParams.m_xDimVisible;
    int yDimVisible = simParams.m_yDimVisible;
    
    if (x < xDimVisible && y < yDimVisible)
    {
        float2 lightPositionOnFloor;
        if (IsInsideObstruction(x, y, obstructions,1.f))
        {
            lightPositionOnFloor = make_float2(x, y);
        }
        else
        {
            lightPositionOnFloor = ComputePositionOfLightOnFloor(vbo, incidentLight, x, y, simParams);
        }

        vbo[j + MAX_XDIM*MAX_YDIM].x = lightPositionOnFloor.x;
        vbo[j + MAX_XDIM*MAX_YDIM].y = lightPositionOnFloor.y;
    }
}

__global__ void ComputeFloorLightIntensitiesFromMeshDeformation(float4* vbo, float* floor_d, 
    Obstruction* obstructions, SimulationParameters simParams)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int xDimVisible = simParams.m_xDimVisible;
    int yDimVisible = simParams.m_yDimVisible;

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
    Obstruction* obstructions, SimulationParameters simParams)
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
            if (obstructions[obstID].state == Obstruction::NEW)
            {
                zcoord = dmin(-0.3f, zcoord + 0.075f);
            }
            else if (obstructions[obstID].state == Obstruction::REMOVED)
            {
                zcoord = dmax(-1.f, zcoord - 0.075f);
            }
            else if (obstructions[obstID].state == Obstruction::ACTIVE)
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
    int xDimVisible = simParams.m_xDimVisible;
    int yDimVisible = simParams.m_yDimVisible;
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
            if (zcoord > -0.29f)
            {
                obstructions[obstID].state = Obstruction::ACTIVE;
            }
            if (zcoord < -0.99f)
            {
                obstructions[obstID].state = Obstruction::INACTIVE;
            }
        }
    }
}

__global__ void RayCast(float4* vbo, float4* rayCastIntersect, float3 rayOrigin, float3 rayDir,
    Obstruction* obstructions, SimulationParameters simParams)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int j = MAX_XDIM*MAX_YDIM + x + y*MAX_XDIM;
    int xDim = simParams.m_xDim;
    int yDim = simParams.m_yDim;
    int xDimVisible = simParams.m_xDimVisible;
    int yDimVisible = simParams.m_yDimVisible;

    if (x > 1 && y > 1 && x < xDimVisible - 1 && y < yDimVisible - 1)
    {
        if (IsInsideObstruction(x, y, obstructions, 1.f))
        {
            float3 nw{ vbo[j+MAX_XDIM].x, vbo[j+MAX_XDIM].y, vbo[j+MAX_XDIM].z };
            float3 ne{ vbo[j+MAX_XDIM+1].x, vbo[j+MAX_XDIM+1].y, vbo[j+MAX_XDIM+1].z };
            float3 se{ vbo[j+1].x, vbo[j+1].y, vbo[j+1].z };
            float3 sw{ vbo[j].x, vbo[j].y, vbo[j].z };

            float3 intersection = GetIntersectionOfRayWithTriangle(rayOrigin, rayDir, nw, ne, se);
            if (IsPointInsideTriangle(nw, ne, se, intersection))
            {
                float distance = Distance(intersection, rayOrigin);
                if (distance < rayCastIntersect[0].w)
                {
                    rayCastIntersect[0] = { intersection.x, intersection.y, intersection.z, distance };
                }
                //printf("distance in kernel: %f\n", distance);
            }
            else{
                intersection = GetIntersectionOfRayWithTriangle(rayOrigin, rayDir, ne, se, sw);
                if (IsPointInsideTriangle(ne, se, sw, intersection))
                {
                    float distance = Distance(intersection, rayOrigin);
                    if (distance < rayCastIntersect[0].w)
                    {
                        rayCastIntersect[0] = { intersection.x, intersection.y, intersection.z, distance };
                    }
                    //printf("distance in kernel: %f\n", distance);
                }
            }
        }
    }

}

/*----------------------------------------------------------------------------------------
 * End of device functions
 */


void InitializeDomain(float4* vis, float* f_d, int* im_d, const float uMax,
    SimulationParameters* simParams)
{
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(MAX_XDIM) / BLOCKSIZEX), MAX_YDIM / BLOCKSIZEY);
    InitializeLBM << <grid, threads >> >(vis, f_d, im_d, uMax, *simParams);
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


void MarchSolution(float4* vis, float* fA_d, float* fB_d, int* im_d, Obstruction* obst_d,
    const ContourVariable contVar, const float contMin, const float contMax,
    const ViewMode viewMode, const float uMax, const float omega, const int tStep,
    SimulationParameters* simParams, const int paused)
{
    int xDim = simParams->GetXDim(simParams);
    int yDim = simParams->GetYDim(simParams);
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    for (int i = 0; i < tStep; i++)
    {
        MarchLBM << <grid, threads >> >(vis, fA_d, fB_d, omega, im_d, obst_d, contVar,
            contMin, contMax, viewMode, uMax, *simParams);
        if (paused == 0)
        {
            MarchLBM << <grid, threads >> >(vis, fB_d, fA_d, omega, im_d, obst_d, contVar,
                contMin, contMax, viewMode, uMax, *simParams);
        }
    }
}

void UpdateDeviceObstructions(Obstruction* obst_d, const int targetObstID,
    const Obstruction &newObst)
{
    UpdateObstructions << <1, 1 >> >(obst_d,targetObstID,newObst);
}

void CleanUpDeviceVBO(float4* vis, SimulationParameters* simParams)
{
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(MAX_XDIM / BLOCKSIZEX, MAX_YDIM / BLOCKSIZEY);
    CleanUpVBO << <grid, threads>> >(vis, *simParams);
}

void LightSurface(float4* vis, Obstruction* obst_d, const float3 cameraPosition,
    SimulationParameters* simParams)
{
    int xDim = simParams->GetXDim(simParams);
    int yDim = simParams->GetYDim(simParams);
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    PhongLighting << <grid, threads>> >(vis, obst_d, cameraPosition, *simParams);
}

void InitializeFloor(float4* vis, float* floor_d, SimulationParameters* simParams)
{
    int xDim = simParams->GetXDim(simParams);
    int yDim = simParams->GetYDim(simParams);
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    InitializeFloorMesh << <grid, threads >> >(vis, floor_d, *simParams);
}

void LightFloor(float4* vis, float* floor_d, Obstruction* obst_d, const float3 cameraPosition,
    SimulationParameters* simParams)
{
    int xDim = simParams->GetXDim(simParams);
    int yDim = simParams->GetYDim(simParams);
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    float3 incidentLight1 = { -0.25f, -0.25f, -1.f };
    DeformFloorMeshUsingCausticRay << <grid, threads >> >
        (vis, incidentLight1, obst_d, *simParams);
    ComputeFloorLightIntensitiesFromMeshDeformation << <grid, threads >> >
        (vis, floor_d, obst_d, *simParams);

    ApplyCausticLightingToFloor << <grid, threads >> >(vis, floor_d, obst_d, *simParams);
    UpdateObstructionTransientStates <<<grid,threads>>> (vis, obst_d);

    //phong lighting on floor mesh to shade obstructions
    PhongLighting << <grid, threads>> >(&vis[MAX_XDIM*MAX_YDIM], obst_d, cameraPosition, *simParams);
}

int RayCastMouseClick(float3 &rayCastIntersectCoord, float4* vis, float4* rayCastIntersect_d, 
    const float3 rayOrigin, const float3 rayDir, Obstruction* obst_d, SimulationParameters* simParams)
{
    int xDim = simParams->GetXDim(simParams);
    int yDim = simParams->GetYDim(simParams);
    float4 intersectionCoord{ 0, 0, 0, 1e6 };
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid(ceil(static_cast<float>(xDim) / BLOCKSIZEX), yDim / BLOCKSIZEY);
    RayCast << <grid, threads >> >(vis, rayCastIntersect_d, rayOrigin, rayDir, obst_d, *simParams);
    cudaMemcpy(&intersectionCoord, rayCastIntersect_d, sizeof(float4), cudaMemcpyDeviceToHost); 
    if (intersectionCoord.w > 1e5) //ray did not intersect with any objects
    {
        return 1;
    }
    else
    {
        cudaMemcpy(&intersectionCoord, rayCastIntersect_d, sizeof(float4), cudaMemcpyDeviceToHost); 
        float4 clearSelectedIndex[1];
        clearSelectedIndex[0] = { 0, 0, 0, 1e6 };
        cudaMemcpy(rayCastIntersect_d, &clearSelectedIndex[0], sizeof(float4), cudaMemcpyHostToDevice); 
        rayCastIntersectCoord.x = intersectionCoord.x;
        rayCastIntersectCoord.y = intersectionCoord.y;
        rayCastIntersectCoord.z = intersectionCoord.z;
        return 0;
    }
}
