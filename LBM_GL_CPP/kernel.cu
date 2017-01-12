#include <string.h>
#include "math.h"
#include "kernel.h"

extern int g_xDim;
extern int g_yDim;
extern int g_paused;

//float uMax = 0.06f;
//float omega = 1.9f;

//int BLOCKSIZEX = 64;
//int BLOCKSIZEY = 1;

//grid and threads for CUDA

//int nBlocks = ((g_xDim + BLOCKSIZEX - 1) / BLOCKSIZEX)*(g_yDim / BLOCKSIZEY);
//int n = nBlocks*BLOCKSIZEX*BLOCKSIZEY;


/*----------------------------------------------------------------------------------------
 *	Device functions
 */

__global__ void UpdateObstructions(Obstruction* obstructions, int obstNumber, float r, float x, float y, Obstruction::Shape shape){
	obstructions[obstNumber].shape = shape;
	obstructions[obstNumber].r1 = r;
	obstructions[obstNumber].x = x;
	obstructions[obstNumber].y = y;
}

inline __device__ bool isInsideObstruction(int x, int y, Obstruction* obstructions, float tolerance = 0.f){
	for (int i = 0; i < MAXOBSTS; i++){
		float r1 = obstructions[i].r1 + tolerance;
		if (obstructions[i].shape == Obstruction::SQUARE){//square
			if (abs(x - obstructions[i].x)<r1 && abs(y - obstructions[i].y)<r1)
				return true;//10;
		}
		else if (obstructions[i].shape == Obstruction::CIRCLE){//circle. shift by 0.5 cells for better looks
			if ((x+0.5f - obstructions[i].x)*(x+0.5f - obstructions[i].x)+(y+0.5f - obstructions[i].y)*(y+0.5f - obstructions[i].y)
					<r1*r1+0.1f)
				return true;//10;
		}
		else if (obstructions[i].shape == Obstruction::HORIZONTAL_LINE){//horizontal line
			if (abs(x - obstructions[i].x)<r1*2 && abs(y - obstructions[i].y)<LINE_OBST_WIDTH*0.501f+tolerance)
				return true;//10;
		}
		else if (obstructions[i].shape == Obstruction::VERTICAL_LINE){//vertical line
			if (abs(y - obstructions[i].y)<r1*2 && abs(x - obstructions[i].x)<LINE_OBST_WIDTH*0.501f+tolerance)
				return true;//10;
		}
	}
	return false;
}

//defines BCs for grid
// no longer in use. 10/29/2016
//inline __device__ int ImageFcn(int x, int y, obstruction* obstructions){
//	//if(y == 0 || x == XDIM-1 || y == YDIM-1)
//	if (x < 0.1f)
//		return 3;//west
//	else if ((XDIM - x) < 1.1f)
//		return 2;//east
//	else if ((YDIM - y) < 1.1f)
//		return 11;//11;//xsymmetry top
//	else if (y < 0.1f)
//		return 12;//12;//xsymmetry bottom

//	for (int i = 0; i < MAXOBSTS; i++){
//		if (abs(x - obstructions[i].x)<obstructions[i].r && abs(y - obstructions[i].y)<obstructions[i].r)
//			return 1;//10;
//	}
//	return 0;
//}

__device__ int dmin(int a, int b)
{
	if (a<b) return a;
	else return b - 1;
}
__device__ int dmax(int a)
{
	if (a>-1) return a;
	else return 0;
}
__device__ int dmax(int a, int b)
{
	if (a>b) return a;
	else return b;
}
__device__ float dmin(float a, float b)
{
	if (a<b) return a;
	else return b;
}
__device__ float dmin(float a, float b, float c, float d)
{
	return dmin(dmin(a, b), dmin(c, d));
}
__device__ float dmax(float a)
{
	if (a>0) return a;
	else return 0;
}
__device__ float dmax(float a, float b)
{
	if (a>b) return a;
	else return b;
}
__device__ float dmax(float a, float b, float c, float d)
{
	return dmax(dmax(a, b), dmax(c, d));
}

inline __device__ int f_mem(int f_num, int x, int y, size_t pitch, int yDim)
{

	return (x + y*pitch) + f_num*pitch*yDim;
}

inline __device__ int f_mem(int f_num, int x, int y)
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

__device__ float DotProduct(float3 u, float3 v)
{
	return u.x*v.x + u.y*v.y + u.z*v.z;
}

__device__ float3 CrossProduct(float3 u, float3 v)
{
	return make_float3(u.y*v.z-u.z*v.y, -(u.x*v.z-u.z*v.x), u.x*v.y-u.y*v.x);
}

__device__ float CrossProductArea(float2 u, float2 v)
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

__device__ float Distance(float3 u, float3 v)
{
	float mag = sqrt(DotProduct((u-v), (u-v)));
}

__device__ bool IsPointsOnSameSide(float2 p1, float2 p2, float2 a, float2 b)
{
	float cp1 = (b - a).x*(p1 - a).y - (b - a).y*(p1 - a).x;
	float cp2 = (b - a).x*(p2 - a).y - (b - a).y*(p2 - a).x;
	if (cp1*cp2 >= 0)
	{
		return true;
	}
	return false;
}

__device__ bool IsPointInsideTriangle(float2 p, float2 a, float2 b, float2 c)
{
	if (IsPointsOnSameSide(p, a, b, c) && IsPointsOnSameSide(p, b, a, c) && IsPointsOnSameSide(p, c, a, b))
	{
		return true;
	}
	return false;
}




__device__	void ChangeCoordinatesToNDC(float &xcoord,float &ycoord, int xDimVisible, int yDimVisible)
{
	xcoord = threadIdx.x + blockDim.x*blockIdx.x;
	ycoord = threadIdx.y + blockDim.y*blockIdx.y;
	xcoord /= xDimVisible *0.5f;
	ycoord /= yDimVisible *0.5f;//(float)(blockDim.y*gridDim.y);
	xcoord -= 1.0;// xdim / maxDim;
	ycoord -= 1.0;// ydim / maxDim;
}

__device__	void ChangeCoordinatesToScaledFloat(float &xcoord,float &ycoord, int xDimVisible, int yDimVisible)
{
	xcoord = threadIdx.x + blockDim.x*blockIdx.x;
	ycoord = threadIdx.y + blockDim.y*blockIdx.y;
	xcoord /= xDimVisible *0.5f;
	ycoord /= xDimVisible *0.5f;//(float)(blockDim.y*gridDim.y);
	xcoord -= 1.0;// xdim / maxDim;
	ycoord -= 1.0;// ydim / maxDim;
}

// Initialize domain using constant velocity
__global__ void initialize_single(float4* pos, float *f, int *Im, int xDim, int yDim, float uMax, int xDimVisible, int yDimVisible) //obstruction* obstruction)//pitch in elements
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
	float u, v, rho, usqr;
	rho = 1.f;
	u = uMax;// u_max;// UMAX;
	v = 0.0f;
	usqr = u*u + v*v;

	f[j + 0 * MAX_XDIM*MAX_YDIM] = 0.4444444444f*(rho - 1.5f*usqr);
	f[j + 1 * MAX_XDIM*MAX_YDIM] = 0.1111111111f*(rho + 3.0f*u + 4.5f*u*u - 1.5f*usqr);
	f[j + 2 * MAX_XDIM*MAX_YDIM] = 0.1111111111f*(rho + 3.0f*v + 4.5f*v*v - 1.5f*usqr);
	f[j + 3 * MAX_XDIM*MAX_YDIM] = 0.1111111111f*(rho - 3.0f*u + 4.5f*u*u - 1.5f*usqr);
	f[j + 4 * MAX_XDIM*MAX_YDIM] = 0.1111111111f*(rho - 3.0f*v + 4.5f*v*v - 1.5f*usqr);
	f[j + 5 * MAX_XDIM*MAX_YDIM] = 0.02777777778*(rho + 3.0f*(u + v) + 4.5f*(u + v)*(u + v) - 1.5f*usqr);
	f[j + 6 * MAX_XDIM*MAX_YDIM] = 0.02777777778*(rho + 3.0f*(-u + v) + 4.5f*(-u + v)*(-u + v) - 1.5f*usqr);
	f[j + 7 * MAX_XDIM*MAX_YDIM] = 0.02777777778*(rho + 3.0f*(-u - v) + 4.5f*(-u - v)*(-u - v) - 1.5f*usqr);
	f[j + 8 * MAX_XDIM*MAX_YDIM] = 0.02777777778*(rho + 3.0f*(u - v) + 4.5f*(u - v)*(u - v) - 1.5f*usqr);

	float xcoord, ycoord, zcoord;
	ChangeCoordinatesToScaledFloat(xcoord, ycoord, xDimVisible, yDimVisible);
	zcoord = 0.f;
	float R(255.f), G(255.f), B(255.f), A(255.f);
	char b[] = { R, G, B, A };
	float color;
	std::memcpy(&color, &b, sizeof(color));
	pos[j] = make_float4(xcoord, ycoord, zcoord, color);
}

// rho=1.0 BC for east side
__device__ void NeumannEast(float &f0, float &f1, float &f2,
	float &f3, float &f4, float &f5,
	float &f6, float &f7, float &f8, int y, int xDim, int yDim)
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
__device__ void DirichletWest(float &f0, float &f1, float &f2,
	float &f3, float &f4, float &f5,
	float &f6, float &f7, float &f8, int y, int xDim, int yDim, float uMax)
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
__device__ void boundaries(float& f0, float& f1, float& f2,
	float& f3, float& f4, float& f5,
	float& f6, float& f7, float& f8,
	int y, int im, int xDim, int yDim, float uMax)
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

// LBM collision step using MRT method
__device__ void mrt_collide(float &f0, float &f1, float &f2,
	float &f3, float &f4, float &f5,
	float &f6, float &f7, float &f8, float omega, float &Q)
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
	float tau = 0.5f*tau0 + 0.5f*sqrt(tau0*tau0 + 18.f*CS*sqrt(2.f)*Q);
	omega = 1.f / tau;

	f0 = f0 - (-m1 + m2)*0.11111111f;//(-4.f*(m1)/36.0f+4.f *(m2)/36.0f);
	f1 = f1 - (-m1*0.027777777f - 0.05555555556f*m2 - 0.16666666667f*m4 + m7*omega*0.25f);
	f2 = f2 - (-m1*0.027777777f - 0.05555555556f*m2 - 0.16666666667f*m6 - m7*omega*0.25f);
	f3 = f3 - (-m1*0.027777777f - 0.05555555556f*m2 + 0.16666666667f*m4 + m7*omega*0.25f);
	f4 = f4 - (-m1*0.027777777f - 0.05555555556f*m2 + 0.16666666667f*m6 - m7*omega*0.25f);
	f5 = f5 - (0.05555555556f*m1 + m2*0.027777777f + 0.08333333333f*m4 + 0.08333333333f*m6 + m8*omega*0.25f);
	f6 = f6 - (0.05555555556f*m1 + m2*0.027777777f - 0.08333333333f*m4 + 0.08333333333f*m6 - m8*omega*0.25f);
	f7 = f7 - (0.05555555556f*m1 + m2*0.027777777f - 0.08333333333f*m4 - 0.08333333333f*m6 + m8*omega*0.25f);
	f8 = f8 - (0.05555555556f*m1 + m2*0.027777777f + 0.08333333333f*m4 - 0.08333333333f*m6 - m8*omega*0.25f);
}


// main LBM function including streaming and colliding
__global__ void mrt_d_single(float4* pos, float* fA, float* fB,
	float omega, int *Im, Obstruction *obstructions, int contourVar, float contMin, float contMax, int viewMode, int xDim, int yDim, float uMax, int xDimVisible, int yDimVisible)//pitch in elements
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
	int im = Im[j];//ImageFcn(x, y, obstructions); // 
	if (isInsideObstruction(x, y, obstructions)) im = 1;
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


	float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
	float u = f1 - f3 + f5 - f6 - f7 + f8;
	float v = f2 - f4 + f5 + f6 - f7 - f8;
	float usqr = u*u+v*v;
	float StrainRate;

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
	else{
		boundaries(f0, f1, f2, f3, f4, f5, f6, f7, f8, y, im, xDim, yDim, uMax);

		mrt_collide(f0, f1, f2, f3, f4, f5, f6, f7, f8, omega, StrainRate);

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

	//Prepare data for visualization

	//need to change x,y,z coordinates to NDC (-1 to 1)
	float xcoord, ycoord, zcoord;
	int index;
	//int xdim = blockDim.x*gridDim.x;
	//int ydim = blockDim.y*gridDim.y;
	//xcoord = threadIdx.x + blockDim.x*blockIdx.x;
	//ycoord = threadIdx.y + blockDim.y*blockIdx.y;
	index = j;// x + y*blockDim.x*gridDim.x;
	////	x /= (float)(blockDim.x*gridDim.x)*0.5f;
	////	y /= (float)(blockDim.x*gridDim.x)*0.5f;//(float)(blockDim.y*gridDim.y);
	//xcoord /= xDim / 2;
	//ycoord /= yDim / 2;//(float)(blockDim.y*gridDim.y);
	//xcoord -= 1.0;// xdim / maxDim;
	//ycoord -= 1.0;// ydim / maxDim;

	ChangeCoordinatesToScaledFloat(xcoord, ycoord, xDimVisible, yDimVisible);

	if (im == 1) rho = 1.0;
	zcoord =  (rho - 1.0f) - 0.5f;// *15.f;//f1-f3+f5-f6-f7+f8;//rho;//(rho-1.0f)*2.f;
	//zcoord = -0.5f;// 0.05f*sinf(0.1f*(x)) + 0.05f*sinf(0.1f*y);// (rho - 1.0f) - 0.5f;// *15.f;//f1-f3+f5-f6-f7+f8;//rho;//(rho-1.0f)*2.f;

	//Color c = Color::FromArgb(1);
	//pos[threadIdx.x+threadIdx.y*blockDim.x] = make_float4(x,y,z,1.0f);

	//for color, need to convert 4 bytes (RGBA) to float
	float color;
	float variableValue = 0.f;
	float maxValue;
	float minValue;

	minValue = contMin;
	maxValue = contMax;

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
	unsigned char R = dmin(255.f,dmax(255 * ((variableValue - minValue) / (maxValue - minValue))));
	unsigned char G = dmin(255.f,dmax(255 * ((variableValue - minValue) / (maxValue - minValue))));
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
		R = 50; G = 120; B = 255;
		A = 155;
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
	if (im == 1){
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


	//if (x > 100 && x < 110 && y > 50 && y < 52)
	//{
	//	R = 255; G = 0; B = 0;
	//}

	
	//char b[] = {(char)R, (char)G, (char)B, (char)A};
	//char b[] = { R*cosTheta, G*cosTheta, B*cosTheta, A };
	char b[] = { R, G, B, A };
	//char b[] = {'100','1','1','100'};
	std::memcpy(&color, &b, sizeof(color));

	//vbo aray to be displayed
	pos[index] = make_float4(xcoord, ycoord, zcoord, color);
	//vel[index] = make_float4(xcoord, ycoord, u, 1.0f);

}

__global__ void CleanUpVBO(float4* pos, int xDimVisible, int yDimVisible)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
	if (x >= xDimVisible || y >= yDimVisible)
	{
		unsigned char b[] = { 0,0,0,255 };
		float color;
		std::memcpy(&color, &b, sizeof(color));
		pos[j] = make_float4(pos[j].x, pos[j].y, -1.f, color);
	}
}

__global__ void Lighting(float4* pos, Obstruction *obstructions, int xDimVisible, int yDimVisible, float3 cameraPosition)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
	unsigned char color[4];
	std::memcpy(color, &(pos[j].w), sizeof(color));
	float R, G, B, A;
	R = color[0];
	G = color[1];
	B = color[2];
	A = color[3];

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
		slope_x = (pos[(x + 1) + y*MAX_XDIM].z - pos[(x - 1) + y*MAX_XDIM].z) / (2.f*cellSize);
		slope_y = (pos[(x)+(y + 1)*MAX_XDIM].z - pos[(x)+(y - 1)*MAX_XDIM].z) / (2.f*cellSize);
		n.x = -slope_x*2.f*cellSize*2.f*cellSize;
		n.y = -slope_y*2.f*cellSize*2.f*cellSize;
		n.z = 2.f*cellSize*2.f*cellSize;
	}
	Normalize(n);
	float3 elementPosition = {pos[j].x,pos[j].y,pos[j].z };
	float3 diffuseLightDirection1 = {0.577367, 0.577367, -0.577367 };
	float3 diffuseLightDirection2 = { -0.577367, 0.577367, -0.577367 };
	//float3 cameraPosition = { -1.5, -1.5, 1.5};
	float3 eyeDirection = elementPosition - cameraPosition;
	float3 diffuseLightColor1 = {0.5f, 0.5f, 0.5f};
	float3 diffuseLightColor2 = {0.5f, 0.5f, 0.5f};
	float3 specularLightColor1 = {0.9f, 0.9f, 0.9f};

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
	
	float3 diffuse1  = 0.f*cosTheta1*diffuseLightColor1;
	float3 diffuse2  = 0.f*cosTheta2*diffuseLightColor2;
	float3 specular1 = cosAlpha*specularLightColor1;

	color[0] = color[0]*dmin(1.f,(diffuse1.x+diffuse2.x+specular1.x+lightAmbient));
	color[1] = color[1]*dmin(1.f,(diffuse1.y+diffuse2.y+specular1.y+lightAmbient));
	color[2] = color[2]*dmin(1.f,(diffuse1.z+diffuse2.z+specular1.z+lightAmbient));
	color[3] = A;

	std::memcpy(&(pos[j].w), color, sizeof(color));
}



__global__ void initialize_Floor(float4* pos, float* floor_d, int xDim, int yDim, int xDimVisible, int yDimVisible) //obstruction* obstruction)//pitch in elements
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int j = MAX_XDIM*MAX_YDIM + x + y*MAX_XDIM;//index on padded mem (pitch in elements)

	float xcoord, ycoord, zcoord;
	ChangeCoordinatesToScaledFloat(xcoord, ycoord, xDimVisible, yDimVisible);
	zcoord = -1.f;
	unsigned char R(255), G(255), B(255), A(255);

	float3 n = { 0, 0, 1 };
	float slope_x = 0.f;
	float slope_y = 0.f;
	float cellSize = 2.f / xDimVisible;
	if (x == 0)
	{
	}
	else if (y == 0)
	{
	}
	else if (x >= xDimVisible - 1)
	{
	}
	else if (y >= yDimVisible - 1)
	{
	}
	else if (x > 0 && x < (xDimVisible - 1) && y > 0 && y < (yDimVisible - 1))
	{
		slope_x = (pos[(x + 1) + y*MAX_XDIM].z - pos[(x - 1) + y*MAX_XDIM].z) / (2.f*cellSize);
		slope_y = (pos[(x)+(y + 1)*MAX_XDIM].z - pos[(x)+(y - 1)*MAX_XDIM].z) / (2.f*cellSize);
		n.x = -slope_x*2.f*cellSize*2.f*cellSize;
		n.y = -slope_y*2.f*cellSize*2.f*cellSize;
		n.z = 2.f*cellSize*2.f*cellSize;
	}
	Normalize(n);
	float theta1 = acosf(n.z / sqrt(DotProduct(n, n)));
	float theta2 = asinf(1.0 / 1.33f)*sin(theta1);
	float dx = sin(theta1 - theta2)*(pos[(x)+(y)*MAX_XDIM].z + 1.f)*(-n.x);
	float dy = sin(theta1 - theta2)*(pos[(x)+(y)*MAX_XDIM].z + 1.f)*(-n.y);

	float attenuation = 0.1f;// (pos[(x)+(y)*MAX_XDIM].z + 1.f)*0.5f;



	R = 255.f*cos(theta1)*attenuation;
	G = 255.f*cos(theta1)*attenuation;
	B = 255.f*cos(theta1)*attenuation;

	char b[] = { R, G, B, A };
	float color;
	std::memcpy(&color, &b, sizeof(color));
	int newX = x;// +dx;
	int newY = y;// + dy;
	if (newX > 0 && newX < xDimVisible && newY > 0 && newY < yDimVisible && false)
	{
		pos[newX + newY*MAX_XDIM] = make_float4(xcoord, ycoord, zcoord, color);
	}
	else{
		pos[j] = make_float4(xcoord, ycoord, zcoord, color);
	}
}

__global__ void update_Floor(float4* pos, float* floor_d, int xDim, int yDim, int xDimVisible, int yDimVisible) //obstruction* obstruction)//pitch in elements
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int j = MAX_XDIM*MAX_YDIM + x + y*MAX_XDIM;//index on padded mem (pitch in elements)

	float xcoord, ycoord, zcoord;
	ChangeCoordinatesToScaledFloat(xcoord, ycoord, xDimVisible, yDimVisible);
	zcoord = -1.f;
	unsigned char R(255), G(255), B(255), A(255);

	float3 n = { 0, 0, 1 };
	float slope_x = 0.f;
	float slope_y = 0.f;
	float cellSize = 2.f / xDimVisible;
	if (x > 0 && x < (xDimVisible - 1) && y > 0 && y < (yDimVisible - 1))
	{
		slope_x = (pos[(x + 1) + y*MAX_XDIM].z - pos[(x - 1) + y*MAX_XDIM].z) / (2.f*cellSize);
		slope_y = (pos[(x)+(y + 1)*MAX_XDIM].z - pos[(x)+(y - 1)*MAX_XDIM].z) / (2.f*cellSize);
		n.x = -slope_x*2.f*cellSize*2.f*cellSize;
		n.y = -slope_y*2.f*cellSize*2.f*cellSize;
		n.z = 2.f*cellSize*2.f*cellSize;
	}
	Normalize(n);
	float theta1 = acosf(n.z / sqrt(DotProduct(n, n)));
	float theta2 = asinf(1.0 / 1.3f*sin(theta1));
	float waterDepth = 80.f;
	float dx = sin(theta1 - theta2)*(pos[(x)+(y)*MAX_XDIM].z + 0.5f)*waterDepth*(-n.x)/sqrt(n.x*n.x + n.y*n.y);
	float dy = sin(theta1 - theta2)*(pos[(x)+(y)*MAX_XDIM].z + 0.5f)*waterDepth*(-n.y)/sqrt(n.x*n.x+n.y*n.y);

	float attenuation = 0.5f;// -0.5f*sin(theta1 - theta2)*(pos[(x)+(y)*MAX_XDIM].z + 0.5f) + 1.f;

	R = 255.f*cos(theta1)*attenuation;
	G = 255.f*cos(theta1)*attenuation;
	B = 255.f*cos(theta1)*attenuation;

	char b[] = { R, G, B, A };
	float color;
	std::memcpy(&color, &b, sizeof(color));
	int newX = x +dx + 0.5f;
	int newY = y + dy+0.5f;
	if (newX > 0 && newX < xDimVisible && newY > 0 && newY < yDimVisible)
	{
		atomicAdd(&floor_d[newX + newY*MAX_XDIM], attenuation);
	}
}

__device__ float2 ComputePositionOfLightOnFloor(float4* pos, float3 incidentLight, int x, int y,  int xDimVisible, int yDimVisible)
{
	int j = MAX_XDIM*MAX_YDIM + x + y*MAX_XDIM;//index on padded mem (pitch in elements)

	unsigned char R(255), G(255), B(255), A(255);

	float3 n = { 0, 0, 1 };
	float slope_x = 0.f;
	float slope_y = 0.f;
	float cellSize = 2.f / xDimVisible;
	if (x > 0 && x < (xDimVisible - 1) && y > 0 && y < (yDimVisible - 1))
	{
		slope_x = (pos[(x + 1) + y*MAX_XDIM].z - pos[(x - 1) + y*MAX_XDIM].z) / (2.f*cellSize);
		slope_y = (pos[(x)+(y + 1)*MAX_XDIM].z - pos[(x)+(y - 1)*MAX_XDIM].z) / (2.f*cellSize);
		n.x = -slope_x*2.f*cellSize*2.f*cellSize;
		n.y = -slope_y*2.f*cellSize*2.f*cellSize;
		n.z = 2.f*cellSize*2.f*cellSize;
	}
	Normalize(n);

	//float2 incidentLightCP, refractedLightCP, normalCP;
	Normalize(incidentLight);
	//incidentLightCP.x = sqrt(incidentLight.x*incidentLight.x + incidentLight.y*incidentLight.y);
	//incidentLightCP.y = incidentLight.z;
	//float incidentLightCP_mag = sqrt(incidentLightCP.x*incidentLightCP.x + incidentLightCP.y*incidentLightCP.y);
	//incidentLightCP.x /= incidentLightCP_mag;
	//incidentLightCP.y /= incidentLightCP_mag;

	//float incidentLightAngleXY = atan2f(incidentLight.y, incidentLight.x);
	//normalCP.x = sqrt(n.x*n.x + n.y*n.y);
	//normalCP.y = n.z;

	//float theta1 = acosf(DotProduct(n,incidentLight));  //incident light is pointing into surface. n is pointing out of surface
	//float theta1 = acosf(n.z / sqrt(DotProduct(n, n)));
	//float theta2 = asinf(1.0 / 1.3f*sin(theta1));
	float waterDepth = 80.f;

	//float rotAngle = theta1 - theta2;
	//if (normalCP.x < incidentLightCP.x) rotAngle = -rotAngle;
	//refractedLightCP.x = incidentLightCP.x*cosf(theta1 - theta2) - incidentLightCP.y*sinf(theta1-theta2);
	//refractedLightCP.y = incidentLightCP.x*cosf(theta1 - theta2) + incidentLightCP.y*sinf(theta1-theta2);

	float3 refractedLight;
	//refractedLight.x = refractedLightCP.x*cosf(incidentLightAngleXY);
	//refractedLight.y = refractedLightCP.x*sinf(incidentLightAngleXY);
	//refractedLight.z = refractedLightCP.y;

	//float gamma = asinf(-incidentLight.z);
	//float alpha = gamma + theta1 - theta2;
	//float deltaFloor;
	//deltaFloor = (pos[(x)+(y)*MAX_XDIM].z + 0.5f)*waterDepth / tanf(alpha);
	//deltaFloor = (pos[(x)+(y)*MAX_XDIM].z + 0.5f)*waterDepth*sinf(PI*0.25f - alpha) / dmax(0.01f, cosf(PI*0.25f - alpha));

	//float dx = deltaFloor*(refractedLight.x) / sqrt(refractedLight.x*refractedLight.x + refractedLight.y*refractedLight.y);
	//float dy = deltaFloor*(refractedLight.y) / sqrt(refractedLight.x*refractedLight.x + refractedLight.y*refractedLight.y);

	float r = 1.0 / 1.3f;
	float c = -(DotProduct(n, incidentLight));
	refractedLight = r*incidentLight + (r*c - sqrt(1.f - r*r*(1.f - c*c)))*n;
	//if (x == 100 && y == 10) printf("light intensity: %ff\n",refractedLight.z);
	//refractedLight = r*(CrossProduct(n,(CrossProduct(-1.f*n,incidentLight))))-1.f*sqrt(1.f-r*r*DotProduct(CrossProduct(n,incidentLight),CrossProduct(n,incidentLight)))*n;


	float dx = -refractedLight.x*(pos[(x)+(y)*MAX_XDIM].z + 1.f)*waterDepth / refractedLight.z;
	float dy = -refractedLight.y*(pos[(x)+(y)*MAX_XDIM].z + 1.f)*waterDepth / refractedLight.z;

	//if (x > 100 && x < 110 && y > 50 && y < 52) printf("%i, %i, %f, %f\n",x,y,dx,dy);

//	float dx = sin(theta1 - theta2)*(pos[(x)+(y)*MAX_XDIM].z + 0.5f)*waterDepth*(-n.x)/sqrt(n.x*n.x + n.y*n.y);
//	float dy = sin(theta1 - theta2)*(pos[(x)+(y)*MAX_XDIM].z + 0.5f)*waterDepth*(-n.y)/sqrt(n.x*n.x+n.y*n.y);

	return float2{ (float)x + dx, (float)y + dy };
}

__device__ float ComputeAreaFrom4Points(float2 nw, float2 ne, float2 sw, float2 se)
{
	float2 vecN = ne - nw;
	float2 vecS = se - sw;
	float2 vecE = ne - se;
	float2 vecW = nw - sw;
	return CrossProductArea(vecN, vecW) + CrossProductArea(vecE, vecS);
}

__global__ void update_LightMesh(float4* pos, float2* lightMesh_d, float3 incidentLight, Obstruction* obstructions, int xDim, int yDim, int xDimVisible, int yDimVisible) //obstruction* obstruction)//pitch in elements
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)
	
	if (x < xDimVisible && y < yDimVisible)
	{

	//float3 incidentLight{ 0.f, -0.f, -1.f };
		float2 lightPositionOnFloor;
		if (isInsideObstruction(x, y, obstructions,1.f))
		{
			lightPositionOnFloor = make_float2(x, y);
		}
		else
		{
			lightPositionOnFloor = ComputePositionOfLightOnFloor(pos, incidentLight, x, y, xDimVisible, yDimVisible);
		}

	lightMesh_d[j] = lightPositionOnFloor;
	}
}

__global__ void LightFloorUsingLightMesh(float* floor_d, float2* lightMesh_d, Obstruction* obstructions, int xDim, int yDim, int xDimVisible, int yDimVisible) //obstruction* obstruction)//pitch in elements
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	if (x < xDimVisible-2 && y < yDimVisible-2)
	{
		float2 nw, ne, sw, se;
		nw = lightMesh_d[(x)+(y+1)*MAX_XDIM];
		ne = lightMesh_d[(x+1)+(y+1)*MAX_XDIM];
		sw = lightMesh_d[(x)+(y)*MAX_XDIM];
		se = lightMesh_d[(x+1)+(y)*MAX_XDIM];
		float areaOfLightMeshOnFloor = ComputeAreaFrom4Points(nw, ne, sw, se);
		float lightIntensity = 0.3f / areaOfLightMeshOnFloor;
		//if (x > 100 && x < 110 && y > 50 && y < 52) lightIntensity = 0.1f;
		//if (areaOfLightMeshOnFloor > 1.f) printf("%f, %f, %f\n",sw.x,sw.y,areaOfLightMeshOnFloor);
		//if (x == 100 && y == 10) printf("light intensity: %ff\n",lightIntensity);

//		nw.x -= 5.f;
//		nw.y += 5.f;
//		sw.x -= 5.f;
//		sw.y -= 5.f;
//		ne.x += 5.f;
//		ne.y += 5.f;
//		se.x += 5.f;
//		se.y -= 5.f;

//		if (x == 100)
//		{
//			lightIntensity = 0.1f;
//		}
//		else
//		{
//			lightIntensity = 0.f;
//		}


//		for (int i = dmax(1.f,dmin(nw.x, ne.x, sw.x, se.x)); i < dmin(xDimVisible-2.f,dmax(nw.x, ne.x, sw.x, se.x)+1.f); i++)
//		{
//			for (int j = dmax(1.f,dmin(nw.y, ne.y, sw.y, se.y)); j < dmin(yDimVisible-2.f,dmax(nw.y, ne.y, sw.y, se.y)+1.f); j++)
//			{
//				if (i >= 0 && i < xDimVisible && j >= 0 && j < yDimVisible)
//				{
//					float2 p = make_float2(i, j);
//					if (IsPointInsideTriangle(p,nw,ne,sw) || IsPointInsideTriangle(p,ne,se,sw))
//					{
//						atomicAdd(&floor_d[i + j*MAX_XDIM], lightIntensity);
//					}
//				}
//			}
//		}
		//floor_d[x + y*MAX_XDIM] = lightIntensity;
		atomicAdd(&floor_d[x + (y)*MAX_XDIM], lightIntensity*0.25f);
		atomicAdd(&floor_d[x+1 + (y)*MAX_XDIM], lightIntensity*0.25f);
		atomicAdd(&floor_d[x+1 + (y+1)*MAX_XDIM], lightIntensity*0.25f);
		atomicAdd(&floor_d[x + (y+1)*MAX_XDIM], lightIntensity*0.25f);
	}
}

__global__ void light_Filter(float* floor_d, float* floorFiltered_d, int xDim, int yDim, int xDimVisible, int yDimVisible) //obstruction* obstruction)//pitch in elements
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int j = x + y*MAX_XDIM;//index on padded mem (pitch in elements)

	if (x> 1 && y>1 && x < xDimVisible-1 && y<yDimVisible-1)
	{ 
		float light = floor_d[x + y*MAX_XDIM];
		//float light = floor_d[x + y*MAX_XDIM] +floor_d[x + 1 + (y)*MAX_XDIM] + floor_d[x - 1 + (y)*MAX_XDIM] +
		//	floor_d[x + (y + 1)*MAX_XDIM] + floor_d[x + 1 + (y + 1)*MAX_XDIM] + floor_d[x - 1 + (y + 1)*MAX_XDIM] +
		//	floor_d[x + (y - 1)*MAX_XDIM] + floor_d[x + 1 + (y - 1)*MAX_XDIM] + floor_d[x - 1 + (y - 1)*MAX_XDIM];
		floorFiltered_d[j] = light / 9.f;

		//if (x == 100 && y == 10) printf("%f\n",light);
	}
	

}

__global__ void light_Floor(float4* pos, float* floor_d, float* floorFiltered_d, float2* lightMesh_d, Obstruction* obstructions, int xDim, int yDim, int xDimVisible, int yDimVisible) //obstruction* obstruction)//pitch in elements
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int j = MAX_XDIM*MAX_YDIM + x + y*MAX_XDIM;//index on padded mem (pitch in elements)
	float xcoord, ycoord, zcoord;

	xcoord = lightMesh_d[x + y*MAX_XDIM].x;
	ycoord = lightMesh_d[x + y*MAX_XDIM].y;
	zcoord = -1.f;

	ChangeCoordinatesToScaledFloat(xcoord, ycoord, xDimVisible, yDimVisible);
	float lightFactor = dmin(1.f,floor_d[x + y*MAX_XDIM]);
	//lightFactor = cosf(x*0.1f);// dmin(1.f, floor_d[x + y*MAX_XDIM]);
	//float lightFactor = floor_d[x + y*MAX_XDIM];
	//if (x == 0 && y == 5) printf("%f\n",floor_d[x + y*MAX_XDIM]);
	floor_d[x + y*MAX_XDIM] = 0.f;

	unsigned char R = 50.f*lightFactor;
	unsigned char G = 120.f*lightFactor;
	unsigned char B = 255.f*lightFactor;
	unsigned char A = 255.f;

	if (isInsideObstruction(x, y, obstructions, 1.f))
	{
		if (isInsideObstruction(x, y, obstructions))
		{
			zcoord = -0.3f;
			lightFactor = 0.8f;
			R = 255.f;
			G = 255.f;
			B = 255.f;
		}
	}

	R *= lightFactor;
	G *= lightFactor;
	B *= lightFactor;

	char b[] = { R, G, B, A };
	float color;
	std::memcpy(&color, &b, sizeof(color));
	pos[j] = make_float4(xcoord, ycoord, zcoord, color);
}
/*----------------------------------------------------------------------------------------
 * End of device functions
 */

__global__ void refraction_Floor(float4* pos, float* floor_d, float* floorFiltered_d, float2* lightMesh_d, Obstruction* obstructions, int xDim, int yDim, int xDimVisible, int yDimVisible) //obstruction* obstruction)//pitch in elements
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;//coord in linear mem
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int j = MAX_XDIM*MAX_YDIM + x + y*MAX_XDIM;//index on padded mem (pitch in elements)
	float xcoord, ycoord, zcoord;

	xcoord = lightMesh_d[x + y*MAX_XDIM].x;
	ycoord = lightMesh_d[x + y*MAX_XDIM].y;

	float2 coordOfFloor = ComputePositionOfLightOnFloor(pos, make_float3(0, 0, -1), x, y, xDimVisible, yDimVisible);
//	float floorX = coordOfFloor.x;
//	float floorY = coordOfFloor.y;

//	float color = floor_d[coordOfFloor.x + coordOfFloor.y*MAX_XDIM];

	ChangeCoordinatesToScaledFloat(xcoord, ycoord, xDimVisible, yDimVisible);
	float lightFactor = cosf(x*0.05f);// dmin(1.f, floor_d[x + y*MAX_XDIM]);
	//float lightFactor = floor_d[x + y*MAX_XDIM];
	//if (x == 0 && y == 5) printf("%f\n",floor_d[x + y*MAX_XDIM]);
	floor_d[x + y*MAX_XDIM] = 0.5f;

	if (isInsideObstruction(x, y, obstructions))
	{
		zcoord = 0.1f;
		lightFactor = 1.f;
	}
	else
	{
		zcoord = -1.f;
	}

	unsigned char R = 50.f*lightFactor;
	unsigned char G = 120.f*lightFactor;
	unsigned char B = 255.f*lightFactor;
	unsigned char A = 255.f;

	char b[] = { R, G, B, A };
	//std::memcpy(&color, &b, sizeof(color));
	//pos[j] = make_float4(xcoord, ycoord, zcoord, color);
}


/*----------------------------------------------------------------------------------------
 * End of device functions
 */
void InitializeDomain(float4* vis, float* f_d, int* im_d, int xDim, int yDim, float uMax, int xDimVisible, int yDimVisible)
{
	dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
	//dim3 grid(g_xDim / BLOCKSIZEX, g_yDim / BLOCKSIZEY);
	dim3 grid(ceil(static_cast<float>(g_xDim) / BLOCKSIZEX), g_yDim / BLOCKSIZEY);
	initialize_single << <grid, threads >> >(vis, f_d, im_d, xDim, yDim, uMax, xDimVisible, yDimVisible);
}

void MarchSolution(float4* vis, float* fA_d, float* fB_d, int* im_d, Obstruction* obst_d,
	ContourVariable contVar, float contMin, float contMax, ViewMode viewMode, int xDim, int yDim, float uMax, float omega, int tStep, int xDimVisible, int yDimVisible)
{
	dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
	dim3 grid(ceil(static_cast<float>(g_xDim) / BLOCKSIZEX), g_yDim / BLOCKSIZEY);
	//dim3 grid(g_xDim / BLOCKSIZEX, g_yDim / BLOCKSIZEY);
	for (int i = 0; i < tStep; i++)
	{
		mrt_d_single << <grid, threads >> >(vis, fA_d, fB_d, omega, im_d, obst_d, contVar, contMin, viewMode, contMax, xDim, yDim, uMax, xDimVisible, yDimVisible);
		if (g_paused == 0)
		{
			mrt_d_single << <grid, threads >> >(vis, fB_d, fA_d, omega, im_d, obst_d, contVar, contMin, contMax, viewMode, xDim, yDim, uMax, xDimVisible, yDimVisible);
		}
	}
}

void UpdateDeviceObstructions(Obstruction* obst_d, int targetObstID, Obstruction newObst)
{
	UpdateObstructions << <1, 1 >> >(obst_d,targetObstID,newObst.r1,newObst.x,newObst.y,newObst.shape);
}

void CleanUpDeviceVBO(float4* vis, int xDimVisible, int yDimVisible)
{
	dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
	dim3 grid(MAX_XDIM / BLOCKSIZEX, MAX_YDIM / BLOCKSIZEY);
	CleanUpVBO << <grid, threads>> >(vis, xDimVisible, yDimVisible);
}

void DeviceLighting(float4* vis, Obstruction* obst_d, int xDimVisible, int yDimVisible, float3 cameraPosition)
{
	dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
	dim3 grid(ceil(static_cast<float>(g_xDim) / BLOCKSIZEX), g_yDim / BLOCKSIZEY);
	Lighting << <grid, threads>> >(vis, obst_d, xDimVisible, yDimVisible, cameraPosition);
}

void InitializeFloor(float4* vis, float* floor_d, int xDim, int yDim, int xDimVisible, int yDimVisible)
{
	dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
	dim3 grid(ceil(static_cast<float>(g_xDim) / BLOCKSIZEX), g_yDim / BLOCKSIZEY);
	initialize_Floor << <grid, threads >> >(vis, floor_d, xDim, yDim, xDimVisible, yDimVisible);
}

void UpdateFloor(float4* vis, float* floor_d, int xDim, int yDim, int xDimVisible, int yDimVisible)
{
	dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
	dim3 grid(ceil(static_cast<float>(g_xDim) / BLOCKSIZEX), g_yDim / BLOCKSIZEY);
	update_Floor << <grid, threads >> >(vis, floor_d, xDim, yDim, xDimVisible, yDimVisible);
}

void LightFloor(float4* vis, float2* lightMesh_d, float* floor_d, float* floorFiltered_d, Obstruction* obst_d, int xDim, int yDim, int xDimVisible, int yDimVisible)
{
	dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
	dim3 grid(ceil(static_cast<float>(g_xDim) / BLOCKSIZEX), g_yDim / BLOCKSIZEY);
	float3 incidentLight4 = { 0, -20.f, -1.f };
	float3 incidentLight2 = { -0.5f, -0.5f, -1.f };
	float3 incidentLight3 = { -0.5f, 0.5f, -1.f };
	float3 incidentLight1 = { -0.25f, -0.25f, -1.f };
	update_LightMesh << <grid, threads >> >(vis, lightMesh_d, incidentLight1, obst_d, xDim, yDim, xDimVisible, yDimVisible);
	LightFloorUsingLightMesh << <grid, threads >> >(floor_d, lightMesh_d, obst_d, xDim, yDim, xDimVisible, yDimVisible);
	//update_LightMesh << <grid, threads >> >(vis, lightMesh_d, incidentLight2, xDim, yDim, xDimVisible, yDimVisible);
	//LightFloorUsingLightMesh << <grid, threads >> >(floor_d, lightMesh_d, xDim, yDim, xDimVisible, yDimVisible);
	//update_LightMesh << <grid, threads >> >(vis, lightMesh_d, incidentLight3, xDim, yDim, xDimVisible, yDimVisible);
	//LightFloorUsingLightMesh << <grid, threads >> >(floor_d, lightMesh_d, xDim, yDim, xDimVisible, yDimVisible);
	//update_LightMesh << <grid, threads >> >(vis, lightMesh_d, incidentLight4, xDim, yDim, xDimVisible, yDimVisible);
	//LightFloorUsingLightMesh << <grid, threads >> >(floor_d, lightMesh_d, xDim, yDim, xDimVisible, yDimVisible);

	light_Floor << <grid, threads >> >(vis, floor_d, floorFiltered_d, lightMesh_d, obst_d, xDim, yDim, xDimVisible, yDimVisible);


	//light_Filter << <grid, threads >> >(floor_d, floorFiltered_d, xDim, yDim, xDimVisible, yDimVisible);
	//update_LightMesh << <grid, threads >> >(vis, lightMesh_d, incidentLight2, xDim, yDim, xDimVisible, yDimVisible);
	//LightFloorUsingLightMesh << <grid, threads >> >(floor_d, lightMesh_d, xDim, yDim, xDimVisible, yDimVisible);
}

void Refraction(float4* vis, float* floor_d, float* floorFiltered_d, float2* lightMesh_d, int xDim, int yDim, int xDimVisible, int yDimVisible)
{
	dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
	dim3 grid(ceil(static_cast<float>(g_xDim) / BLOCKSIZEX), g_yDim / BLOCKSIZEY);
	//refraction_Floor << <grid, threads >> >(vis, floor_d, floorFiltered_d, lightMesh_d, obst_d, xDim, yDim, xDimVisible, yDimVisible);
}

int runCUDA()
{
    return 0;
}
