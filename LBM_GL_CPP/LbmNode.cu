#include "LbmNode.h"

__host__ __device__ LbmNode::LbmNode()
{
    for (int i = 0; i < 9; i++)
    {
        m_f[9] = 0.f;
    }
    m_xDim = MAX_XDIM;
    m_yDim = MAX_YDIM;
}

__device__ int LbmNode::GetXDim()
{
    return m_xDim;
}

__device__ int LbmNode::GetYDim()
{
    return m_yDim;
}

__device__ void LbmNode::SetXDim(int xDim)
{
    m_xDim = xDim;
}

__device__ void LbmNode::SetYDim(int yDim)
{
    m_yDim = yDim;
}

__device__ float LbmNode::ComputeRho()
{
    return m_f[0] + m_f[1] + m_f[2] + m_f[3] + m_f[4] + m_f[5] + m_f[6] + m_f[7] + m_f[8];
}

__device__ float LbmNode::ComputeU()
{
    return m_f[1] - m_f[3] + m_f[5] - m_f[6] - m_f[7] + m_f[8];
}

__device__ float LbmNode::ComputeV()
{
    return m_f[2] - m_f[4] + m_f[5] + m_f[6] - m_f[7] - m_f[8];
}

__device__ void LbmNode::ReadIncomingDistributions(float* f, const int x, const int y)
{
    int j = x + y*MAX_XDIM;
    int xDim = GetXDim();
    int yDim = GetYDim();
    m_f[0] = f[j];
    m_f[1] = f[f_mem(1, dmax(x - 1), y)];
    m_f[3] = f[f_mem(3, dmin(x + 1, xDim), y)];
    m_f[2] = f[f_mem(2, x, y - 1)];
    m_f[5] = f[f_mem(5, dmax(x - 1), y - 1)];
    m_f[6] = f[f_mem(6, dmin(x + 1, xDim), y - 1)];
    m_f[4] = f[f_mem(4, x, y + 1)];
    m_f[7] = f[f_mem(7, dmin(x + 1, xDim), y + 1)];
    m_f[8] = f[f_mem(8, dmax(x - 1), dmin(y + 1, yDim))];
}

__device__ void LbmNode::ReadDistributions(float* f, const int x, const int y)
{
    for (int i = 0; i < 9; i++)
    {
        m_f[i] = f[f_mem(i, x, y)];
    }
}

__device__ void LbmNode::Initialize(float* f, const float rho,
    const float u, const float v)
{
    float fEq[9];
    ComputeFeqs(fEq, rho, u, v);
    for (int i = 0; i < 9; i++)
    {
        m_f[i] = fEq[i];
    }
}

__device__ void LbmNode::WriteDistributions(float* f, const int x, const int y)
{
    for (int i = 0; i < 9; i++)
    {
        f[f_mem(i, x, y)] = m_f[i];
    }
}

__device__ void LbmNode::ComputeFeqs(float* fOut, const float rho, const float u, const float v)
{
    float usqr = u*u + v*v;
    fOut[0] = 0.4444444444f*(rho - 1.5f*usqr);
    fOut[1] = 0.1111111111f*(rho + 3.0f*u + 4.5f*u*u - 1.5f*usqr);
    fOut[2] = 0.1111111111f*(rho + 3.0f*v + 4.5f*v*v - 1.5f*usqr);
    fOut[3] = 0.1111111111f*(rho - 3.0f*u + 4.5f*u*u - 1.5f*usqr);
    fOut[4] = 0.1111111111f*(rho - 3.0f*v + 4.5f*v*v - 1.5f*usqr);
    fOut[5] = 0.02777777778*(rho + 3.0f*(u + v) + 4.5f*(u + v)*(u + v) - 1.5f*usqr);
    fOut[6] = 0.02777777778*(rho + 3.0f*(-u + v) + 4.5f*(-u + v)*(-u + v) - 1.5f*usqr);
    fOut[7] = 0.02777777778*(rho + 3.0f*(-u - v) + 4.5f*(-u - v)*(-u - v) - 1.5f*usqr);
    fOut[8] = 0.02777777778*(rho + 3.0f*(u - v) + 4.5f*(u - v)*(u - v) - 1.5f*usqr);   
}

__device__ void LbmNode::ComputeFeqs(float* fOut)
{
    float rho = ComputeRho();
    float u = ComputeU();
    float v = ComputeV();

    ComputeFeqs(fOut, rho, u, v);
}

__device__ float LbmNode::ComputeStrainRateMagnitude()
{
    float fEq[9];
    ComputeFeqs(fEq);
    float qxx = (m_f[1]-fEq[1]) + (m_f[3]-fEq[3]) + (m_f[5]-fEq[5]) + (m_f[6]-fEq[6])
        + (m_f[7]-fEq[7]) + (m_f[8]-fEq[8]);
    float qxy = (m_f[5]-fEq[5]) - (m_f[6]-fEq[6]) + (m_f[7]-fEq[7]) - (m_f[8]-fEq[8]) ;
    float qyy = (m_f[5]-fEq[5]) + (m_f[2]-fEq[2]) + (m_f[6]-fEq[6]) + (m_f[7]-fEq[7])
        + (m_f[4]-fEq[4]) + (m_f[8]-fEq[8]);
    return sqrt(qxx*qxx + qxy*qxy * 2 + qyy*qyy);
}

__device__ void LbmNode::DirichletWest(const int y, const int xDim, const int yDim, const float uMax)
{
    if (y == 0){
        m_f[2] = m_f[4];
        m_f[6] = m_f[7];
    }
    else if (y == yDim - 1){
        m_f[4] = m_f[2];
        m_f[7] = m_f[6];
    }
    float u, v;
    u = uMax;
    v = 0.0f;
    m_f[1] = m_f[3] + u*0.66666667f;
    m_f[5] = m_f[7] - 0.5f*(m_f[2] - m_f[4]) + v*0.5f + u*0.166666667f;
    m_f[8] = m_f[6] + 0.5f*(m_f[2] - m_f[4]) - v*0.5f + u*0.166666667f;
}

__device__ void LbmNode::NeumannEast(const int y, const int xDim, const int yDim)
{
    if (y == 0){
        m_f[2] = m_f[4];
        m_f[5] = m_f[8];
    }
    else if (y == yDim - 1){
        m_f[4] = m_f[2];
        m_f[8] = m_f[5];
    }
    float rho, u, v;
    v = 0.0f;
    rho = 1.f;
    u = -rho + ((m_f[0] + m_f[2] + m_f[4]) + 2.0f*m_f[1] + 2.0f*m_f[5] + 2.0f*m_f[8]);
    m_f[3] = m_f[1] - u*0.66666667f;
    m_f[7] = m_f[5] + 0.5f*(m_f[2] - m_f[4]) - v*0.5f - u*0.166666667f;
    m_f[6] = m_f[8] - 0.5f*(m_f[2] - m_f[4]) + v*0.5f - u*0.166666667f;
}

__device__ void LbmNode::ApplyBCs(const int y, const int im, const int xDim,
    const int yDim, const float uMax)
{
    if (im == 2)//NeumannEast
    {
        NeumannEast(y, xDim, yDim);
    }
    else if (im == 3)//DirichletWest
    {
        DirichletWest(y, xDim, yDim, uMax);
    }
    else if (im == 11)//xsymmetry
    {
        m_f[4] = m_f[2];
        m_f[7] = m_f[6];
        m_f[8] = m_f[5];
    }
    else if (im == 12)//xsymmetry
    {
        m_f[2] = m_f[4];
        m_f[6] = m_f[7];
        m_f[5] = m_f[8];
    }  
}

__device__ void LbmNode::MovingWall(const float rho, const float u, const float v)
{
    float fEq[9];
    ComputeFeqs(fEq, rho, u, v);
    for (int i = 0; i < 9; i++)
    {
        m_f[i] = fEq[i];
    }
}

__device__ void LbmNode::BounceBackWall()
{
    Swap(m_f[1], m_f[3]);
    Swap(m_f[2], m_f[4]);
    Swap(m_f[5], m_f[7]);
    Swap(m_f[6], m_f[8]);
}

__device__ void LbmNode::Collide(const float omega)
{
    float Q = ComputeStrainRateMagnitude();
    float tau0 = 1.f / omega;
    float tau = 0.5f*tau0 + 0.5f*sqrt(tau0*tau0 + 18.f*SMAG_CONST*sqrt(2.f)*Q);
    float omegaTurb = 1.f / tau;

    float m1, m2, m4, m6, m7, m8;

    float u = ComputeU();
    float v = ComputeV();

    m1 = -2.f*m_f[0] + m_f[1] + m_f[2] + m_f[3] + m_f[4] + 4.f*m_f[5] + 4.f*m_f[6] + 4.f*m_f[7]
        + 4.f*m_f[8] - 3.0f*(u*u + v*v);
    m2 = 3.f*m_f[0] - 3.f*m_f[1] - 3.f*m_f[2] - 3.f*m_f[3] - 3.f*m_f[4] + 3.0f*(u*u + v*v); //ep
    m4 = -m_f[1] + m_f[3] + 2.f*m_f[5] - 2.f*m_f[6] - 2.f*m_f[7] + 2.f*m_f[8];;//qx_eq
    m6 = -m_f[2] + m_f[4] + 2.f*m_f[5] + 2.f*m_f[6] - 2.f*m_f[7] - 2.f*m_f[8];;//qy_eq
    m7 = m_f[1] - m_f[2] + m_f[3] - m_f[4] - (u*u - v*v);//pxx_eq
    m8 = m_f[5] - m_f[6] + m_f[7] - m_f[8] - (u*v);//pxy_eq

    m_f[0] = m_f[0] - (-m1 + m2)*0.11111111f;
    m_f[1] = m_f[1] - (-m1*0.027777777f - 0.05555555556f*m2 - 0.16666666667f*m4 + m7*omegaTurb*0.25f);
    m_f[2] = m_f[2] - (-m1*0.027777777f - 0.05555555556f*m2 - 0.16666666667f*m6 - m7*omegaTurb*0.25f);
    m_f[3] = m_f[3] - (-m1*0.027777777f - 0.05555555556f*m2 + 0.16666666667f*m4 + m7*omegaTurb*0.25f);
    m_f[4] = m_f[4] - (-m1*0.027777777f - 0.05555555556f*m2 + 0.16666666667f*m6 - m7*omegaTurb*0.25f);
    m_f[5] = m_f[5] - (0.05555555556f*m1 + m2*0.027777777f + 0.08333333333f*m4 + 0.08333333333f*m6
        + m8*omegaTurb*0.25f);
    m_f[6] = m_f[6] - (0.05555555556f*m1 + m2*0.027777777f - 0.08333333333f*m4 + 0.08333333333f*m6
        - m8*omegaTurb*0.25f);
    m_f[7] = m_f[7] - (0.05555555556f*m1 + m2*0.027777777f - 0.08333333333f*m4 - 0.08333333333f*m6
        + m8*omegaTurb*0.25f);
    m_f[8] = m_f[8] - (0.05555555556f*m1 + m2*0.027777777f + 0.08333333333f*m4 - 0.08333333333f*m6
        - m8*omegaTurb*0.25f);
}

