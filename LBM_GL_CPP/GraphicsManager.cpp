#include "GraphicsManager.h"
#include "Mouse.h"
#include "kernel.h"


extern Obstruction* g_obst_d;
extern int g_xDim, g_yDim, g_xDimVisible, g_yDimVisible, g_tStep;
extern Obstruction::Shape g_currentShape;
extern float g_currentSize;
extern float g_initialScaleUp;
extern ViewMode g_viewMode;
extern GLint viewport[4];
extern GLdouble projectionMatrix[16];
extern GLdouble modelMatrix[16];
extern cudaGraphicsResource *g_cudaSolutionField;

GraphicsManager::GraphicsManager()
{

}

GraphicsManager::GraphicsManager(Panel* panel)
{
    m_parent = panel;
}

void GraphicsManager::GetSimCoordFromMouseCoord(int &xOut, int &yOut, Mouse mouse)
{
    float xf = intCoordToFloatCoord(mouse.m_x, mouse.m_winW);
    float yf = intCoordToFloatCoord(mouse.m_y, mouse.m_winH);
    RectFloat coordsInRelFloat = RectFloat(xf, yf, 1.f, 1.f) / m_parent->m_rectFloat_abs;
    //float graphicsToSimDomainScalingFactor = static_cast<float>(g_xDim) / m_parent->m_rectInt_abs.m_w;
    //float graphicsToSimDomainScalingFactor = 1.f/g_initialScaleUp;// static_cast<float>(g_xDim) / m_parent->m_rectInt_abs.m_w;
    float graphicsToSimDomainScalingFactorX = static_cast<float>(g_xDimVisible) / min(m_parent->m_rectInt_abs.m_w, MAX_XDIM*g_initialScaleUp);
    float graphicsToSimDomainScalingFactorY = static_cast<float>(g_yDimVisible) / min(m_parent->m_rectInt_abs.m_h, MAX_YDIM*g_initialScaleUp);
    xOut = floatCoordToIntCoord(coordsInRelFloat.m_x, m_parent->m_rectInt_abs.m_w)*graphicsToSimDomainScalingFactorX;
    yOut = floatCoordToIntCoord(coordsInRelFloat.m_y, m_parent->m_rectInt_abs.m_h)*graphicsToSimDomainScalingFactorY;
}

void GraphicsManager::GetSimCoordFromFloatCoord(int &xOut, int &yOut, float xf, float yf)
{
    RectFloat coordsInRelFloat = RectFloat(xf, yf, 1.f, 1.f) / m_parent->m_rectFloat_abs;
    //float graphicsToSimDomainScalingFactor = static_cast<float>(g_xDim) / m_parent->m_rectInt_abs.m_w;
    //float graphicsToSimDomainScalingFactor = 1.f/g_initialScaleUp;// static_cast<float>(g_xDim) / m_parent->m_rectInt_abs.m_w;
    float graphicsToSimDomainScalingFactorX = static_cast<float>(g_xDimVisible) / min(m_parent->m_rectInt_abs.m_w, MAX_XDIM*g_initialScaleUp);
    float graphicsToSimDomainScalingFactorY = static_cast<float>(g_yDimVisible) / min(m_parent->m_rectInt_abs.m_h, MAX_YDIM*g_initialScaleUp);
    xOut = floatCoordToIntCoord(coordsInRelFloat.m_x, m_parent->m_rectInt_abs.m_w)*graphicsToSimDomainScalingFactorX;
    yOut = floatCoordToIntCoord(coordsInRelFloat.m_y, m_parent->m_rectInt_abs.m_h)*graphicsToSimDomainScalingFactorY;
}

void GraphicsManager::GetMouseRay(float3 &rayOrigin, float3 &rayDir, int mouseX, int mouseY)
{
    double x, y, z;
    gluUnProject(mouseX, mouseY, 0.0f, modelMatrix, projectionMatrix, viewport, &x, &y, &z);
    //printf("Origin: %f, %f, %f\n", x, y, z);
    rayOrigin.x = x;
    rayOrigin.y = y;
    rayOrigin.z = z;
    gluUnProject(mouseX, mouseY, 1.0f, modelMatrix, projectionMatrix, viewport, &x, &y, &z);
    rayDir.x = x-rayOrigin.x;
    rayDir.y = y-rayOrigin.y;
    rayDir.z = z-rayOrigin.z;
    float mag = sqrt(rayDir.x*rayDir.x + rayDir.y*rayDir.y + rayDir.z*rayDir.z);
    rayDir.x /= mag;
    rayDir.y /= mag;
    rayDir.z /= mag;
}

int GraphicsManager::GetSimCoordFrom3DMouseClickOnObstruction(int &xOut, int &yOut, Mouse mouse)
{
    float3 rayOrigin, rayDir;
    GetMouseRay(rayOrigin, rayDir, mouse.GetX(), mouse.GetY());
    int returnVal = 0;

    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    cudaGraphicsMapResources(1, &g_cudaSolutionField, 0);
    size_t num_bytes,num_bytes2;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, g_cudaSolutionField);

    float3 selectedCoordF;
    if (RayCastMouseClick(selectedCoordF, dptr, rayOrigin, rayDir, g_obst_d) == 0)
    {
        m_currentZ = selectedCoordF.z;

        xOut = (selectedCoordF.x + 1.f)*0.5f*g_xDimVisible;
        yOut = (selectedCoordF.y + 1.f)*0.5f*g_xDimVisible;
    }
    else
    {
        returnVal = 1;
    }

    cudaGraphicsUnmapResources(1, &g_cudaSolutionField, 0);

    //GetSimCoordFromFloatCoord(xOut, yOut, selectedCoordF.x, selectedCoordF.y);
    //printf("Coords: %i, %i\n", xOut,yOut);

    return returnVal;
}

// get simulation coordinates from mouse ray casting on plane on m_currentZ
void GraphicsManager::GetSimCoordFrom2DMouseRay(int &xOut, int &yOut, Mouse mouse)
{
    float3 rayOrigin, rayDir;
    GetMouseRay(rayOrigin, rayDir, mouse.GetX(), mouse.GetY());

    float t = (m_currentZ - rayOrigin.z)/rayDir.z;
    float xf = rayOrigin.x + t*rayDir.x;
    float yf = rayOrigin.y + t*rayDir.y;

    xOut = (xf + 1.f)*0.5f*g_xDimVisible;
    yOut = (yf + 1.f)*0.5f*g_xDimVisible;
}

void GraphicsManager::GetSimCoordFrom2DMouseRay(int &xOut, int &yOut, int mouseX, int mouseY)
{
    float3 rayOrigin, rayDir;
    GetMouseRay(rayOrigin, rayDir, mouseX, mouseY);

    float t = (m_currentZ - rayOrigin.z)/rayDir.z;
    float xf = rayOrigin.x + t*rayDir.x;
    float yf = rayOrigin.y + t*rayDir.y;

    xOut = (xf + 1.f)*0.5f*g_xDimVisible;
    yOut = (yf + 1.f)*0.5f*g_xDimVisible;
}

void GraphicsManager::ClickDown(Mouse mouse)
{
    int mod = glutGetModifiers();
    if (g_viewMode == ViewMode::TWO_DIMENSIONAL)
    {
        if (mouse.m_rmb == 1)
        {
            AddObstruction(mouse);
        }
        else if (mouse.m_mmb == 1)
        {
            if (IsInClosestObstruction(mouse))
            {
                RemoveObstruction(mouse);
            }
        }
        else if (mouse.m_lmb == 1)
        {
            if (IsInClosestObstruction(mouse))
            {
                m_currentObstId = FindClosestObstructionId(mouse);
            }
            else
            {
                m_currentObstId = -1;
            }
        }
    }
    else
    {
        if (mouse.m_lmb == 1)
        {
            m_currentObstId = -1;
            int x{ 0 }, y{ 0 }, z{ 0 };
            if (GetSimCoordFrom3DMouseClickOnObstruction(x, y, mouse) == 0)
            {
                m_currentObstId = FindClosestObstructionId(x, y);
            }
        }
        else if (mouse.m_rmb == 1)
        {
            m_currentObstId = -1;
            int x{ 0 }, y{ 0 }, z{ 0 };
            m_currentZ = -0.5f;
            GetSimCoordFrom2DMouseRay(x, y, mouse);
            AddObstruction(x,y);
        }
        else if (mouse.m_mmb == 1 && mod != GLUT_ACTIVE_CTRL)
        {
            m_currentObstId = -1;
            int x{ 0 }, y{ 0 }, z{ 0 };
            if (GetSimCoordFrom3DMouseClickOnObstruction(x, y, mouse) == 0)
            {
                m_currentObstId = FindClosestObstructionId(x, y);
                RemoveObstruction(x,y);
            }
    
        }
    }
}

void GraphicsManager::Drag(int xi, int yi, float dxf, float dyf)
{
    MoveObstruction(xi,yi,dxf,dyf);
}

void GraphicsManager::AddObstruction(Mouse mouse)
{
    int xi, yi;
    GetSimCoordFromMouseCoord(xi, yi, mouse);
    Obstruction obst = { g_currentShape, xi, yi, g_currentSize, 0, 0, 0, Obstruction::NEW };
    int obstId = FindUnusedObstructionId();
    m_obstructions[obstId] = obst;
    UpdateDeviceObstructions(g_obst_d, obstId, obst);
}

void GraphicsManager::AddObstruction(int simX, int simY)
{
    Obstruction obst = { g_currentShape, simX, simY, g_currentSize, 0, 0, 0, Obstruction::NEW  };
    int obstId = FindUnusedObstructionId();
    m_obstructions[obstId] = obst;
    UpdateDeviceObstructions(g_obst_d, obstId, obst);
}

void GraphicsManager::RemoveObstruction(Mouse mouse)
{
    int obstId = FindClosestObstructionId(mouse);
    if (obstId < 0) return;
    //Obstruction obst = m_obstructions[obstId];// { g_currentShape, -100, -100, 0, 0, Obstruction::NEW };
    m_obstructions[obstId].state = Obstruction::REMOVED;
    UpdateDeviceObstructions(g_obst_d, obstId, m_obstructions[obstId]);
}

void GraphicsManager::RemoveObstruction(int simX, int simY)
{
    int obstId = FindObstructionPointIsInside(simX,simY,1.f);
    if (obstId >= 0)
    {
        m_obstructions[obstId].state = Obstruction::REMOVED;
        //m_obstructions[obstId] = obst;
        UpdateDeviceObstructions(g_obst_d, obstId, m_obstructions[obstId]);
    }
}

void GraphicsManager::MoveObstruction(int xi, int yi, float dxf, float dyf)
{
    if (m_currentObstId > -1)
    {
        if (g_viewMode == ViewMode::TWO_DIMENSIONAL)
        {
            Obstruction obst = m_obstructions[m_currentObstId];
            float dxi, dyi;
            int windowWidth = m_parent->GetRootPanel()->m_rectInt_abs.m_w;
            int windowHeight = m_parent->GetRootPanel()->m_rectInt_abs.m_h;
            dxi = dxf*static_cast<float>(g_xDimVisible) / min(m_parent->m_rectFloat_abs.m_w, g_xDimVisible*g_initialScaleUp/windowWidth*2.f);
            dyi = dyf*static_cast<float>(g_yDimVisible) / min(m_parent->m_rectFloat_abs.m_h, g_yDimVisible*g_initialScaleUp/windowHeight*2.f);
            obst.x += dxi;
            obst.y += dyi;
            float u = max(-0.1f,min(0.1f,static_cast<float>(dxi) / (2.f*g_tStep)));
            float v = max(-0.1f,min(0.1f,static_cast<float>(dyi) / (2.f*g_tStep)));
            obst.u = u;
            obst.v = v;
            m_obstructions[m_currentObstId] = obst;
            UpdateDeviceObstructions(g_obst_d, m_currentObstId, obst);
        }
        else
        {
            int simX1, simY1, simX2, simY2;
            int dxi, dyi;
            int windowWidth = m_parent->GetRootPanel()->m_rectInt_abs.m_w;
            int windowHeight = m_parent->GetRootPanel()->m_rectInt_abs.m_h;
            dxi = dxf*static_cast<float>(windowWidth) / 2.f;
            dyi = dyf*static_cast<float>(windowHeight) / 2.f;
            GetSimCoordFrom2DMouseRay(simX1, simY1, xi-dxi, yi-dyi);
            GetSimCoordFrom2DMouseRay(simX2, simY2, xi, yi);
            Obstruction obst = m_obstructions[m_currentObstId];
            obst.x += simX2-simX1;
            obst.y += simY2-simY1;
            float u = max(-0.1f,min(0.1f,static_cast<float>(simX2-simX1) / (2.f*g_tStep)));
            float v = max(-0.1f,min(0.1f,static_cast<float>(simY2-simY1) / (2.f*g_tStep)));
            obst.u = u;
            obst.v = v;
            obst.state = Obstruction::ACTIVE;
            m_obstructions[m_currentObstId] = obst;
            UpdateDeviceObstructions(g_obst_d, m_currentObstId, obst);
        }
    }
}

int GraphicsManager::FindUnusedObstructionId()
{
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if (m_obstructions[i].state == Obstruction::REMOVED || m_obstructions[i].state == Obstruction::INACTIVE)
        {
            return i;
        }
    }
    MessageBox(0, "Object could not be added. You are currently using the maximum number of objects.", "Error", MB_OK);
    return 0;
}

int GraphicsManager::FindClosestObstructionId(Mouse mouse)
{
    int xi, yi;
    GetSimCoordFromMouseCoord(xi, yi, mouse);
    float dist = 999999999999.f;
    int closestObstId = -1;
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if (m_obstructions[i].state != Obstruction::REMOVED)
        {
            float newDist = GetDistanceBetweenTwoPoints(xi, yi, m_obstructions[i].x, m_obstructions[i].y);
            if (newDist < dist)
            {
                dist = newDist;
                closestObstId = i;
            }
        }
    }
    return closestObstId;
}

int GraphicsManager::FindClosestObstructionId(int simX, int simY)
{
    float dist = 999999999999.f;
    int closestObstId = -1;
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if (m_obstructions[i].state != Obstruction::REMOVED)
        {
            float newDist = GetDistanceBetweenTwoPoints(simX, simY, m_obstructions[i].x, m_obstructions[i].y);
            if (newDist < dist)
            {
                dist = newDist;
                closestObstId = i;
            }
        }
    }
    return closestObstId;
}

int GraphicsManager::FindObstructionPointIsInside(int simX, int simY, float tolerance)
{
    float dist = 999999999999.f;
    int closestObstId = -1;
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if (m_obstructions[i].state != Obstruction::REMOVED)
        {
            float newDist = GetDistanceBetweenTwoPoints(simX, simY, m_obstructions[i].x, m_obstructions[i].y);
            if (newDist < dist && newDist < m_obstructions[i].r1+tolerance)
            {
                dist = newDist;
                closestObstId = i;
            }
        }
    }
    //printf("closest obst: %i", closestObstId);
    return closestObstId;
}

bool GraphicsManager::IsInClosestObstruction(Mouse mouse)
{
    int closestObstId = FindClosestObstructionId(mouse);
    int xi, yi;
    GetSimCoordFromMouseCoord(xi, yi, mouse);
    return (GetDistanceBetweenTwoPoints(xi,yi,m_obstructions[closestObstId].x, m_obstructions[closestObstId].y) < m_obstructions[closestObstId].r1);
}

float GetDistanceBetweenTwoPoints(float x1, float y1, float x2, float y2)
{
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrt(dx*dx + dy*dy);
}