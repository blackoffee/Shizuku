#include "GraphicsManager.h"
#include "Mouse.h"
#include "kernel.h"


extern Obstruction* g_obst_d;
extern Domain g_simDomain;
extern cudaGraphicsResource *g_cudaSolutionField;

GraphicsManager::GraphicsManager(Panel* panel)
{
    m_parent = panel;
}

float3 GraphicsManager::GetRotationTransforms()
{
    return float3{ m_rotate_x, m_rotate_y, m_rotate_z };
}

float3 GraphicsManager::GetTranslationTransforms()
{
    return float3{ m_translate_x, m_translate_y, m_translate_z };
}

void GraphicsManager::SetCurrentObstSize(const float size)
{
    m_currentObstSize = size;
}

void GraphicsManager::SetCurrentObstShape(const Obstruction::Shape shape)
{
    m_currentObstShape = shape;
}

ViewMode GraphicsManager::GetViewMode()
{
    return m_viewMode;
}

void GraphicsManager::SetViewMode(const ViewMode viewMode)
{
    m_viewMode = viewMode;
}

ContourVariable GraphicsManager::GetContourVar()
{
    return m_contourVar;
}

void GraphicsManager::SetContourVar(const ContourVariable contourVar)
{
    m_contourVar = contourVar;
}


Obstruction::Shape GraphicsManager::GetCurrentObstShape()
{
    return m_currentObstShape;
}

void GraphicsManager::SetObstructionsPointer(Obstruction* obst)
{
    m_obstructions = obst;
}

bool GraphicsManager::IsPaused()
{
    return m_paused;
}

void GraphicsManager::TogglePausedState()
{
    m_paused = !m_paused;
}

float GraphicsManager::GetScaleFactor()
{
    return m_scaleFactor;
}

void GraphicsManager::SetScaleFactor(const float scaleFactor)
{
    m_scaleFactor = scaleFactor;
}

void GraphicsManager::GetSimCoordFromMouseCoord(int &xOut, int &yOut, Mouse mouse)
{
    int xDimVisible = g_simDomain.GetXDimVisible();
    int yDimVisible = g_simDomain.GetYDimVisible();
    float xf = intCoordToFloatCoord(mouse.m_x, mouse.m_winW);
    float yf = intCoordToFloatCoord(mouse.m_y, mouse.m_winH);
    RectFloat coordsInRelFloat = RectFloat(xf, yf, 1.f, 1.f) / m_parent->GetRectFloatAbs();
    float graphicsToSimDomainScalingFactorX = static_cast<float>(xDimVisible) /
        std::min(static_cast<float>(m_parent->GetRectIntAbs().m_w), MAX_XDIM*m_scaleFactor);
    float graphicsToSimDomainScalingFactorY = static_cast<float>(yDimVisible) /
        std::min(static_cast<float>(m_parent->GetRectIntAbs().m_h), MAX_YDIM*m_scaleFactor);
    xOut = floatCoordToIntCoord(coordsInRelFloat.m_x, m_parent->GetRectIntAbs().m_w)*
        graphicsToSimDomainScalingFactorX;
    yOut = floatCoordToIntCoord(coordsInRelFloat.m_y, m_parent->GetRectIntAbs().m_h)*
        graphicsToSimDomainScalingFactorY;
}

void GraphicsManager::GetSimCoordFromFloatCoord(int &xOut, int &yOut, 
    const float xf, const float yf)
{
    int xDimVisible = g_simDomain.GetXDimVisible();
    int yDimVisible = g_simDomain.GetYDimVisible();
    RectFloat coordsInRelFloat = RectFloat(xf, yf, 1.f, 1.f) / m_parent->GetRectFloatAbs();
    float graphicsToSimDomainScalingFactorX = static_cast<float>(xDimVisible) /
        std::min(static_cast<float>(m_parent->GetRectIntAbs().m_w), MAX_XDIM*m_scaleFactor);
    float graphicsToSimDomainScalingFactorY = static_cast<float>(yDimVisible) /
        std::min(static_cast<float>(m_parent->GetRectIntAbs().m_h), MAX_YDIM*m_scaleFactor);
    xOut = floatCoordToIntCoord(coordsInRelFloat.m_x, m_parent->GetRectIntAbs().m_w)*
        graphicsToSimDomainScalingFactorX;
    yOut = floatCoordToIntCoord(coordsInRelFloat.m_y, m_parent->GetRectIntAbs().m_h)*
        graphicsToSimDomainScalingFactorY;
}

void GraphicsManager::GetMouseRay(float3 &rayOrigin, float3 &rayDir,
    const int mouseX, const int mouseY)
{
    double x, y, z;
    gluUnProject(mouseX, mouseY, 0.0f, m_modelMatrix, m_projectionMatrix, m_viewport, &x, &y, &z);
    //printf("Origin: %f, %f, %f\n", x, y, z);
    rayOrigin.x = x;
    rayOrigin.y = y;
    rayOrigin.z = z;
    gluUnProject(mouseX, mouseY, 1.0f, m_modelMatrix, m_projectionMatrix, m_viewport, &x, &y, &z);
    rayDir.x = x-rayOrigin.x;
    rayDir.y = y-rayOrigin.y;
    rayDir.z = z-rayOrigin.z;
    float mag = sqrt(rayDir.x*rayDir.x + rayDir.y*rayDir.y + rayDir.z*rayDir.z);
    rayDir.x /= mag;
    rayDir.y /= mag;
    rayDir.z /= mag;
}

int GraphicsManager::GetSimCoordFrom3DMouseClickOnObstruction(int &xOut, int &yOut,
    Mouse mouse)
{
    int xDimVisible = g_simDomain.GetXDimVisible();
    int yDimVisible = g_simDomain.GetYDimVisible();
    float3 rayOrigin, rayDir;
    GetMouseRay(rayOrigin, rayDir, mouse.GetX(), mouse.GetY());
    int returnVal = 0;

    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    cudaGraphicsMapResources(1, &g_cudaSolutionField, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, g_cudaSolutionField);

    float3 selectedCoordF;
    int rayCastResult = RayCastMouseClick(selectedCoordF, dptr, m_rayCastIntersect_d, 
        rayOrigin, rayDir, g_obst_d, g_simDomain);
    if (rayCastResult == 0)
    {
        m_currentZ = selectedCoordF.z;

        xOut = (selectedCoordF.x + 1.f)*0.5f*xDimVisible;
        yOut = (selectedCoordF.y + 1.f)*0.5f*xDimVisible;
    }
    else
    {
        returnVal = 1;
    }

    cudaGraphicsUnmapResources(1, &g_cudaSolutionField, 0);

    return returnVal;
}

// get simulation coordinates from mouse ray casting on plane on m_currentZ
void GraphicsManager::GetSimCoordFrom2DMouseRay(int &xOut, int &yOut, Mouse mouse)
{
    int xDimVisible = g_simDomain.GetXDimVisible();
    float3 rayOrigin, rayDir;
    GetMouseRay(rayOrigin, rayDir, mouse.GetX(), mouse.GetY());

    float t = (m_currentZ - rayOrigin.z)/rayDir.z;
    float xf = rayOrigin.x + t*rayDir.x;
    float yf = rayOrigin.y + t*rayDir.y;

    xOut = (xf + 1.f)*0.5f*xDimVisible;
    yOut = (yf + 1.f)*0.5f*xDimVisible;
}

void GraphicsManager::GetSimCoordFrom2DMouseRay(int &xOut, int &yOut, 
    const int mouseX, const int mouseY)
{
    int xDimVisible = g_simDomain.GetXDimVisible();
    float3 rayOrigin, rayDir;
    GetMouseRay(rayOrigin, rayDir, mouseX, mouseY);

    float t = (m_currentZ - rayOrigin.z)/rayDir.z;
    float xf = rayOrigin.x + t*rayDir.x;
    float yf = rayOrigin.y + t*rayDir.y;

    xOut = (xf + 1.f)*0.5f*xDimVisible;
    yOut = (yf + 1.f)*0.5f*xDimVisible;
}

void GraphicsManager::ClickDown(Mouse mouse)
{
    int mod = glutGetModifiers();
    if (m_viewMode == ViewMode::TWO_DIMENSIONAL)
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

void GraphicsManager::Drag(const int xi, const int yi, const float dxf,
    const float dyf, const int button)
{
    if (button == GLUT_LEFT_BUTTON)
    {
        MoveObstruction(xi,yi,dxf,dyf);
    }
    else if (button == GLUT_MIDDLE_BUTTON)
    {
        int mod = glutGetModifiers();
        if (mod == GLUT_ACTIVE_CTRL)
        {
            m_translate_x += dxf;
            m_translate_y += dyf;
        }
        else
        {
            m_rotate_x += dyf*45.f;
            m_rotate_z += dxf*45.f;
        }

    }
}

void GraphicsManager::Wheel(const int button, const int dir, const int x, const int y)
{
    if (dir > 0){
        m_translate_z -= 0.3f;
    }
    else
    {
        m_translate_z += 0.3f;
    }
}

void GraphicsManager::AddObstruction(Mouse mouse)
{
    int xi, yi;
    GetSimCoordFromMouseCoord(xi, yi, mouse);
    Obstruction obst = { m_currentObstShape, xi, yi, m_currentObstSize, 0, 0, 0, Obstruction::NEW };
    int obstId = FindUnusedObstructionId();
    m_obstructions[obstId] = obst;
    UpdateDeviceObstructions(g_obst_d, obstId, obst);
}

void GraphicsManager::AddObstruction(const int simX, const int simY)
{
    Obstruction obst = { m_currentObstShape, simX, simY, m_currentObstSize, 0, 0, 0, Obstruction::NEW  };
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

void GraphicsManager::RemoveObstruction(const int simX, const int simY)
{
    int obstId = FindObstructionPointIsInside(simX,simY,1.f);
    if (obstId >= 0)
    {
        m_obstructions[obstId].state = Obstruction::REMOVED;
        UpdateDeviceObstructions(g_obst_d, obstId, m_obstructions[obstId]);
    }
}

void GraphicsManager::MoveObstruction(const int xi, const int yi,
    const float dxf, const float dyf)
{
    int xDimVisible = g_simDomain.GetXDimVisible();
    int yDimVisible = g_simDomain.GetYDimVisible();
    if (m_currentObstId > -1)
    {
        if (m_viewMode == ViewMode::TWO_DIMENSIONAL)
        {
            Obstruction obst = m_obstructions[m_currentObstId];
            float dxi, dyi;
            int windowWidth = m_parent->GetRootPanel()->GetRectIntAbs().m_w;
            int windowHeight = m_parent->GetRootPanel()->GetRectIntAbs().m_h;
            dxi = dxf*static_cast<float>(xDimVisible) / 
                std::min(m_parent->GetRectFloatAbs().m_w, xDimVisible*m_scaleFactor/windowWidth*2.f);
            dyi = dyf*static_cast<float>(yDimVisible) / 
                std::min(m_parent->GetRectFloatAbs().m_h, yDimVisible*m_scaleFactor/windowHeight*2.f);
            obst.x += dxi;
            obst.y += dyi;
            float u = std::max(-0.1f,std::min(0.1f,static_cast<float>(dxi) / (TIMESTEPS_PER_FRAME)));
            float v = std::max(-0.1f,std::min(0.1f,static_cast<float>(dyi) / (TIMESTEPS_PER_FRAME)));
            obst.u = u;
            obst.v = v;
            m_obstructions[m_currentObstId] = obst;
            UpdateDeviceObstructions(g_obst_d, m_currentObstId, obst);
        }
        else
        {
            int simX1, simY1, simX2, simY2;
            int dxi, dyi;
            int windowWidth = m_parent->GetRootPanel()->GetRectIntAbs().m_w;
            int windowHeight = m_parent->GetRootPanel()->GetRectIntAbs().m_h;
            dxi = dxf*static_cast<float>(windowWidth) / 2.f;
            dyi = dyf*static_cast<float>(windowHeight) / 2.f;
            GetSimCoordFrom2DMouseRay(simX1, simY1, xi-dxi, yi-dyi);
            GetSimCoordFrom2DMouseRay(simX2, simY2, xi, yi);
            Obstruction obst = m_obstructions[m_currentObstId];
            obst.x += simX2-simX1;
            obst.y += simY2-simY1;
            float u = std::max(-0.1f,std::min(0.1f,static_cast<float>(simX2-simX1) / (TIMESTEPS_PER_FRAME)));
            float v = std::max(-0.1f,std::min(0.1f,static_cast<float>(simY2-simY1) / (TIMESTEPS_PER_FRAME)));
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
        if (m_obstructions[i].state == Obstruction::REMOVED || 
            m_obstructions[i].state == Obstruction::INACTIVE)
        {
            return i;
        }
    }
    MessageBox(0, "Object could not be added. You are currently using the maximum number of objects.",
        "Error", MB_OK);
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

int GraphicsManager::FindClosestObstructionId(const int simX, const int simY)
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

int GraphicsManager::FindObstructionPointIsInside(const int simX, const int simY,
    const float tolerance)
{
    float dist = 999999999999.f;
    int closestObstId = -1;
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if (m_obstructions[i].state != Obstruction::REMOVED)
        {
            float newDist = GetDistanceBetweenTwoPoints(simX, simY, m_obstructions[i].x,
                m_obstructions[i].y);
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
    float dist = GetDistanceBetweenTwoPoints(xi, yi, m_obstructions[closestObstId].x, 
        m_obstructions[closestObstId].y);
    return (dist < m_obstructions[closestObstId].r1);
}

void GraphicsManager::UpdateViewTransformations()
{
    glGetIntegerv(GL_VIEWPORT, m_viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, m_modelMatrix);
    glGetDoublev(GL_PROJECTION_MATRIX, m_projectionMatrix);
}

glm::vec4 GraphicsManager::GetViewportMatrix()
{
    return glm::make_vec4(m_viewport);
}

glm::mat4 GraphicsManager::GetModelMatrix()
{
    return glm::make_mat4(m_modelMatrix);
}

glm::mat4 GraphicsManager::GetProjectionMatrix()
{
    return glm::make_mat4(m_projectionMatrix);
}


float GetDistanceBetweenTwoPoints(const float x1, const float y1,
    const float x2, const float y2)
{
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrt(dx*dx + dy*dy);
}