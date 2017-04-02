#include "GraphicsManager.h"
#include "ShaderManager.h"
#include "CudaLbm.h"
#include "Shader.h"
#include "Panel.h"
#include "Layout.h"
#include "kernel.h"
#include "Domain.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "helper_cuda.h"
#include "helper_cuda_gl.h"
#include <glm/gtc/type_ptr.hpp>
#include <GLEW/glew.h>
#include <algorithm>
#undef min
#undef max

GraphicsManager::GraphicsManager(Panel* panel)
{
    m_parent = panel;
    m_graphics = new ShaderManager;
    m_graphics->CreateCudaLbm();
    m_obstructions = m_graphics->GetCudaLbm()->GetHostObst();
    m_rotate = { 60.f, 0.f, 30.f };
    m_translate = { 0.f, 0.8f, 0.2f };
}

void GraphicsManager::UseCuda(bool useCuda)
{
    m_useCuda = useCuda;
}

float3 GraphicsManager::GetRotationTransforms()
{
    return m_rotate;
}

float3 GraphicsManager::GetTranslationTransforms()
{
    return m_translate;
}

void GraphicsManager::SetCurrentObstSize(const float size)
{
    m_currentObstSize = size;
}

void GraphicsManager::SetCurrentObstShape(const Shape shape)
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

float GraphicsManager::GetContourMinValue()
{
    return m_contourMinValue;
}

float GraphicsManager::GetContourMaxValue()
{
    return m_contourMaxValue;
}

void GraphicsManager::SetContourMinValue(const float contourMinValue)
{
    m_contourMinValue = contourMinValue;
}

void GraphicsManager::SetContourMaxValue(const float contourMaxValue)
{
    m_contourMaxValue = contourMaxValue;
}

ContourVariable GraphicsManager::GetContourVar()
{
    return m_contourVar;
}

void GraphicsManager::SetContourVar(const ContourVariable contourVar)
{
    m_contourVar = contourVar;
}


Shape GraphicsManager::GetCurrentObstShape()
{
    return m_currentObstShape;
}

void GraphicsManager::SetObstructionsPointer(Obstruction* obst)
{
    m_obstructions = m_graphics->GetCudaLbm()->GetHostObst();
}

float GraphicsManager::GetScaleFactor()
{
    return m_scaleFactor;
}

void GraphicsManager::SetScaleFactor(const float scaleFactor)
{
    m_scaleFactor = scaleFactor;
}

CudaLbm* GraphicsManager::GetCudaLbm()
{
    return m_graphics->GetCudaLbm();
}

ShaderManager* GraphicsManager::GetGraphics()
{
    return m_graphics;
}

void GraphicsManager::CenterGraphicsViewToGraphicsPanel(const int leftPanelWidth)
{
    Panel* rootPanel = m_parent->GetRootPanel();
    float scaleUp = rootPanel->GetSlider("Slider_Resolution")->m_sliderBar1->GetValue();
    SetScaleFactor(scaleUp);

    int windowWidth = rootPanel->GetWidth();
    int windowHeight = rootPanel->GetHeight();

    int xDimVisible = GetCudaLbm()->GetDomain()->GetXDimVisible();
    int yDimVisible = GetCudaLbm()->GetDomain()->GetYDimVisible();

    float xTranslation = -((static_cast<float>(windowWidth)-xDimVisible*scaleUp)*0.5
        - static_cast<float>(leftPanelWidth)) / windowWidth*2.f;
    float yTranslation = -((static_cast<float>(windowHeight)-yDimVisible*scaleUp)*0.5)
        / windowHeight*2.f;

    //get view transformations
    float3 cameraPosition = { m_translate.x, 
        m_translate.y, - m_translate.z };

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glTranslatef(xTranslation,yTranslation,0.f);
    glScalef((static_cast<float>(xDimVisible*scaleUp) / windowWidth),
        (static_cast<float>(yDimVisible*scaleUp) / windowHeight), 1.f);


    if (m_viewMode == ViewMode::TWO_DIMENSIONAL)
    {
        glOrtho(-1,1,-1,static_cast<float>(yDimVisible)/xDimVisible*2.f-1.f,-100,20);
    }
    else
    {
        gluPerspective(45.0, static_cast<float>(xDimVisible) / yDimVisible, 0.1, 10.0);
        glTranslatef(m_translate.x, m_translate.y, -2+m_translate.z);
        glRotatef(-m_rotate.x,1,0,0);
        glRotatef(m_rotate.z,0,0,1);
    }
}

void GraphicsManager::SetUpGLInterop()
{
    unsigned int solutionMemorySize = MAX_XDIM*MAX_YDIM * 4 * sizeof(float);
    unsigned int floorSize = MAX_XDIM*MAX_YDIM * 4 * sizeof(float);
    ShaderManager* graphics = GetGraphics();
    graphics->CreateVboForCudaInterop(solutionMemorySize+floorSize);
}

void GraphicsManager::SetUpShaders()
{
    ShaderManager* graphics = GetGraphics();
    graphics->CompileShaders();
    graphics->AllocateStorageBuffers();
    graphics->InitializeObstSsbo();
    UpdateLbmInputs();
    graphics->InitializeComputeShaderData();
}

void GraphicsManager::SetUpCuda()
{
    float4 rayCastIntersect{ 0, 0, 0, 1e6 };

    cudaMalloc((void **)&m_rayCastIntersect_d, sizeof(float4));
    cudaMemcpy(m_rayCastIntersect_d, &rayCastIntersect, sizeof(float4), cudaMemcpyHostToDevice);

    CudaLbm* cudaLbm = GetCudaLbm();
    cudaLbm->AllocateDeviceMemory();
    cudaLbm->InitializeDeviceMemory();

    float u = m_parent->GetRootPanel()->GetSlider("Slider_InletV")->m_sliderBar1->GetValue();

    float* fA_d = cudaLbm->GetFA();
    float* fB_d = cudaLbm->GetFB();
    int* im_d = cudaLbm->GetImage();
    float* floor_d = cudaLbm->GetFloorTemp();
    Obstruction* obst_d = cudaLbm->GetDeviceObst();

    ShaderManager* graphics = GetGraphics();
    cudaGraphicsResource* cudaSolutionField = graphics->GetCudaSolutionGraphicsResource();
    float4 *dptr;
    cudaGraphicsMapResources(1, &cudaSolutionField, 0);
    size_t num_bytes,num_bytes2;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cudaSolutionField);

    Domain* domain = cudaLbm->GetDomain();
    InitializeDomain(dptr, fA_d, im_d, u, *domain);
    InitializeDomain(dptr, fB_d, im_d, u, *domain);

    InitializeFloor(dptr, floor_d, *domain);

    cudaGraphicsUnmapResources(1, &cudaSolutionField, 0);
}

void GraphicsManager::RunCuda()
{
    // map OpenGL buffer object for writing from CUDA
    CudaLbm* cudaLbm = GetCudaLbm();
    ShaderManager* graphics = GetGraphics();
    cudaGraphicsResource* vbo_resource = graphics->GetCudaSolutionGraphicsResource();
    Panel* rootPanel = m_parent->GetRootPanel();

    float4 *dptr;

    cudaGraphicsMapResources(1, &vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, vbo_resource);

    UpdateLbmInputs();
    float u = cudaLbm->GetInletVelocity();
    float omega = cudaLbm->GetOmega();

    float* floorTemp_d = cudaLbm->GetFloorTemp();
    Obstruction* obst_d = cudaLbm->GetDeviceObst();
    Obstruction* obst_h = cudaLbm->GetHostObst();

    Domain* domain = cudaLbm->GetDomain();
    MarchSolution(cudaLbm);
    UpdateSolutionVbo(dptr, cudaLbm, m_contourVar, m_contourMinValue, m_contourMaxValue, m_viewMode);
 
    SetObstructionVelocitiesToZero(obst_h, obst_d);
    float3 cameraPosition = { m_translate.x, m_translate.y, - m_translate.z };

    if (ShouldRenderFloor())
    {
        LightSurface(dptr, obst_d, cameraPosition, *domain);
    }
    LightFloor(dptr, floorTemp_d, obst_d, cameraPosition, *domain);
    CleanUpDeviceVBO(dptr, *domain);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, &vbo_resource, 0);

}

void GraphicsManager::RunComputeShader()
{
    GetGraphics()->RunComputeShader(m_translate);
}

void GraphicsManager::RunSimulation()
{
    if (m_useCuda)
    {
        RunCuda();
    }
    else
    {
        RunComputeShader();
    }
}

void GraphicsManager::RenderVbo()
{
    CudaLbm* cudaLbm = GetCudaLbm();
    ////Colors in Cuda vbo not rendered properly
//    if (m_useCuda)
//    {
//        GetGraphics()->RenderVbo(ShouldRenderFloor(), *cudaLbm->GetDomain(),
//            GetModelMatrix(), GetProjectionMatrix());
//    }
//    else
//    {
        GetGraphics()->RenderVboUsingShaders(ShouldRenderFloor(), *cudaLbm->GetDomain(),
            GetModelMatrix(), GetProjectionMatrix());
//    }
}

bool GraphicsManager::ShouldRenderFloor()
{
    if (m_viewMode == ViewMode::THREE_DIMENSIONAL || m_contourVar == ContourVariable::WATER_RENDERING)
    {
        return true;
    }
    return false;
}


void GraphicsManager::GetSimCoordFromMouseCoord(int &xOut, int &yOut, const int mouseX, const int mouseY)
{
    int xDimVisible = GetCudaLbm()->GetDomain()->GetXDimVisible();
    int yDimVisible = GetCudaLbm()->GetDomain()->GetYDimVisible();
    float xf = intCoordToFloatCoord(mouseX, m_parent->GetRootPanel()->GetWidth());
    float yf = intCoordToFloatCoord(mouseY, m_parent->GetRootPanel()->GetHeight());
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
    int xDimVisible = GetCudaLbm()->GetDomain()->GetXDimVisible();
    int yDimVisible = GetCudaLbm()->GetDomain()->GetYDimVisible();    
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
    const int mouseX, const int mouseY)
{
    int xDimVisible = GetCudaLbm()->GetDomain()->GetXDimVisible();
    int yDimVisible = GetCudaLbm()->GetDomain()->GetYDimVisible();
    float3 rayOrigin, rayDir;
    GetMouseRay(rayOrigin, rayDir, mouseX, mouseY);
    int returnVal = 0;
    float3 selectedCoordF;
    int rayCastResult;

    if (m_useCuda)
    {
        // map OpenGL buffer object for writing from CUDA
        cudaGraphicsResource* cudaSolutionField = m_graphics->GetCudaSolutionGraphicsResource();
        float4 *dptr;
        cudaGraphicsMapResources(1, &cudaSolutionField, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cudaSolutionField);

        Obstruction* obst_d = GetCudaLbm()->GetDeviceObst();
        Domain* domain = GetCudaLbm()->GetDomain();
        rayCastResult = RayCastMouseClick(selectedCoordF, dptr, m_rayCastIntersect_d, 
            rayOrigin, rayDir, obst_d, *domain);
        cudaGraphicsUnmapResources(1, &cudaSolutionField, 0);
    }
    else
    {
        rayCastResult = GetGraphics()->RayCastMouseClick(selectedCoordF, rayOrigin, rayDir);
    }

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

    return returnVal;
}

void GraphicsManager::GetSimCoordFromMouseRay(int &xOut, int &yOut,
    const int mouseX, const int mouseY)
{
    GetSimCoordFromMouseRay(xOut, yOut, mouseX, mouseY, m_currentZ);
}

void GraphicsManager::GetSimCoordFromMouseRay(int &xOut, int &yOut,
    const float mouseXf, const float mouseYf, const float planeZ)
{
    int mouseX = floatCoordToIntCoord(mouseXf, m_parent->GetRootPanel()->GetWidth());
    int mouseY = floatCoordToIntCoord(mouseYf, m_parent->GetRootPanel()->GetHeight());
    GetSimCoordFromMouseRay(xOut, yOut, mouseX, mouseY, planeZ);
}

void GraphicsManager::GetSimCoordFromMouseRay(int &xOut, int &yOut, 
    const int mouseX, const int mouseY, const float planeZ)
{
    int xDimVisible = GetCudaLbm()->GetDomain()->GetXDimVisible();
    float3 rayOrigin, rayDir;
    GetMouseRay(rayOrigin, rayDir, mouseX, mouseY);

    float t = (planeZ - rayOrigin.z)/rayDir.z;
    float xf = rayOrigin.x + t*rayDir.x;
    float yf = rayOrigin.y + t*rayDir.y;

    xOut = (xf + 1.f)*0.5f*xDimVisible;
    yOut = (yf + 1.f)*0.5f*xDimVisible;
}


void GraphicsManager::Pan(const float dx, const float dy)
{
    m_translate.x += dx;
    m_translate.y += dy;
}

void GraphicsManager::Rotate(const float dx, const float dy)
{
    m_rotate.x += dy;
    m_rotate.z += dx;
}

int GraphicsManager::PickObstruction(const float mouseXf, const float mouseYf)
{
    int mouseX = floatCoordToIntCoord(mouseXf, m_parent->GetRootPanel()->GetWidth());
    int mouseY = floatCoordToIntCoord(mouseYf, m_parent->GetRootPanel()->GetHeight());
    return PickObstruction(mouseX, mouseY);
}

int GraphicsManager::PickObstruction(const int mouseX, const int mouseY)
{
    int simX, simY;
    if (GetSimCoordFrom3DMouseClickOnObstruction(simX, simY, mouseX, mouseY) == 0)
    {
        return FindClosestObstructionId(simX, simY);
    }
    return -1;
}


void GraphicsManager::MoveObstruction(int obstId, const float mouseXf, const float mouseYf,
    const float dxf, const float dyf)
{
    int simX1, simY1, simX2, simY2;
    int xi, yi, dxi, dyi;
    int windowWidth = m_parent->GetRootPanel()->GetRectIntAbs().m_w;
    int windowHeight = m_parent->GetRootPanel()->GetRectIntAbs().m_h;
    xi = floatCoordToIntCoord(mouseXf, windowWidth);
    yi = floatCoordToIntCoord(mouseYf, windowHeight);
    dxi = dxf*static_cast<float>(windowWidth) / 2.f;
    dyi = dyf*static_cast<float>(windowHeight) / 2.f;
    GetSimCoordFromMouseRay(simX1, simY1, xi-dxi, yi-dyi);
    GetSimCoordFromMouseRay(simX2, simY2, xi, yi);
    Obstruction obst = m_obstructions[obstId];
    obst.x += simX2-simX1;
    obst.y += simY2-simY1;
    float u = std::max(-0.1f,std::min(0.1f,static_cast<float>(simX2-simX1) / (TIMESTEPS_PER_FRAME)));
    float v = std::max(-0.1f,std::min(0.1f,static_cast<float>(simY2-simY1) / (TIMESTEPS_PER_FRAME)));
    obst.u = u;
    obst.v = v;
    obst.state = State::ACTIVE;
    m_obstructions[obstId] = obst;
    if (m_useCuda)
    {
        Obstruction* obst_d = GetCudaLbm()->GetDeviceObst();
        UpdateDeviceObstructions(obst_d, obstId, obst);  
    }
    else
    {
        GetGraphics()->UpdateObstructionsUsingComputeShader(obstId, obst);
    }
}


void GraphicsManager::Zoom(const int dir, const float mag)
{
    if (dir > 0){
        m_translate.z -= mag;
    }
    else
    {
        m_translate.z += mag;
    }   
}

void GraphicsManager::AddObstruction(const int simX, const int simY)
{
    Obstruction obst = { m_currentObstShape, simX, simY, m_currentObstSize, 0, 0, 0, State::NEW  };
    int obstId = FindUnusedObstructionId();
    m_obstructions[obstId] = obst;
    Obstruction* obst_d = GetCudaLbm()->GetDeviceObst();
    UpdateDeviceObstructions(obst_d, obstId, obst);
    GetGraphics()->UpdateObstructionsUsingComputeShader(obstId, obst);
}

void GraphicsManager::RemoveObstruction(const int simX, const int simY)
{
    int obstId = FindObstructionPointIsInside(simX,simY,1.f);
    RemoveSpecifiedObstruction(obstId);
}

void GraphicsManager::RemoveSpecifiedObstruction(const int obstId)
{
    if (obstId >= 0)
    {
        m_obstructions[obstId].state = State::REMOVED;
        Obstruction* obst_d = GetCudaLbm()->GetDeviceObst();
        UpdateDeviceObstructions(obst_d, obstId, m_obstructions[obstId]);
        GetGraphics()->UpdateObstructionsUsingComputeShader(obstId, m_obstructions[obstId]);
    }
}

void GraphicsManager::MoveObstruction(const int xi, const int yi,
    const float dxf, const float dyf)
{
    int xDimVisible = GetCudaLbm()->GetDomain()->GetXDimVisible();
    int yDimVisible = GetCudaLbm()->GetDomain()->GetYDimVisible();
    if (m_currentObstId > -1)
    {
        int simX1, simY1, simX2, simY2;
        int dxi, dyi;
        int windowWidth = m_parent->GetRootPanel()->GetRectIntAbs().m_w;
        int windowHeight = m_parent->GetRootPanel()->GetRectIntAbs().m_h;
        dxi = dxf*static_cast<float>(windowWidth) / 2.f;
        dyi = dyf*static_cast<float>(windowHeight) / 2.f;
        GetSimCoordFromMouseRay(simX1, simY1, xi-dxi, yi-dyi);
        GetSimCoordFromMouseRay(simX2, simY2, xi, yi);
        Obstruction obst = m_obstructions[m_currentObstId];
        obst.x += simX2-simX1;
        obst.y += simY2-simY1;
        float u = std::max(-0.1f,std::min(0.1f,static_cast<float>(simX2-simX1) / (TIMESTEPS_PER_FRAME)));
        float v = std::max(-0.1f,std::min(0.1f,static_cast<float>(simY2-simY1) / (TIMESTEPS_PER_FRAME)));
        obst.u = u;
        obst.v = v;
        obst.state = State::ACTIVE;
        m_obstructions[m_currentObstId] = obst;
        Obstruction* obst_d = GetCudaLbm()->GetDeviceObst();
        UpdateDeviceObstructions(obst_d, m_currentObstId, obst);
        GetGraphics()->UpdateObstructionsUsingComputeShader(m_currentObstId, obst);
    }
}

int GraphicsManager::FindUnusedObstructionId()
{
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if (m_obstructions[i].state == State::REMOVED || 
            m_obstructions[i].state == State::INACTIVE)
        {
            return i;
        }
    }
    printf("Object could not be added. You are currently using the maximum number of objects.");
    return 0;
}


int GraphicsManager::FindClosestObstructionId(const int simX, const int simY)
{
    float dist = 999999999999.f;
    int closestObstId = -1;
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if (m_obstructions[i].state != State::REMOVED)
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
        if (m_obstructions[i].state != State::REMOVED)
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

bool GraphicsManager::IsInClosestObstruction(const int mouseX, const int mouseY)
{
    int simX, simY;
    GetSimCoordFromMouseCoord(simX, simY, mouseX, mouseY);
    int closestObstId = FindClosestObstructionId(simX, simY);
    float dist = GetDistanceBetweenTwoPoints(simX, simY, m_obstructions[closestObstId].x, 
        m_obstructions[closestObstId].y);
    return (dist < m_obstructions[closestObstId].r1);
}

void GraphicsManager::UpdateViewTransformations()
{
    glGetIntegerv(GL_VIEWPORT, m_viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, m_modelMatrix);
    glGetDoublev(GL_PROJECTION_MATRIX, m_projectionMatrix);
}

void GraphicsManager::UpdateGraphicsInputs()
{
    Panel* rootPanel = m_parent->GetRootPanel();
    m_contourMinValue = Layout::GetCurrentContourSliderValue(*rootPanel, 1);
    m_contourMaxValue = Layout::GetCurrentContourSliderValue(*rootPanel, 2);
    m_currentObstSize = Layout::GetCurrentSliderValue(*rootPanel, "Slider_Size");
}

void GraphicsManager::UpdateLbmInputs()
{
    Panel* rootPanel = m_parent->GetRootPanel();
    float u = rootPanel->GetSlider("Slider_InletV")->m_sliderBar1->GetValue();
    float omega = rootPanel->GetSlider("Slider_Visc")->m_sliderBar1->GetValue();
    CudaLbm* cudaLbm = GetCudaLbm();
    cudaLbm->SetInletVelocity(u);
    cudaLbm->SetOmega(omega);
    ShaderManager* graphics = GetGraphics();
    graphics->UpdateLbmInputs(u, omega);
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

