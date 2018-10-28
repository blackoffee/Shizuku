#include "GraphicsManager.h"
#include "ShaderManager.h"
#include "CudaLbm.h"
#include "Shader.h"
#include "Panel/Slider.h"
#include "Panel/SliderBar.h"
#include "Panel/Panel.h"
#include "Layout.h"
#include "kernel.h"
#include "Domain.h"
#include "CudaCheck.h"
#include <GLEW/glew.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <algorithm>
#undef min
#undef max

GraphicsManager::GraphicsManager(Panel* panel)
{
    m_parent = panel;
    m_graphics = new ShaderManager;
    m_graphics->CreateCudaLbm();
    m_obstructions = m_graphics->GetCudaLbm()->GetHostObst();
    m_rotate = { 45.f, 0.f, 45.f };
    m_translate = { 0.f, 0.f, 0.0f };
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

bool GraphicsManager::IsCudaCapable()
{
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) > 0)
    {
        return true;
    }
    return false;
}

void GraphicsManager::UpdateViewMatrices()
{
    Panel* rootPanel = m_parent->GetRootPanel();
    float scaleUp = rootPanel->GetSlider("Slider_Resolution")->m_sliderBar1->GetValue();
    SetScaleFactor(scaleUp);

    int xDimVisible = GetCudaLbm()->GetDomain()->GetXDimVisible();
    int yDimVisible = GetCudaLbm()->GetDomain()->GetYDimVisible();
;
    SetProjectionMatrix(glm::perspective(45.0f, static_cast<float>(xDimVisible) / yDimVisible, 0.1f, 10.0f));
    //SetProjectionMatrix(glm::ortho(-1,1,-1,1));
    glm::mat4 modelMat;
    modelMat = glm::translate(modelMat, glm::vec3{ 0.2, 0.5, -2.0 });
    modelMat = glm::scale(modelMat, glm::vec3{ 0.7f+0.1f*m_translate.z });
    modelMat = glm::translate(modelMat, glm::vec3{ m_translate.x, m_translate.y, 0.f });
    modelMat = glm::rotate(modelMat, -m_rotate.x*(float)PI/180.0f, glm::vec3{ 1, 0, 0 });
    modelMat = glm::rotate(modelMat, m_rotate.z*(float)PI/180.0f, glm::vec3{ 0, 0, 1 });
    SetModelMatrix(modelMat);
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

    graphics->SetUpTextures();
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

int TimeStepSelector(const int nodes)
{
    if (nodes < 100000)
    {
        return std::min(60.f, 60 - 40 * (float)(nodes - 10000)/100000);
    }
    else
    {
        return std::max(10.f, 20 - 20 * (float)(nodes - 100000)/150000);
    }
}

void GraphicsManager::RunCuda()
{
    // map OpenGL buffer object for writing from CUDA
    CudaLbm* cudaLbm = GetCudaLbm();
    ShaderManager* graphics = GetGraphics();
    cudaGraphicsResource* vbo_resource = graphics->GetCudaSolutionGraphicsResource();
    cudaGraphicsResource* floorTextureResource = graphics->GetCudaFloorLightTextureResource();
    Panel* rootPanel = m_parent->GetRootPanel();

    float4 *dptr;
    cudaArray *floorTexture;

    cudaGraphicsMapResources(1, &vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, vbo_resource);

    cudaGraphicsMapResources(1, &floorTextureResource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&floorTexture, &num_bytes, floorTextureResource);


    UpdateLbmInputs();
    float u = cudaLbm->GetInletVelocity();
    float omega = cudaLbm->GetOmega();

    float* floorTemp_d = cudaLbm->GetFloorTemp();
    Obstruction* obst_d = cudaLbm->GetDeviceObst();
    Obstruction* obst_h = cudaLbm->GetHostObst();

    Domain* domain = cudaLbm->GetDomain();
    cudaLbm->SetTimeStepsPerFrame(TimeStepSelector(domain->GetXDim()*domain->GetYDim()));
    //printf("scalef: %i\n", TimeStepSelector(domain->GetXDim()*domain->GetYDim()));
    MarchSolution(cudaLbm);
    UpdateSolutionVbo(dptr, cudaLbm, m_contourVar, m_contourMinValue, m_contourMaxValue, m_viewMode);
 
    SetObstructionVelocitiesToZero(obst_h, obst_d, m_scaleFactor);
    float3 cameraPosition = { m_translate.x, m_translate.y, - m_translate.z };

    if (ShouldRenderFloor() && !ShouldRefractSurface())
    {
        LightSurface(dptr, obst_d, cameraPosition, *domain);
    }
    LightFloor(dptr, floorTemp_d, obst_d, cameraPosition, *domain);
    CleanUpDeviceVBO(dptr, *domain);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
    cudaGraphicsUnmapResources(1, &floorTextureResource, 0);

}

void GraphicsManager::RunSurfaceRefraction()
{
    if (ShouldRefractSurface())
    {
        // map OpenGL buffer object for writing from CUDA
        CudaLbm* cudaLbm = GetCudaLbm();
        ShaderManager* graphics = GetGraphics();
        cudaGraphicsResource* vbo_resource = graphics->GetCudaSolutionGraphicsResource();
        cudaGraphicsResource* floorLightTextureResource = graphics->GetCudaFloorLightTextureResource();
        cudaGraphicsResource* envTextureResource = graphics->GetCudaEnvTextureResource();
        Panel* rootPanel = m_parent->GetRootPanel();

        float4 *dptr;
        cudaArray *floorLightTexture;
        cudaArray *envTexture;

        cudaGraphicsMapResources(1, &vbo_resource, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, vbo_resource);

        cudaGraphicsMapResources(1, &floorLightTextureResource, 0);
        cudaGraphicsSubResourceGetMappedArray(&floorLightTexture, floorLightTextureResource, 0, 0);
        cudaGraphicsMapResources(1, &envTextureResource, 0);
        cudaGraphicsSubResourceGetMappedArray(&envTexture, envTextureResource, 0, 0);

        Obstruction* obst_d = cudaLbm->GetDeviceObst();

        Domain* domain = cudaLbm->GetDomain();
        glm::mat4 modelMatrixInv = glm::inverse(GetProjectionMatrix()*GetModelMatrix());
        glm::vec4 origin = { 0, 0, 0, 1 };


        glm::vec4 cameraPos;
        glm::vec4 cameraDir = GetCameraDirection();
        //std::cout << "CameraDir: " << cameraDir.x << "," << cameraDir.y << "," << cameraDir.z << std::endl;

        if (!m_rayTracingPaused)
        {
            cameraPos = GetCameraPosition();
            m_cameraPosition = cameraPos;
        }
        else
        {
            cameraPos = m_cameraPosition;
        }

        RefractSurface(dptr, floorLightTexture, envTexture, obst_d, cameraPos, *domain);

        // unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_resource, 0));
        gpuErrchk(cudaGraphicsUnmapResources(1, &floorLightTextureResource, 0));
        gpuErrchk(cudaGraphicsUnmapResources(1, &envTextureResource, 0));
    }
}

void GraphicsManager::RunComputeShader()
{
    GetGraphics()->RunComputeShader(m_translate, m_contourVar, m_contourMinValue, m_contourMaxValue);
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

void GraphicsManager::RenderFloorToTexture()
{
    CudaLbm* cudaLbm = GetCudaLbm();
    GetGraphics()->RenderFloorToTexture(*cudaLbm->GetDomain());
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
    return true;
}

bool GraphicsManager::ShouldRefractSurface()
{
    if (m_contourVar == ContourVariable::WATER_RENDERING)
    {
        return true;
    }
    return false;
}

// only used in 2D mode
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
    glm::mat4 mvp = glm::make_mat4(m_projectionMatrix)*glm::make_mat4(m_modelMatrix);
    glm::mat4 mvpInv = glm::inverse(mvp);
    glm::vec4 v1 = { (float)mouseX/(m_viewport[2]-m_viewport[0])*2.f-1.f, (float)mouseY/(m_viewport[3]-m_viewport[1])*2.f-1.f, 0.0f*2.f-1.f, 1.0f };
    glm::vec4 v2 = { (float)mouseX/(m_viewport[2]-m_viewport[0])*2.f-1.f, (float)mouseY/(m_viewport[3]-m_viewport[1])*2.f-1.f, 1.0f*2.f-1.f, 1.0f };
    glm::vec4 r1 = mvpInv*v1;
    glm::vec4 r2 = mvpInv*v2;
    rayOrigin.x = r1.x/r1.w;
    rayOrigin.y = r1.y/r1.w;
    rayOrigin.z = r1.z/r1.w;
    //printf("Origin: %f, %f, %f\n", r1.x, r1.y, r1.z);
    //printf("Viewport: %f, %f, %f, %f\n", m_viewport[0], m_viewport[1], m_viewport[2], m_viewport[3]);
    rayDir.x = r2.x/r2.w-rayOrigin.x ;
    rayDir.y = r2.y/r2.w-rayOrigin.y ;
    rayDir.z = r2.z/r2.w-rayOrigin.z ;
    float mag = sqrt(rayDir.x*rayDir.x + rayDir.y*rayDir.y + rayDir.z*rayDir.z);
    rayDir.x /= mag;
    rayDir.y /= mag;
    rayDir.z /= mag;


}


glm::vec4 GraphicsManager::GetCameraDirection()
{
    glm::mat4 proj = glm::make_mat4(m_projectionMatrix);
    glm::mat4 model = glm::make_mat4(m_modelMatrix);
    glm::vec4 v = { 0.f, 0.f, 1.f, 1.f };
    //return glm::vec4 { 0.f, 2.f, -1.f, 1.f };
    return glm::inverse(proj*model)*v;
}


glm::vec4 GraphicsManager::GetCameraPosition()
{
    glm::mat4 proj = glm::make_mat4(m_projectionMatrix);
    glm::mat4 model = glm::make_mat4(m_modelMatrix);
    return glm::inverse(proj*model)*glm::vec4(0, 0, 0, 1);
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

    glm::vec4 cameraPos = GetCameraPosition();
    //printf("Origin: %f, %f, %f\n", cameraPos.x, cameraPos.y, cameraPos.z);
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
    obst.x = simX2*m_scaleFactor;
    obst.y = simY2*m_scaleFactor;
    float u = std::max(-0.1f,std::min(0.1f,static_cast<float>(simX2-simX1) / (TIMESTEPS_PER_FRAME)));
    float v = std::max(-0.1f,std::min(0.1f,static_cast<float>(simY2-simY1) / (TIMESTEPS_PER_FRAME)));
    obst.u = u;
    obst.v = v;
    obst.state = State::ACTIVE;
    m_obstructions[obstId] = obst;
    if (m_useCuda)
    {
        Obstruction* obst_d = GetCudaLbm()->GetDeviceObst();
        UpdateDeviceObstructions(obst_d, obstId, obst, m_scaleFactor);  
    }
    else
    {
        GetGraphics()->UpdateObstructionsUsingComputeShader(obstId, obst, m_scaleFactor);
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
    Obstruction obst = { m_currentObstShape, simX*m_scaleFactor, simY*m_scaleFactor, m_currentObstSize, 0, 0, 0, State::NEW  };
    int obstId = FindUnusedObstructionId();
    m_obstructions[obstId] = obst;
    Obstruction* obst_d = GetCudaLbm()->GetDeviceObst();
    if (m_useCuda)
        UpdateDeviceObstructions(obst_d, obstId, obst, m_scaleFactor);
    else
        GetGraphics()->UpdateObstructionsUsingComputeShader(obstId, obst, m_scaleFactor);
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
        if (m_useCuda)
            UpdateDeviceObstructions(obst_d, obstId, m_obstructions[obstId], m_scaleFactor);
        else
            GetGraphics()->UpdateObstructionsUsingComputeShader(obstId, m_obstructions[obstId], m_scaleFactor);
    }
}


void GraphicsManager::SetRayTracingPausedState(const bool state)
{
    m_rayTracingPaused = state;
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
            float newDist = GetDistanceBetweenTwoPoints(simX, simY, m_obstructions[i].x/m_scaleFactor, m_obstructions[i].y/m_scaleFactor);
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
            float newDist = GetDistanceBetweenTwoPoints(simX, simY, m_obstructions[i].x/m_scaleFactor,
                m_obstructions[i].y/m_scaleFactor);
            if (newDist < dist && newDist < m_obstructions[i].r1/m_scaleFactor+tolerance)
            {
                dist = newDist;
                closestObstId = i;
            }
        }
    }
    //printf("closest obst: %i", closestObstId);
    return closestObstId;
}

void GraphicsManager::UpdateViewTransformations()
{
    glGetIntegerv(GL_VIEWPORT, m_viewport);
    //glGetDoublev(GL_MODELVIEW_MATRIX, m_modelMatrix);
    //glGetDoublev(GL_PROJECTION_MATRIX, m_projectionMatrix);
}

void GraphicsManager::UpdateGraphicsInputs()
{
    Panel* rootPanel = m_parent->GetRootPanel();
    m_contourMinValue = Layout::GetCurrentContourSliderValue(*rootPanel, 1);
    m_contourMaxValue = Layout::GetCurrentContourSliderValue(*rootPanel, 2);
    m_currentObstSize = Layout::GetCurrentSliderValue(*rootPanel, "Slider_Size");
    m_scaleFactor = Layout::GetCurrentSliderValue(*rootPanel, "Slider_Resolution");
    UpdateDomainDimensions();
    UpdateObstructionScales();
}

void GraphicsManager::UpdateDomainDimensions()
{
    CudaLbm* cudaLbm = GetCudaLbm();
    cudaLbm->GetDomain()->SetXDimVisible(MAX_XDIM / m_scaleFactor);
    cudaLbm->GetDomain()->SetYDimVisible(MAX_YDIM / m_scaleFactor);
}

void GraphicsManager::UpdateObstructionScales()
{
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if (m_obstructions[i].state != State::REMOVED)
        {
            if (m_useCuda)
            {
                Obstruction* obst_d = GetCudaLbm()->GetDeviceObst();
                UpdateDeviceObstructions(obst_d, i, m_obstructions[i], m_scaleFactor);  
            }
            else
            {
                GetGraphics()->UpdateObstructionsUsingComputeShader(i, m_obstructions[i], m_scaleFactor);
            }
        }
    }
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
    return glm::transpose(glm::make_mat4(m_modelMatrix));
}

glm::mat4 GraphicsManager::GetProjectionMatrix()
{
    return glm::transpose(glm::make_mat4(m_projectionMatrix));
}

void GraphicsManager::SetModelMatrix(glm::mat4 modelMatrix)
{
    const float *source = (const float*)glm::value_ptr(modelMatrix);
    for (int i = 0; i < 16; ++i)
    {
        m_modelMatrix[i] = source[i];
    }
}

void GraphicsManager::SetProjectionMatrix(glm::mat4 projMatrix)
{
    const float *source = (const float*)glm::value_ptr(projMatrix);
    for (int i = 0; i < 16; ++i)
    {
        m_projectionMatrix[i] = source[i];
    }
}



float GetDistanceBetweenTwoPoints(const float x1, const float y1,
    const float x2, const float y2)
{
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrt(dx*dx + dy*dy);
}

