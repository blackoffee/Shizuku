#include "GraphicsManager.h"
#include "WaterSurface.h"
#include "ObstManager.h"
#include "Floor.h"
#include "CudaLbm.h"
#include "kernel.h"
#include "Domain.h"
#include "CudaCheck.h"
#include "TimerKey.h"
#include "ObstDefinition.h"
#include "PillarDefinition.h"
#include "RenderParams.h"
#include "HitParams.h"

#include "Shizuku.Core/Ogl/Shader.h"
#include "Shizuku.Core/Types/Box.h"
#include "Shizuku.Core/Types/Point.h"

#include "helper_cuda.h"

#include <GLEW/glew.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <algorithm>
#include <iostream>

using namespace Shizuku::Core;
using namespace Shizuku::Flow;

namespace
{
    float GetDistanceBetweenTwoPoints(const float x1, const float y1,
        const float x2, const float y2)
    {
        float dx = x2 - x1;
        float dy = y2 - y1;
        return sqrt(dx*dx + dy*dy);
    }

    float PillarHeightFromDepth(const float p_depth)
    {
        return p_depth + 0.3f;
    }


    //TODO this is used in many places
    void GetMouseRay(glm::vec3 &p_rayOrigin, glm::vec3 &p_rayDir, const HitParams& p_params)
    {
        glm::mat4 mvp = p_params.Projection*p_params.Modelview;
        glm::mat4 mvpInv = glm::inverse(mvp);
        glm::vec4 v1 = { (float)p_params.ScreenPos.X / (p_params.ViewSize.Width)*2.f - 1.f, (float)p_params.ScreenPos.Y / (p_params.ViewSize.Height)*2.f - 1.f, 0.0f*2.f - 1.f, 1.0f };
        glm::vec4 v2 = { (float)p_params.ScreenPos.X / (p_params.ViewSize.Width)*2.f - 1.f, (float)p_params.ScreenPos.Y / (p_params.ViewSize.Height)*2.f - 1.f, 1.0f*2.f - 1.f, 1.0f };
        glm::vec4 r1 = mvpInv * v1;
        glm::vec4 r2 = mvpInv * v2;
        p_rayOrigin.x = r1.x / r1.w;
        p_rayOrigin.y = r1.y / r1.w;
        p_rayOrigin.z = r1.z / r1.w;
        p_rayDir.x = r2.x / r2.w - p_rayOrigin.x;
        p_rayDir.y = r2.y / r2.w - p_rayOrigin.y;
        p_rayDir.z = r2.z / r2.w - p_rayOrigin.z;
        float mag = sqrt(p_rayDir.x*p_rayDir.x + p_rayDir.y*p_rayDir.y + p_rayDir.z*p_rayDir.z);
        p_rayDir.x /= mag;
        p_rayDir.y /= mag;
        p_rayDir.z /= mag;
    }

    //! Hits against water surface and floor
    glm::vec3 GetFloorCoordFromScreenPos(const HitParams& p_params, const boost::optional<float> p_modelSpaceZPos, const float p_waterDepth)
    {
        glm::vec3 rayOrigin, rayDir;
        GetMouseRay(rayOrigin, rayDir, p_params);

        float t;
        if (p_modelSpaceZPos.is_initialized())
        {
            const float z = p_modelSpaceZPos.value();
            t = (z - rayOrigin.z) / rayDir.z;
            return rayOrigin + t * rayDir;
        }
        else
        {
            const float t1 = (-1.f - rayOrigin.z) / rayDir.z;
            const float t2 = (-1.f + p_waterDepth - rayOrigin.z) / rayDir.z;
            t = std::min(t1, t2);
            glm::vec3 res = rayOrigin + t * rayDir;

            if (res.x <= 1.f && res.y <= 1.f && res.x >= -1.f && res.y >= -1.f)
            {
                return res;
            }
            else
            {
                t = std::max(t1, t2);
                return rayOrigin + t * rayDir;
            }
        }
    }
}

GraphicsManager::GraphicsManager()
{
    m_waterSurface = std::make_shared<WaterSurface>();
    m_waterSurface->CreateCudaLbm();
    m_floor = std::make_shared<Floor>(m_waterSurface->Ogl);
    m_obstMgr = std::make_shared<ObstManager>(m_waterSurface->Ogl);
    m_obstructions = m_waterSurface->GetCudaLbm()->GetHostObst();
    m_rotate = { 55.f, 60.f, 30.f };
    m_translate = { 0.f, 0.6f, 0.f };
    m_surfaceShadingMode = RayTracing;
    m_waterDepth = 0.2f;
    m_currentObstShape = Shape::SQUARE;
    m_currentObstSize = 0.04f;
    m_drawFloorWireframe = false;
    m_lightProbeEnabled = false;
    m_perspectiveViewAngle = 60.f;
    m_topView = false;
    m_schema = Schema{
        Types::Color(glm::vec4(0.1)), //background
        Types::Color(glm::vec4(0.8)), //obst
        Types::Color(glm::uvec4(255, 255, 153, 255)) //obst highlight
    };

    const int framesForAverage = 20;
    m_timers[TimerKey::SolveFluid] = Stopwatch(framesForAverage);
    m_timers[TimerKey::PrepareFloor] = Stopwatch(framesForAverage);
    m_timers[TimerKey::PrepareSurface] = Stopwatch(framesForAverage);
    m_timers[TimerKey::ProcessFloor] = Stopwatch(framesForAverage);
    m_timers[TimerKey::ProcessSurface] = Stopwatch(framesForAverage);
}

void GraphicsManager::Initialize()
{
    SetUpGLInterop();
    SetUpCuda();
    SetUpShaders();

    m_obstMgr->Initialize();
}

void GraphicsManager::SetUpFrame()
{
    const auto bkg = m_schema.Background.Value();
    glClearColor(bkg.r, bkg.g, bkg.b, bkg.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void GraphicsManager::SetViewport(const Rect<int>& size)
{
    m_viewSize = size;
}

void Shizuku::Flow::GraphicsManager::SetToTopView(bool p_ortho)
{
    m_topView = p_ortho;
}

void Shizuku::Flow::GraphicsManager::SetPerspectiveViewAngle(float p_angleInDeg)
{
    m_perspectiveViewAngle = p_angleInDeg;
}

void GraphicsManager::UseCuda(bool useCuda)
{
    m_useCuda = useCuda;
}

void GraphicsManager::SetCurrentObstSize(const float size)
{
    m_currentObstSize = size;
}

void GraphicsManager::SetCurrentObstShape(const Shape shape)
{
    m_currentObstShape = shape;
}

void GraphicsManager::SetContourMinMax(const MinMax<float>& p_minMax)
{
    m_contourMinMax = p_minMax;
}

ContourVariable GraphicsManager::GetContourVar()
{
    return m_contourVar;
}

void GraphicsManager::SetContourVar(const ContourVariable contourVar)
{
    m_contourVar = contourVar;
}

void GraphicsManager::SetSurfaceShadingMode(const ShadingMode p_mode)
{
    m_surfaceShadingMode = p_mode;
}

Shape GraphicsManager::GetCurrentObstShape()
{
    return m_currentObstShape;
}

void GraphicsManager::SetWaterDepth(const float p_depth)
{
    m_waterDepth = p_depth;
    m_obstMgr->SetWaterHeight(p_depth);
}

float GraphicsManager::GetScaleFactor()
{
    return m_scaleFactor;
}

void GraphicsManager::SetScaleFactor(const float scaleFactor)
{
    m_scaleFactor = scaleFactor;
    m_obstTouched = true;
}

void GraphicsManager::SetVelocity(const float p_velocity)
{
    CudaLbm* cudaLbm = GetCudaLbm();
    cudaLbm->SetInletVelocity(p_velocity);
}

void GraphicsManager::SetViscosity(const float p_viscosity)
{
    CudaLbm* cudaLbm = GetCudaLbm();
    cudaLbm->SetOmega(1.0f/p_viscosity);
}

void GraphicsManager::SetTimestepsPerFrame(const int p_steps)
{
    CudaLbm* cudaLbm = GetCudaLbm();
    cudaLbm->SetTimeStepsPerFrame(p_steps);
}

void GraphicsManager::SetFloorWireframeVisibility(const bool p_visible)
{
    m_drawFloorWireframe = p_visible;
}

void GraphicsManager::EnableLightProbe(const bool p_enable)
{
    m_lightProbeEnabled = p_enable;
}

CudaLbm* GraphicsManager::GetCudaLbm()
{
    return m_waterSurface->GetCudaLbm().get();
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
    if (m_topView)
    {
        m_projection = glm::ortho(-1,1,-1,1);
        glm::mat4 modelMat;
        glm::mat4 scale = glm::scale(glm::mat4(1), glm::vec3(std::max(0.05f, 0.5f+0.1f*m_translate.z)));
        glm::mat4 trans = glm::translate(glm::mat4(1), glm::vec3{ m_translate.x, -0.5f+m_translate.y, 0 });
        modelMat = trans*scale;
        m_modelView = modelMat;
    }
    else
    {
        m_projection = glm::perspective(glm::radians(m_perspectiveViewAngle), static_cast<float>(m_viewSize.Width)/m_viewSize.Height, 0.1f, 100.0f);
        glm::mat4 modelMat;
        glm::mat4 rot = glm::rotate(glm::mat4(1), -m_rotate.x*(float)PI / 180, glm::vec3(1, 0, 0));
        rot = glm::rotate(rot, m_rotate.z*(float)PI / 180, glm::vec3(0, 0, 1));
        glm::mat4 trans = glm::translate(glm::mat4(1), glm::vec3{ m_translate.x, m_translate.y, -2.5f+0.3f*m_translate.z });
        modelMat = trans*rot;
        m_modelView = modelMat;
    }
}

void GraphicsManager::SetUpGLInterop()
{
    m_waterSurface->CreateVboForCudaInterop();
    m_floor->SetVbo(m_waterSurface->GetVbo());
}

void GraphicsManager::SetUpShaders()
{
    m_waterSurface->CompileShaders();
    m_waterSurface->AllocateStorageBuffers();
    m_waterSurface->InitializeObstSsbo();
    UpdateLbmInputs();
    m_waterSurface->InitializeComputeShaderData();

    m_waterSurface->SetUpEnvironmentTexture();
    m_floor->Initialize();
    m_waterSurface->SetUpOutputTexture(m_viewSize);
    m_waterSurface->SetUpSurfaceVao();
    m_waterSurface->SetUpOutputVao();
}

void GraphicsManager::SetUpCuda()
{
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

    CudaLbm* cudaLbm = GetCudaLbm();
    cudaLbm->AllocateDeviceMemory();
    cudaLbm->InitializeDeviceMemory();

    DoInitializeFlow();
}

void GraphicsManager::InitializeFlow()
{
    DoInitializeFlow();
}

void GraphicsManager::DoInitializeFlow()
{
    CudaLbm* cudaLbm = GetCudaLbm();
    const float u = cudaLbm->GetInletVelocity();

    float* fA_d = cudaLbm->GetFA();
    float* fB_d = cudaLbm->GetFB();
    int* im_d = cudaLbm->GetImage();
    float* floor_d = cudaLbm->GetFloorTemp();
    ObstDefinition* obst_d = cudaLbm->GetDeviceObst();

    cudaGraphicsResource* cudaSolutionField = m_waterSurface->GetCudaPosColorResource();
    float4 *dptr;
    cudaGraphicsMapResources(1, &cudaSolutionField, 0);
    size_t num_bytes,num_bytes2;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cudaSolutionField);

    Domain* domain = cudaLbm->GetDomain();
    InitializeDomain(dptr, fA_d, im_d, u, *domain);
    InitializeDomain(dptr, fB_d, im_d, u, *domain);

    InitializeSurface(dptr, *domain);
    InitializeFloor(dptr, *domain);

    cudaGraphicsUnmapResources(1, &cudaSolutionField, 0);
}

void GraphicsManager::RunCuda()
{
    m_timers[TimerKey::SolveFluid].Tick();

    if (m_scaleFactor != m_oldScaleFactor)
    {
        DoInitializeFlow();
        m_oldScaleFactor = m_scaleFactor;
    }

    // map OpenGL buffer object for writing from CUDA
    CudaLbm* cudaLbm = GetCudaLbm();
    cudaGraphicsResource* vbo_resource = m_waterSurface->GetCudaPosColorResource();
    cudaGraphicsResource* normalResource = m_waterSurface->GetCudaNormalResource();
    cudaGraphicsResource* obstResource = m_obstMgr->GetCudaObstsResource();

    float4* dptr;
    float4* dptrNormal;
    ObstDefinition* dObsts;

    gpuErrchk(cudaGraphicsResourceSetMapFlags(vbo_resource, cudaGraphicsRegisterFlagsWriteDiscard));
    gpuErrchk(cudaGraphicsResourceSetMapFlags(normalResource, cudaGraphicsRegisterFlagsWriteDiscard));
    gpuErrchk(cudaGraphicsResourceSetMapFlags(obstResource, cudaGraphicsRegisterFlagsReadOnly));

    cudaGraphicsResource* resources[3] = { vbo_resource, normalResource, obstResource };
    gpuErrchk(cudaGraphicsMapResources(3, resources, 0));
    size_t num_bytes;
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, vbo_resource));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void **)&dptrNormal, &num_bytes, normalResource));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void **)&dObsts, &num_bytes, obstResource));

    UpdateLbmInputs();

    float* floorTemp_d = cudaLbm->GetFloorTemp();
    ObstDefinition* obst_d = cudaLbm->GetDeviceObst();
    ObstDefinition* obst_h = cudaLbm->GetHostObst();

    Domain* domain = cudaLbm->GetDomain();
    if (!cudaLbm->IsPaused())
        MarchSolution(cudaLbm);

    cudaThreadSynchronize();
    m_timers[TimerKey::SolveFluid].Tock();

    m_timers[TimerKey::PrepareFloor].Tick();

    UpdateSolutionVbo(dptr, dptrNormal, cudaLbm, m_contourVar, m_contourMinMax.Min, m_contourMinMax.Max,
        m_waterDepth);
 
    //SetObstructionVelocitiesToZero(obst_h, obst_d, *domain);
    float3 cameraPosition = { m_translate.x, m_translate.y, - m_translate.z };

    if ( !ShouldRefractSurface())
    {
        if (m_surfaceShadingMode == Phong)
        {
            SurfacePhongLighting(dptr, dptrNormal, dObsts, cameraPosition, *domain);
        }
    }

    const float obstHeight = PillarHeightFromDepth(m_waterDepth);
    LightFloor(dptr, dptrNormal, floorTemp_d, dObsts, m_obstMgr->ObstCount(), cameraPosition, *domain, *GetCudaLbm(), m_waterDepth, obstHeight);

    gpuErrchk(cudaGraphicsUnmapResources(3, resources, 0));

    cudaThreadSynchronize();
    m_timers[TimerKey::PrepareFloor].Tock();
}

void GraphicsManager::RunSurfaceRefraction()
{
    m_timers[TimerKey::PrepareSurface].Tick();

    if (ShouldRefractSurface())
    {
        CudaLbm* cudaLbm = GetCudaLbm();
        cudaGraphicsResource* vbo_resource = m_waterSurface->GetCudaPosColorResource();
        cudaGraphicsResource* floorLightTextureResource = m_floor->CudaFloorLightTextureResource();
        cudaGraphicsResource* envTextureResource = m_waterSurface->GetCudaEnvTextureResource();
        cudaGraphicsResource* normalResource = m_waterSurface->GetCudaNormalResource();

        float4* dptr;
        float4* dptrNormal;
        cudaArray* floorLightTexture;
        cudaArray* envTexture;

        size_t num_bytes;

        gpuErrchk(cudaGraphicsResourceSetMapFlags(normalResource, cudaGraphicsRegisterFlagsReadOnly));
        cudaGraphicsResource* resources[4] = { vbo_resource, floorLightTextureResource, envTextureResource, normalResource };
        gpuErrchk(cudaGraphicsMapResources(4, resources, 0));
        gpuErrchk(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, vbo_resource));
        gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&floorLightTexture, floorLightTextureResource, 0, 0));
        gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&envTexture, envTextureResource, 0, 0));
        gpuErrchk(cudaGraphicsResourceGetMappedPointer((void **)&dptrNormal, &num_bytes, normalResource));

        const Point<float> cameraDatumPos(m_cameraPosition.x, m_cameraPosition.y);
        const Box<float> cameraDatumSize(0.05f, 0.05f, m_cameraPosition.z);
        m_waterSurface->UpdateCameraDatum(PillarDefinition(cameraDatumPos, cameraDatumSize));

        ObstDefinition* obst_d = cudaLbm->GetDeviceObst();
        Domain* domain = cudaLbm->GetDomain();
        const float obstHeight = PillarHeightFromDepth(m_waterDepth);
        RefractSurface(dptr, dptrNormal, floorLightTexture, envTexture, obst_d, m_cameraPosition, *domain, m_waterDepth, obstHeight,
            m_surfaceShadingMode == SimplifiedRayTracing);

        gpuErrchk(cudaGraphicsUnmapResources(4, resources, 0));
    }

    cudaThreadSynchronize();
    m_timers[TimerKey::PrepareSurface].Tock();
}

void GraphicsManager::RunComputeShader()
{
    m_waterSurface->RunComputeShader(m_translate, m_contourVar, m_contourMinMax);
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

void GraphicsManager::RenderCausticsToTexture()
{
    CudaLbm* cudaLbm = GetCudaLbm();
    m_floor->RenderCausticsToTexture(*cudaLbm->GetDomain(), m_viewSize);
}

void GraphicsManager::Render()
{
    const float obstHeight = PillarHeightFromDepth(m_waterDepth);
    const RenderParams& params{ m_topView, m_modelView, m_projection, glm::vec3(m_cameraPosition), m_schema };
    m_obstMgr->Render(params);

    CudaLbm* cudaLbm = GetCudaLbm();
    m_floor->Render(*cudaLbm->GetDomain(), params);

    if (m_drawFloorWireframe)
        m_floor->RenderCausticsMesh(*cudaLbm->GetDomain(), params);

    m_waterSurface->Render(m_contourVar, *cudaLbm->GetDomain(),
        params, m_drawFloorWireframe, m_viewSize, obstHeight, m_obstMgr->ObstCount(), m_floor->CausticsTex());

    if (m_lightProbeEnabled)
        m_floor->RenderCausticsBeams(*cudaLbm->GetDomain(), params);
}

bool GraphicsManager::ShouldRefractSurface()
{
    if (m_contourVar == ContourVariable::WATER_RENDERING && 
        (m_surfaceShadingMode == RayTracing || m_surfaceShadingMode == SimplifiedRayTracing))
    {
        return true;
    }
    return false;
}

glm::vec4 GraphicsManager::GetCameraPosition()
{
    return glm::inverse(m_modelView)*glm::vec4(0, 0, 0, 1);
}

//! Hits against water surface and floor
Point<float> GraphicsManager::GetModelSpaceCoordFromScreenPos(const Point<int>& p_screenPos, boost::optional<const float> p_modelSpaceZPos)
{
    const glm::vec3 modelPoint = m_obstMgr->GetSurfaceOrFloorIntersect(HitParams{ p_screenPos, m_modelView, m_projection, m_viewSize });
    return Point<float>(modelPoint.x, modelPoint.y);
}

void GraphicsManager::Pan(const Point<int>& p_posDiff)
{
    const float dx = ((float)p_posDiff.X / m_viewSize.Width)*2.f;
    const float dy = ((float)p_posDiff.Y / m_viewSize.Height)*2.f;
    m_translate.x += dx;
    m_translate.y += dy;
}

void GraphicsManager::Rotate(const Point<int>& p_posDiff)
{
    const float dx = ((float)p_posDiff.X / m_viewSize.Width)*100.f;
    const float dy = ((float)p_posDiff.Y / m_viewSize.Height)*100.f;
    m_rotate.x += dy;
    m_rotate.z += dx;
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

bool GraphicsManager::TryStartMoveSelectedObstructions(const Point<int>& p_screenPos)
{
    return m_obstMgr->TryStartMoveSelectedObsts(HitParams{ p_screenPos, m_modelView, m_projection, m_viewSize });
}

void GraphicsManager::MoveSelectedObstructions(const Point<int>& p_screenPos)
{
    m_obstMgr->MoveSelectedObsts(HitParams{ p_screenPos, m_modelView, m_projection, m_viewSize });
    m_obstTouched = true;
}

void GraphicsManager::AddObstruction(const Point<float>& p_modelSpacePos)
{
    const ObstDefinition obst = { m_currentObstShape, p_modelSpacePos.X, p_modelSpacePos.Y, m_currentObstSize, 0, 0, 0, State::NORMAL };
    m_obstMgr->CreateObst(obst);
    m_obstTouched = true;
}

void GraphicsManager::PreSelectObstruction(const Point<int>& p_screenPos)
{
    m_obstMgr->ClearPreSelection();
    m_obstMgr->AddObstructionToPreSelection(HitParams{ p_screenPos, m_modelView, m_projection, m_viewSize });
}

void GraphicsManager::AddPreSelectionToSelection()
{
    m_obstMgr->AddPreSelectionToSelection();
}

void GraphicsManager::RemovePreSelectionFromSelection()
{
    m_obstMgr->RemovePreSelectionFromSelection();
}

void GraphicsManager::TogglePreSelection()
{
    m_obstMgr->TogglePreSelectionInSelection();
}

void GraphicsManager::DeleteSelectedObstructions()
{
    m_obstMgr->DeleteSelectedObsts();
    m_obstTouched = true;
}

void GraphicsManager::ClearSelection()
{
    m_obstMgr->ClearSelection();
}

int GraphicsManager::ObstCount()
{
    return m_obstMgr->ObstCount();
}

int GraphicsManager::SelectedObstCount()
{
    return m_obstMgr->SelectedObstCount();
}

int GraphicsManager::PreSelectedObstCount()
{
    return m_obstMgr->PreSelectedObstCount();
}

boost::optional<const Info::ObstInfo> GraphicsManager::ObstInfo(const Point<int>& p_screenPos)
{
    return m_obstMgr->ObstInfo(HitParams{ p_screenPos, m_modelView, m_projection, m_viewSize });
}

void GraphicsManager::SetRayTracingPausedState(const bool state)
{
    m_rayTracingPaused = state;
}

bool GraphicsManager::IsRayTracingPaused()
{
    return m_rayTracingPaused;
}

void GraphicsManager::UpdateGraphicsInputs()
{
    glViewport(0, 0, m_viewSize.Width, m_viewSize.Height);
    UpdateDomainDimensions();
    if (!m_rayTracingPaused)
    {
        m_cameraPosition = GetCameraPosition();
    }

    if (!GetCudaLbm()->IsPaused() && m_obstTouched)
    {
        GetCudaLbm()->UpdateDeviceImage(*m_obstMgr);
        m_obstTouched = false;
    }
}

void GraphicsManager::UpdateDomainDimensions()
{
    CudaLbm* cudaLbm = GetCudaLbm();
    cudaLbm->GetDomain()->SetXDimVisible(MAX_XDIM / m_scaleFactor);
    cudaLbm->GetDomain()->SetYDimVisible(MAX_YDIM / m_scaleFactor);
}

void GraphicsManager::UpdateLbmInputs()
{
    float omega = 1.975f;
    CudaLbm* cudaLbm = GetCudaLbm();
    cudaLbm->SetOmega(omega);
    const float u = cudaLbm->GetInletVelocity();
    m_waterSurface->UpdateLbmInputs(u, omega);
}

std::map<TimerKey, Stopwatch>& GraphicsManager::GetTimers()
{
    return m_timers;
}

void GraphicsManager::ProbeLightPaths(const Point<int>& p_screenPos)
{
    const glm::vec3 point = GetFloorCoordFromScreenPos(HitParams{ p_screenPos, m_modelView, m_projection, m_viewSize }, -1.f, m_waterDepth);
    m_floor->SetProbeRegion(Floor::ProbeRegion{ Point<float>(point.x, point.y), Rect<float>(0.1,0.1) });
}
