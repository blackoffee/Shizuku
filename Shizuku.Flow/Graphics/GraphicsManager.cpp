#include "GraphicsManager.h"
#include "ShaderManager.h"
#include "ObstManager.h"
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
}

GraphicsManager::GraphicsManager()
{
    m_graphics = new ShaderManager;
    m_graphics->CreateCudaLbm();
	m_obstMgr = std::make_shared<ObstManager>(m_graphics->Ogl);
    m_obstructions = m_graphics->GetCudaLbm()->GetHostObst();
    m_rotate = { 55.f, 60.f, 30.f };
    m_translate = { 0.f, 0.6f, 0.f };
    m_surfaceShadingMode = RayTracing;
    m_waterDepth = 0.2f;
    m_currentObstShape = Shape::SQUARE;
    m_currentObstSize = 0.05f;
    m_drawFloorWireframe = false;
	m_schema = Schema{
		Types::Color(glm::vec4(0.1)),
		Types::Color(glm::vec4(0.8)),
		Types::Color(glm::uvec4(245, 180, 60, 255))
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

Rect<int>& GraphicsManager::GetViewport()
{
    return m_viewSize;
}

void GraphicsManager::UseCuda(bool useCuda)
{
    m_useCuda = useCuda;
}

glm::vec3 GraphicsManager::GetRotationTransforms()
{
    return m_rotate;
}

glm::vec3 GraphicsManager::GetTranslationTransforms()
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

CudaLbm* GraphicsManager::GetCudaLbm()
{
    return m_graphics->GetCudaLbm().get();
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
    m_projection = glm::perspective(glm::radians(60.f), static_cast<float>(m_viewSize.Width)/m_viewSize.Height, 0.1f, 100.0f);
    //SetProjectionMatrix(glm::ortho(-1,1,-1,1));
    glm::mat4 modelMat;
    glm::mat4 rot = glm::rotate(glm::mat4(1), -m_rotate.x*(float)PI / 180, glm::vec3(1, 0, 0));
    rot = glm::rotate(rot, m_rotate.z*(float)PI / 180, glm::vec3(0, 0, 1));
    glm::mat4 trans = glm::translate(glm::mat4(1), glm::vec3{ m_translate.x, m_translate.y, -2.5f+0.3f*m_translate.z });
    modelMat = trans*rot;
    m_modelView = modelMat;
}

void GraphicsManager::SetUpGLInterop()
{
    ShaderManager* graphics = GetGraphics();
    graphics->CreateVboForCudaInterop();
}

void GraphicsManager::SetUpShaders()
{
    ShaderManager* graphics = GetGraphics();
    graphics->CompileShaders();
    graphics->AllocateStorageBuffers();
    graphics->InitializeObstSsbo();
    UpdateLbmInputs();
    graphics->InitializeComputeShaderData();

    graphics->SetUpEnvironmentTexture();
    graphics->SetUpCausticsTexture();
    graphics->SetUpFloorTexture();
    graphics->SetUpOutputTexture(m_viewSize);
    graphics->SetUpSurfaceVao();
    graphics->SetUpFloorVao();
    graphics->SetUpOutputVao();
}

void GraphicsManager::SetUpCuda()
{
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

    ShaderManager* graphics = GetGraphics();
    cudaGraphicsResource* cudaSolutionField = graphics->GetCudaPosColorResource();
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
    ShaderManager* graphics = GetGraphics();
    cudaGraphicsResource* vbo_resource = graphics->GetCudaPosColorResource();
    cudaGraphicsResource* normalResource = graphics->GetCudaNormalResource();
    cudaGraphicsResource* floorTextureResource = graphics->GetCudaFloorLightTextureResource();

    float4* dptr;
    float4* dptrNormal;
    cudaArray *floorTexture;

    size_t num_bytes;
    cudaGraphicsResource* resources[3] = { vbo_resource, normalResource, floorTextureResource };
    cudaGraphicsMapResources(3, resources, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, vbo_resource);
    cudaGraphicsResourceSetMapFlags(vbo_resource, cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsResourceGetMappedPointer((void **)&floorTexture, &num_bytes, floorTextureResource);
    cudaGraphicsResourceGetMappedPointer((void **)&dptrNormal, &num_bytes, normalResource);
    cudaGraphicsResourceSetMapFlags(normalResource, cudaGraphicsRegisterFlagsWriteDiscard);

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
        m_viewMode, m_waterDepth);
 
    SetObstructionVelocitiesToZero(obst_h, obst_d, *domain);
    float3 cameraPosition = { m_translate.x, m_translate.y, - m_translate.z };

    if ( !ShouldRefractSurface())
    {
        if (m_surfaceShadingMode == Phong)
        {
            SurfacePhongLighting(dptr, dptrNormal, obst_d, cameraPosition, *domain);
        }
    }

    const float obstHeight = PillarHeightFromDepth(m_waterDepth);
    LightFloor(dptr, dptrNormal, floorTemp_d, obst_d, cameraPosition, *domain, *GetCudaLbm(), m_waterDepth, obstHeight);

    cudaGraphicsUnmapResources(3, resources, 0);

    cudaThreadSynchronize();
    m_timers[TimerKey::PrepareFloor].Tock();
}

void GraphicsManager::RunSurfaceRefraction()
{
    m_timers[TimerKey::PrepareSurface].Tick();

    if (ShouldRefractSurface())
    {
        CudaLbm* cudaLbm = GetCudaLbm();
        ShaderManager* graphics = GetGraphics();
        cudaGraphicsResource* vbo_resource = graphics->GetCudaPosColorResource();
        cudaGraphicsResource* floorLightTextureResource = graphics->GetCudaFloorLightTextureResource();
        cudaGraphicsResource* envTextureResource = graphics->GetCudaEnvTextureResource();
        cudaGraphicsResource* normalResource = graphics->GetCudaNormalResource();

        float4* dptr;
        float4* dptrNormal;
        cudaArray* floorLightTexture;
        cudaArray* envTexture;

        size_t num_bytes;

        cudaGraphicsResource* resources[4] = { vbo_resource, floorLightTextureResource, envTextureResource, normalResource };
        cudaGraphicsMapResources(4, resources, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, vbo_resource);
        cudaGraphicsSubResourceGetMappedArray(&floorLightTexture, floorLightTextureResource, 0, 0);
        cudaGraphicsSubResourceGetMappedArray(&envTexture, envTextureResource, 0, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&dptrNormal, &num_bytes, normalResource);
        cudaGraphicsResourceSetMapFlags(normalResource, cudaGraphicsRegisterFlagsReadOnly);


        const Point<float> cameraDatumPos(m_cameraPosition.x, m_cameraPosition.y);
        const Box<float> cameraDatumSize(0.05f, 0.05f, m_cameraPosition.z);
        m_graphics->UpdateCameraDatum(PillarDefinition(cameraDatumPos, cameraDatumSize));

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
    GetGraphics()->RunComputeShader(m_translate, m_contourVar, m_contourMinMax);
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
    GetGraphics()->RenderCausticsToTexture(*cudaLbm->GetDomain(), m_viewSize);
}

void GraphicsManager::Render()
{
    CudaLbm* cudaLbm = GetCudaLbm();
    const float obstHeight = PillarHeightFromDepth(m_waterDepth);
	const RenderParams& params{ m_modelView, m_projection, glm::vec3(m_cameraPosition), m_schema };
    GetGraphics()->Render(m_surfaceShadingMode, *cudaLbm->GetDomain(),
        params, m_drawFloorWireframe, m_viewSize, obstHeight, m_obstMgr->ObstCount());
	m_obstMgr->Render(params);
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
}

void GraphicsManager::AddObstruction(const Point<float>& p_modelSpacePos)
{
    const ObstDefinition obst = { m_currentObstShape, p_modelSpacePos.X, p_modelSpacePos.Y, m_currentObstSize, 0, 0, 0, State::NORMAL };
	m_obstMgr->CreateObst(obst);
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

void GraphicsManager::DeleteSelectedObstructions()
{
	m_obstMgr->DeleteSelectedObsts();
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

	// TODO: don't update every loop
	GetCudaLbm()->UpdateDeviceImage(*m_obstMgr);
}

void GraphicsManager::UpdateDomainDimensions()
{
    CudaLbm* cudaLbm = GetCudaLbm();
    cudaLbm->GetDomain()->SetXDimVisible(MAX_XDIM / m_scaleFactor);
    cudaLbm->GetDomain()->SetYDimVisible(MAX_YDIM / m_scaleFactor);
}

void GraphicsManager::UpdateLbmInputs()
{
    float omega = 1.97f;
    CudaLbm* cudaLbm = GetCudaLbm();
    cudaLbm->SetOmega(omega);
    const float u = cudaLbm->GetInletVelocity();
    ShaderManager* graphics = GetGraphics();
    graphics->UpdateLbmInputs(u, omega);
}

std::map<TimerKey, Stopwatch>& GraphicsManager::GetTimers()
{
    return m_timers;
}
