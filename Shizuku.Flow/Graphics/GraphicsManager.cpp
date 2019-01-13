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

float GraphicsManager::GetFloorZ()
{
    return -1.f;
}

float GraphicsManager::GetWaterHeight()
{
    return GetFloorZ() + m_waterDepth;
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
    float4 rayCastIntersect{ 0, 0, 0, 1e6 };

    cudaMalloc((void **)&m_rayCastIntersect_d, sizeof(float4));
    cudaMemcpy(m_rayCastIntersect_d, &rayCastIntersect, sizeof(float4), cudaMemcpyHostToDevice);

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
    LightFloor(dptr, dptrNormal, floorTemp_d, obst_d, cameraPosition, *domain, m_waterDepth, obstHeight);

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

void GraphicsManager::GetMouseRay(glm::vec3 &rayOrigin, glm::vec3 &rayDir, const Point<int>& p_pos)
{
    glm::mat4 mvp = m_projection*m_modelView;
    glm::mat4 mvpInv = glm::inverse(mvp);

    glm::vec4 v1 = { (float)p_pos.X/(m_viewSize.Width-1)*2.f-1.f, (float)p_pos.Y/(m_viewSize.Height-1)*2.f-1.f, 0.0f*2.f-1.f, 1.0f };
    glm::vec4 v2 = { (float)p_pos.X/(m_viewSize.Width-1)*2.f-1.f, (float)p_pos.Y/(m_viewSize.Height-1)*2.f-1.f, 1.0f*2.f-1.f, 1.0f };
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


glm::vec4 GraphicsManager::GetCameraPosition()
{
    return glm::inverse(m_modelView)*glm::vec4(0, 0, 0, 1);
}

int GraphicsManager::GetSimCoordFrom3DMouseClickOnObstruction(int &xOut, int &yOut, const Point<int>& p_pos)
{
    int xDimVisible = GetCudaLbm()->GetDomain()->GetXDimVisible();
    int yDimVisible = GetCudaLbm()->GetDomain()->GetYDimVisible();
    int rayCastResult;
    int returnVal = 0;

    if (m_useCuda)
    {
        glm::vec3 rayOrigin;
        glm::vec3 rayDir;
        GetMouseRay(rayOrigin, rayDir, p_pos);
        float3 selectedCoordF;

        // map OpenGL buffer object for writing from CUDA
        cudaGraphicsResource* cudaSolutionField = m_graphics->GetCudaPosColorResource();
        float4 *dptr;
        cudaGraphicsMapResources(1, &cudaSolutionField, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cudaSolutionField);

        ObstDefinition* obst_d = GetCudaLbm()->GetDeviceObst();
        Domain* domain = GetCudaLbm()->GetDomain();
        rayCastResult = RayCastMouseClick(selectedCoordF, dptr, m_rayCastIntersect_d,
            float3{ rayOrigin.x, rayOrigin.y, rayOrigin.z }, float3{ rayDir.x, rayDir.y, rayDir.z }, obst_d, *domain);

        cudaGraphicsUnmapResources(1, &cudaSolutionField, 0);

        if (rayCastResult == 0)
        {
            m_currentZ = selectedCoordF.z;

            const Types::Point<int> simPos = SimPosFromModelSpacePos(Types::Point<float>(selectedCoordF.x, selectedCoordF.y));
            xOut = simPos.X;
            yOut = simPos.Y;
        }
        else
        {
            returnVal = 1;
        }
    }
    else
    {
        glm::vec3 rayOrigin, rayDir;
        GetMouseRay(rayOrigin, rayDir, p_pos);
        glm::vec3 selectedCoordF;

        rayCastResult = GetGraphics()->RayCastMouseClick(selectedCoordF, rayOrigin, rayDir);

        if (rayCastResult == 0)
        {
            m_currentZ = selectedCoordF.z;

            xOut = selectedCoordF.x;
            yOut = selectedCoordF.y;
        }
        else
        {
            returnVal = 1;
        }
    }

    return returnVal;
}

//! Hits against water surface and floor
Point<float> GraphicsManager::GetModelSpaceCoordFromScreenPos(const Point<int>& p_screenPos, boost::optional<const float> p_modelSpaceZPos)
{
    glm::vec3 rayOrigin, rayDir;
    GetMouseRay(rayOrigin, rayDir, p_screenPos);

    float t;
    if (p_modelSpaceZPos.is_initialized())
    {
        const float z = p_modelSpaceZPos.value();
        t = (z - rayOrigin.z)/rayDir.z;
        const float xf = rayOrigin.x + t*rayDir.x;
        const float yf = rayOrigin.y + t*rayDir.y;

        return Point<float>(xf, yf);
    }
    else
    {
        const float t1 = (-1.f - rayOrigin.z)/rayDir.z;
        const float t2 = (-1.f + m_waterDepth - rayOrigin.z) / rayDir.z;
        t = std::min(t1, t2);
        float xf = rayOrigin.x + t*rayDir.x;
        float yf = rayOrigin.y + t*rayDir.y;

        if (xf <= 1.f && yf <= 1.f && xf >= -1.f && yf >= -1.f)
        {
            return Point<float>(xf, yf);
        }
        else
        {
            t = std::max(t1, t2);
            xf = rayOrigin.x + t*rayDir.x;
            yf = rayOrigin.y + t*rayDir.y;
            return Point<float>(xf, yf);
        }
    }
}

Point<int> GraphicsManager::GetSimCoordFromScreenPos(const Point<int>& p_screenPos, boost::optional<const float> p_modelSpaceZPos)
{
    Point<float> modelPos = GetModelSpaceCoordFromScreenPos(p_screenPos, p_modelSpaceZPos);
    return SimPosFromModelSpacePos(modelPos);
}

Point<int> GraphicsManager::SimPosFromModelSpacePos(const Point<float>& p_modelPos)
{
    const int xDimVisible = GetCudaLbm()->GetDomain()->GetXDimVisible();
    return Point<int>((p_modelPos.X + 1.f)*0.5f*xDimVisible, (p_modelPos.Y + 1.f)*0.5f*xDimVisible);
}

Point<float> GraphicsManager::ModelSpacePosFromSimPos(const Point<int>& p_simPos)
{
    const int xDimVisible = GetCudaLbm()->GetDomain()->GetXDimVisible();
    return Point<float>(static_cast<float>(p_simPos.X) / xDimVisible*2.f - 1.f, static_cast<float>(p_simPos.Y) / xDimVisible*2.f - 1.f);
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

void GraphicsManager::SelectObstruction(const Point<int>& p_screenPos)
{
	m_obstMgr->AddObstructionToSelection(HitParams{ p_screenPos, m_modelView, m_projection, m_viewSize });
}

int GraphicsManager::PickObstruction(const Point<int>& p_pos)
{
	SelectObstruction(p_pos);
//	std::shared_ptr<ObstDefinition> obst = m_obstMgr->Obsts().lock()->front();
//	m_obstMgr->RemoveObst(*obst);
//    int simX, simY;
//    if (GetSimCoordFrom3DMouseClickOnObstruction(simX, simY, p_pos) == 0)
//    {
//        return FindClosestObstructionId(simX, simY);
//    }
    return 1;
}

bool GraphicsManager::TryStartMoveSelectedObstructions(const Point<int>& p_screenPos)
{
	return m_obstMgr->TryStartMoveSelectedObsts(HitParams{ p_screenPos, m_modelView, m_projection, m_viewSize });
}

void GraphicsManager::MoveSelectedObstructions(const Point<int>& p_screenPos)
{
	m_obstMgr->MoveSelectedObsts(HitParams{ p_screenPos, m_modelView, m_projection, m_viewSize });
}

void GraphicsManager::MoveObstruction(int obstId, const Point<int>& p_pos, const Point<int>& p_diff)
{
    Point<int> simCoord1 = GetSimCoordFromScreenPos(p_pos-p_diff, m_currentZ);
    Point<int> simCoord2 = GetSimCoordFromScreenPos(p_pos, m_currentZ);
    ObstDefinition obst = m_obstructions[obstId];
    Point<float> modelCoord1 = ModelSpacePosFromSimPos(simCoord1);
    Point<float> modelCoord2 = ModelSpacePosFromSimPos(simCoord2);
    obst.x = modelCoord2.X;
    obst.y = modelCoord2.Y;
    const int stepsPerFrame = GetCudaLbm()->GetTimeStepsPerFrame();
    const float u = std::max(-0.1f,std::min(0.1f,static_cast<float>(simCoord2.X-simCoord1.X) / stepsPerFrame));
    const float v = std::max(-0.1f,std::min(0.1f,static_cast<float>(simCoord2.Y-simCoord1.Y) / stepsPerFrame));
    obst.u = u;
    obst.v = v;
    obst.state = State::NORMAL;
    m_obstructions[obstId] = obst;
    if (m_useCuda)
    {
        ObstDefinition* obst_d = GetCudaLbm()->GetDeviceObst();
        //UpdateDeviceObstructions(obst_d, obstId, obst, *GetCudaLbm()->GetDomain());  
        GetGraphics()->UpdateObstructionsUsingComputeShader(obstId, obst, m_scaleFactor);
    }
    else
    {
        GetGraphics()->UpdateObstructionsUsingComputeShader(obstId, obst, m_scaleFactor);
    }
    
    UpdatePillar(obstId, obst);
}

void GraphicsManager::UpdatePillar(const int p_obstId, const ObstDefinition& p_obst)
{
    const Point<float> pillarPos(p_obst.x, p_obst.y);
    const Box<float> pillarSize(2.f*p_obst.r1, 2.f*p_obst.r1, PillarHeightFromDepth(m_waterDepth));
    //m_obstMgr->UpdatePillar(p_obstId, PillarDefinition(pillarPos, pillarSize));
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

void GraphicsManager::AddObstruction(const Point<float>& p_modelSpacePos)
{
    const ObstDefinition obst = { m_currentObstShape, p_modelSpacePos.X, p_modelSpacePos.Y, m_currentObstSize, 0, 0, 0, State::NORMAL };
	m_obstMgr->CreateObst(obst);
}

// TODO: Remove this
void GraphicsManager::AddObstruction(const Point<int>& p_simPos)
{
    if (p_simPos.X > MAX_XDIM - 1 || p_simPos.Y > MAX_YDIM - 1 || p_simPos.X < 0 || p_simPos.Y < 0)
    {
        return;
    }

    const Point<float> modelCoord = ModelSpacePosFromSimPos(p_simPos);

    ObstDefinition obst = { m_currentObstShape, modelCoord.X, modelCoord.Y, m_currentObstSize, 0, 0, 0, State::NORMAL  };

	//m_obstMgr->Obsts().lock()->push_front(obst);
	const int obstId = m_obstMgr->Obsts().lock()->size()+1;

    //const int obstId = FindUnusedObstructionId();
    m_obstructions[obstId] = obst;
    ObstDefinition* obst_d = GetCudaLbm()->GetDeviceObst();
    if (m_useCuda)
    {
        //UpdateDeviceObstructions(obst_d, obstId, obst, *GetCudaLbm()->GetDomain());
        //GetGraphics()->UpdateObstructionsUsingComputeShader(obstId, obst, m_scaleFactor);
		m_obstMgr->CreateObst(obst);
    }
    else
        GetGraphics()->UpdateObstructionsUsingComputeShader(obstId, obst, m_scaleFactor);
    UpdatePillar(obstId, obst);
}

void GraphicsManager::RemoveObstruction(const int simX, const int simY)
{
//    int obstId = FindObstructionPointIsInside(simX,simY,1.f);
//    //RemoveSpecifiedObstruction(obstId);
//	std::shared_ptr<Obst> obst = m_obstMgr->Obsts().lock()->front();
//	//m_obstMgr->RemoveObst(*obst);
}

void GraphicsManager::RemoveSpecifiedObstruction(const int obstId)
{
	m_obstMgr->DeleteSelectedObsts();
//    if (obstId >= 0)
//    {
//        m_obstructions[obstId].state = State::SELECTED;
//        ObstDefinition* obst_d = GetCudaLbm()->GetDeviceObst();
//        if (m_useCuda)
//        {
//            //UpdateDeviceObstructions(obst_d, obstId, m_obstructions[obstId], *GetCudaLbm()->GetDomain());
//            GetGraphics()->UpdateObstructionsUsingComputeShader(obstId, m_obstructions[obstId], m_scaleFactor);
//        }
//        else
//            GetGraphics()->UpdateObstructionsUsingComputeShader(obstId, m_obstructions[obstId], m_scaleFactor);
//        GetGraphics()->RemovePillar(obstId);
//    }
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

int GraphicsManager::FindUnusedObstructionId()
{
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if (m_obstructions[i].state == State::SELECTED)
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
    const Point<float> modelCoord = ModelSpacePosFromSimPos(Point<int>(simX, simY));
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if (m_obstructions[i].state != State::SELECTED)
        {
            float newDist = GetDistanceBetweenTwoPoints(modelCoord.X, modelCoord.Y, m_obstructions[i].x, m_obstructions[i].y);
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
    const Point<float> modelCoord = ModelSpacePosFromSimPos(Point<int>(simX, simY));
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if (m_obstructions[i].state != State::SELECTED)
        {
            float newDist = GetDistanceBetweenTwoPoints(modelCoord.X, modelCoord.Y, m_obstructions[i].x,
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

void GraphicsManager::UpdateGraphicsInputs()
{
    glViewport(0, 0, m_viewSize.Width, m_viewSize.Height);
    UpdateDomainDimensions();
    UpdateObstructionScales();
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

void GraphicsManager::UpdateObstructionScales()
{
    for (int i = 0; i < MAXOBSTS; i++)
    {
        if (m_obstructions[i].state != State::SELECTED)
        {
            if (m_useCuda)
            {
                ObstDefinition* obst_d = GetCudaLbm()->GetDeviceObst();
                //UpdateDeviceObstructions(obst_d, i, m_obstructions[i], *GetCudaLbm()->GetDomain());  
                GetGraphics()->UpdateObstructionsUsingComputeShader(i, m_obstructions[i], m_scaleFactor);
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
