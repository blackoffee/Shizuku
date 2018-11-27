#pragma once
#include "../common.h"
#include "ShadingMode.h"
#include "Shizuku.Core/Rect.h"
#include "Shizuku.Core/Types/MinMax.h"
#include "Shizuku.Core/Types/Point.h"
#include "Shizuku.Core/Utilities/Stopwatch.h"
#include <GLEW/glew.h>
#include <glm/glm.hpp>
#include <boost/optional.hpp>
#include <boost/none.hpp>
#include <vector>
#include <string>

#ifdef SHIZUKU_FLOW_EXPORTS  
#define FLOW_API __declspec(dllexport)   
#else  
#define FLOW_API __declspec(dllimport)   
#endif  

using namespace Shizuku::Core;
using namespace Shizuku::Core::Types;
using namespace Shizuku::Flow;

class ShaderManager;
class CudaLbm;
struct float4;

namespace Shizuku{ namespace Flow{
    class GraphicsManager
    {
    public:
        enum TimerKey
        {
            SolveFluid,
            PrepareSurface,
            PrepareFloor,
            ProcessSurface,
            ProcessFloor
        };

    private:
        float m_currentZ = -1000.f;
        //view transformations
        glm::vec3 m_rotate;
        glm::vec3 m_translate;
        int m_currentObstId = -1;
        float m_currentObstSize = 0.f;
        Shape m_currentObstShape = Shape::SQUARE;
        ViewMode m_viewMode;
        bool m_rayTracingPaused = false;
        glm::vec4 m_cameraPosition;
        Obstruction* m_obstructions;
        float m_scaleFactor = 1.f;
        GLint m_viewport[4];
        GLdouble m_modelMatrix[16];
        GLdouble m_projectionMatrix[16];
        MinMax<float> m_contourMinMax;
        ContourVariable m_contourVar;
        ShaderManager* m_graphics;
        bool m_useCuda = true;
        float4* m_rayCastIntersect_d;
        ShadingMode m_surfaceShadingMode;

        Rect<int> m_viewSize;
        Stopwatch m_stopwatch;

    public:
        GraphicsManager();

        void SetViewport(const Rect<int>& size);
        Rect<int>& GetViewport();

        void UseCuda(bool useCuda);

        glm::vec3 GetRotationTransforms();
        glm::vec3 GetTranslationTransforms();

        void SetCurrentObstSize(const float size);

        Shape GetCurrentObstShape();
        void SetCurrentObstShape(const Shape shape);

        ViewMode GetViewMode();
        void SetViewMode(const ViewMode viewMode);

        void SetContourMinMax(const MinMax<float>& p_minMax);
        ContourVariable GetContourVar();
        void SetContourVar(const ContourVariable contourVar);

        void SetObstructionsPointer(Obstruction* obst);

        float GetScaleFactor();
        void SetScaleFactor(const float scaleFactor);
        void SetVelocity(const float p_velocity);
        void SetViscosity(const float p_viscosity);
        void SetTimestepsPerFrame(const int p_steps);

        CudaLbm* GetCudaLbm();
        ShaderManager* GetGraphics();

        bool IsCudaCapable();

        void UpdateViewMatrices();
        void SetUpGLInterop();
        void SetUpShaders();
        void SetUpCuda();
        void RunCuda();
        void RunSurfaceRefraction();
        void RunComputeShader();
        void RunSimulation();
        void RenderFloorToTexture();
        void RenderVbo();
        bool ShouldRefractSurface();

        void SetSurfaceShadingMode(const ShadingMode p_mode);

        void Zoom(const int dir, const float mag);
        void Pan(const Point<int>& p_posDiff);
        void Rotate(const Point<int>& p_posDiff);
        int PickObstruction(const Point<int>& p_pos);
        void UnpickObstruction();
        void MoveObstruction(int obstId, const Point<int>& p_pos, const Point<int>& p_diff);
        void SetRayTracingPausedState(const bool state);
        bool IsRayTracingPaused();
       
        void UpdateViewTransformations();
        void UpdateGraphicsInputs();
        void UpdateDomainDimensions();
        void UpdateObstructionScales();
        void UpdateLbmInputs();
        glm::vec4 GetViewportMatrix();
        glm::mat4 GetModelMatrix();
        glm::mat4 GetProjectionMatrix();
        void SetModelMatrix(glm::mat4 modelMatrix);
        void SetProjectionMatrix(glm::mat4 projMatrix);

        void GetMouseRay(glm::vec3 &rayOrigin, glm::vec3 &rayDir, const Point<int>& p_pos);
        glm::vec4 GetCameraDirection();
        glm::vec4 GetCameraPosition();
        int GetSimCoordFrom3DMouseClickOnObstruction(int &xOut, int &yOut, const Point<int>& p_pos);
        void GetSimCoordFromMouseRay(int &xOut, int &yOut, const Point<int>& p_pos,
            boost::optional<const float> planeZ = boost::none);
        void AddObstruction(const int simX, const int simY);
        void RemoveObstruction(const int simX, const int simY);
        void RemoveSpecifiedObstruction(const int obstId);
        int FindUnusedObstructionId();
        int FindClosestObstructionId(const int simX, const int simY);
        int FindObstructionPointIsInside(const int x, const int y, const float tolerance=0.f);
     
    };
} }
