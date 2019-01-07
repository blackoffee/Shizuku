#pragma once
#include "../common.h"
#include "ShadingMode.h"
#include "TimerKey.h"
#include "Shizuku.Core/Rect.h"
#include "Shizuku.Core/Types/MinMax.h"
#include "Shizuku.Core/Types/Point.h"
#include "Shizuku.Core/Utilities/Stopwatch.h"
#include <GLEW/glew.h>
#include <glm/glm.hpp>
#include <boost/optional.hpp>
#include <boost/none.hpp>
#include <memory>
#include <vector>
#include <string>
#include <map>

using namespace Shizuku::Core;
using namespace Shizuku::Core::Types;
using namespace Shizuku::Flow;

class ShaderManager;
class CudaLbm;
struct float4;

namespace Shizuku{ namespace Flow{
    struct Obstruction;
    class ObstManager;
    enum Shape;

    class GraphicsManager
    {
    private:
        float m_currentZ = -1000.f;
        //view transformations
        glm::vec3 m_rotate;
        glm::vec3 m_translate;
        int m_currentObstId = -1;
        float m_currentObstSize = 0.f;
        Shape m_currentObstShape;
        ViewMode m_viewMode;
        bool m_rayTracingPaused = false;
        glm::vec4 m_cameraPosition;
        Obstruction* m_obstructions;
        float m_scaleFactor = 1.f;
        float m_oldScaleFactor = 1.f;
        glm::mat4 m_modelView;
        glm::mat4 m_projection;
        MinMax<float> m_contourMinMax;
        ContourVariable m_contourVar;
        ShaderManager* m_graphics;
        bool m_useCuda = true;
        float4* m_rayCastIntersect_d;
        ShadingMode m_surfaceShadingMode;
        float m_waterDepth;
        bool m_drawFloorWireframe;

        Rect<int> m_viewSize;
        std::map<TimerKey, Stopwatch> m_timers;
        std::shared_ptr<ObstManager> m_obstMgr;

    public:
        GraphicsManager();

		void Initialize();

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

        float GetFloorZ();
        float GetWaterHeight();
        void SetWaterDepth(const float p_depth);
        float GetScaleFactor();
        void SetScaleFactor(const float scaleFactor);
        void SetVelocity(const float p_velocity);
        void SetViscosity(const float p_viscosity);
        void SetTimestepsPerFrame(const int p_steps);
        void SetFloorWireframeVisibility(const bool p_visible);

        CudaLbm* GetCudaLbm();
        ShaderManager* GetGraphics();

        bool IsCudaCapable();

        void UpdateGraphicsInputs();
        void UpdateViewMatrices();
        void SetUpGLInterop();
        void SetUpShaders();
        void SetUpCuda();
        void RunCuda();
        void RunSurfaceRefraction();
        void RunComputeShader();
        void RunSimulation();
        void RenderCausticsToTexture();
        void Render();
        void InitializeFlow();
        void UpdateDomainDimensions();
        void UpdateObstructionScales();
        void UpdateLbmInputs();

        void SetSurfaceShadingMode(const ShadingMode p_mode);

        void Zoom(const int dir, const float mag);
        void Pan(const Point<int>& p_posDiff);
        void Rotate(const Point<int>& p_posDiff);

        void SetRayTracingPausedState(const bool state);
        bool IsRayTracingPaused();
       
        Point<float> GetModelSpaceCoordFromScreenPos(const Point<int>& p_screenPos, boost::optional<const float> p_modelSpaceZPos = boost::optional<const float>());
        Point<int> GetSimCoordFromScreenPos(const Point<int>& p_screenPos, boost::optional<const float> p_modelSpaceZPos = boost::optional<const float>());
        void AddObstruction(const Point<int>& p_simPos);
        void AddObstruction(const Point<float>& p_modelSpacePos);
        void RemoveObstruction(const int simX, const int simY);
        void RemoveSpecifiedObstruction(const int obstId);
        int PickObstruction(const Point<int>& p_pos);
        void MoveObstruction(int obstId, const Point<int>& p_pos, const Point<int>& p_diff);
     
        std::map<TimerKey, Stopwatch>& GetTimers();

    private:
        void DoInitializeFlow();
        bool ShouldRefractSurface();

        void GetMouseRay(glm::vec3 &rayOrigin, glm::vec3 &rayDir, const Point<int>& p_pos);
        int GetSimCoordFrom3DMouseClickOnObstruction(int &xOut, int &yOut, const Point<int>& p_pos);
        glm::vec4 GetCameraPosition();

        void UpdatePillar(const int p_obstId, const Obstruction& p_obst);
        int FindUnusedObstructionId();
        int FindClosestObstructionId(const int simX, const int simY);
        int FindObstructionPointIsInside(const int x, const int y, const float tolerance=0.f);

        Point<int> SimPosFromModelSpacePos(const Point<float>& p_modelPos);
        Point<float> ModelSpacePosFromSimPos(const Point<int>& p_simPos);
    };
} }
