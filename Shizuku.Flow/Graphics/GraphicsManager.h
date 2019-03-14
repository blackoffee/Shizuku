#pragma once
#include "../common.h"
#include "ShadingMode.h"
#include "TimerKey.h"
#include "Schema.h"
#include "Info/ObstInfo.h"
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

class CudaLbm;

namespace Shizuku{ namespace Flow{
    struct ObstDefinition;
    class ObstManager;
    enum Shape;
    class Floor;
    class WaterSurface;

    class GraphicsManager
    {
    private:
        //view transformations
        glm::vec3 m_rotate;
        glm::vec3 m_translate;
        int m_currentObstId = -1;
        float m_currentObstSize = 0.f;
        Shape m_currentObstShape;
        bool m_rayTracingPaused = false;
        glm::vec4 m_cameraPosition;
        ObstDefinition* m_obstructions;
        float m_scaleFactor = 1.f;
        float m_oldScaleFactor = 1.f;
        glm::mat4 m_modelView;
        glm::mat4 m_projection;
        MinMax<float> m_contourMinMax;
        ContourVariable m_contourVar;
        std::shared_ptr<WaterSurface> m_waterSurface;
        std::shared_ptr<Floor> m_floor;
        bool m_useCuda = true;
        ShadingMode m_surfaceShadingMode;
        float m_waterDepth;
        bool m_drawFloorWireframe;
        bool m_lightProbeEnabled;
        bool m_topView;
        float m_perspectiveViewAngle;

        Rect<int> m_viewSize;
        std::map<TimerKey, Stopwatch> m_timers;
        std::shared_ptr<ObstManager> m_obstMgr;
        Schema m_schema;

        bool m_obstTouched;

    public:
        GraphicsManager();

        void Initialize();

        void SetUpFrame();

        void SetViewport(const Rect<int>& size);
        void SetToTopView(bool p_ortho);
        void SetPerspectiveViewAngle(float p_angleInDeg);

        void UseCuda(bool useCuda);

        void SetCurrentObstSize(const float size);

        Shape GetCurrentObstShape();
        void SetCurrentObstShape(const Shape shape);

        void SetContourMinMax(const MinMax<float>& p_minMax);
        ContourVariable GetContourVar();
        void SetContourVar(const ContourVariable contourVar);

        void SetWaterDepth(const float p_depth);
        float GetScaleFactor();
        void SetScaleFactor(const float scaleFactor);
        void SetVelocity(const float p_velocity);
        void SetViscosity(const float p_viscosity);
        void SetTimestepsPerFrame(const int p_steps);
        void SetFloorWireframeVisibility(const bool p_visible);

        void EnableLightProbe(const bool enable);
        void ProbeLightPaths(const Point<int>& p_screenPos);

        CudaLbm* GetCudaLbm();

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
        void UpdateLbmInputs();

        void SetSurfaceShadingMode(const ShadingMode p_mode);

        void Zoom(const int dir, const float mag);
        void Pan(const Point<int>& p_posDiff);
        void Rotate(const Point<int>& p_posDiff);

        void SetRayTracingPausedState(const bool state);
        bool IsRayTracingPaused();

        Point<float> GetModelSpaceCoordFromScreenPos(const Point<int>& p_screenPos, boost::optional<const float> p_modelSpaceZPos = boost::optional<const float>());
        void AddObstruction(const Point<int>& p_simPos);
        void AddObstruction(const Point<float>& p_modelSpacePos);

        bool TryStartMoveSelectedObstructions(const Point<int>& p_screenPos);
        void MoveSelectedObstructions(const Point<int>& p_screenPos);
        void ClearSelection();

        int ObstCount();
        int SelectedObstCount();
        int PreSelectedObstCount();
        boost::optional<const Info::ObstInfo> ObstInfo(const Point<int>& p_screenPos);

        void PreSelectObstruction(const Point<int>& p_screenPos);
        void AddPreSelectionToSelection();
        void RemovePreSelectionFromSelection();
        void TogglePreSelection();
        void DeleteSelectedObstructions();

        std::map<TimerKey, Stopwatch>& GetTimers();

    private:
        void DoInitializeFlow();
        bool ShouldRefractSurface();

        glm::vec4 GetCameraPosition();
    };
} }
