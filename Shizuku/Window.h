#pragma once
#include "TimeHistory.h"
#include "Shizuku.Core/Utilities/FpsTracker.h"
#include "Shizuku.Core/Rect.h"
#include "Shizuku.Core/Types/MinMax.h"
#include <memory>

struct GLFWwindow;

namespace Shizuku{
    namespace Flow{
        class Flow;
        class Query;
        namespace Command{
            class Zoom;
            class Pan;
            class Rotate;
            class AddObstruction;
            class RemoveObstruction;
            class MoveObstruction;
			class PreSelectObstruction;
			class AddPreSelectionToSelection;
			class DeleteSelectedObstructions;
			class TogglePreSelection;
			class ClearSelection;
            class PauseSimulation; 
            class PauseRayTracing;
            class RestartSimulation;
            class SetSimulationScale;
            class SetTimestepsPerFrame;
            class SetContourMode;
            class SetContourMinMax;
            class SetSurfaceShadingMode;
            class SetInletVelocity;
            class SetWaterDepth;
            class SetFloorWireframeVisibility;
			class ProbeLightPaths;
            enum ContourMode;
            enum SurfaceShadingMode;
        }
    }
}

using namespace Shizuku::Core;
using namespace Shizuku::Core::Types;
using namespace Shizuku::Flow::Command;

namespace Shizuku{ namespace Presentation{
    class Window
    {
    private:
        std::shared_ptr<Zoom> m_zoom;
        std::shared_ptr<Pan> m_pan;
        std::shared_ptr<Rotate> m_rotate;
        std::shared_ptr<AddObstruction> m_addObstruction;
        std::shared_ptr<MoveObstruction> m_moveObstruction;
		std::shared_ptr<PreSelectObstruction> m_preSelectObst;
		std::shared_ptr<AddPreSelectionToSelection> m_addPreSelectionToSelection;
		std::shared_ptr<TogglePreSelection> m_togglePreSelection;
		std::shared_ptr<DeleteSelectedObstructions> m_deleteSelectedObstructions;
		std::shared_ptr<ClearSelection> m_clearSelection;
        std::shared_ptr<PauseSimulation> m_pauseSimulation;
        std::shared_ptr<PauseRayTracing> m_pauseRayTracing;
        std::shared_ptr<RestartSimulation> m_restartSimulation;
        std::shared_ptr<SetSimulationScale> m_setSimulationScale;
        std::shared_ptr<SetTimestepsPerFrame> m_timestepsPerFrame;
        std::shared_ptr<SetInletVelocity> m_setVelocity;
        std::shared_ptr<SetContourMode> m_setContourMode;
        std::shared_ptr<SetContourMinMax> m_setContourMinMax;
        std::shared_ptr<SetSurfaceShadingMode> m_setSurfaceShadingMode;
        std::shared_ptr<SetWaterDepth> m_setDepth;
        std::shared_ptr<SetFloorWireframeVisibility> m_setFloorWireframeVisibility;
        std::shared_ptr<ProbeLightPaths> m_probeLightPaths;
        FpsTracker m_fpsTracker;
        std::shared_ptr<Shizuku::Flow::Query> m_query;
        GLFWwindow* m_window;
        Rect<int> m_size;

        float m_resolution;
        int m_timesteps;
        float m_velocity;
        float m_viscosity;
        float m_depth;
        ContourMode m_contourMode;
        MinMax<float> m_contourMinMax;
        bool m_paused;
        SurfaceShadingMode m_shadingMode;
        bool m_rayTracingPaused;
        bool m_floorWireframeVisible;
        bool m_diagEnabled;
        bool m_debug;

        TimeHistory m_history;

        bool m_firstUIDraw;

		bool m_imguiHandlingMouseEvent;

        std::shared_ptr<Shizuku::Flow::Flow> m_flow;
    public:
        Window();
        void SetGraphics(std::shared_ptr<Shizuku::Flow::Flow> flow);
        void RegisterCommands();
        void RegisterGlfwInputs();
        void Resize(const Rect<int>& size);
        void Display();
        void InitializeGlfw();
        void InitializeImGui();
        void ApplyInitialFlowSettings();
        void EnableDebug();
        void EnableDiagnostics();

        void Resize(GLFWwindow* window, int width, int height);
        void MouseButton(const int button, const int state, const int mod);
        void MouseMotion(const int x, const int y);
        void MouseWheel(double xwheel, double ywheel);
        void Keyboard(int key, int scancode, int action, int mode);
        void UpdateWindowTitle(const float fps, const Rect<int> &domainSize, const int tSteps);

        static Window& Instance()
        {
            static Window s_window = Window();
            return s_window;
        }


    private:
        void Draw3D();
        void DrawUI();

        void TogglePaused();
        void SetPaused(const bool p_paused);
        void SetRayTracingPaused(const bool p_paused);
    };
}}