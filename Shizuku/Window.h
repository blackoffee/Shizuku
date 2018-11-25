#pragma once
#include "Shizuku.Core/Utilities/FpsTracker.h"
#include "Shizuku.Core/Rect.h"
#include "Shizuku.Core/Types/MinMax.h"
#include <memory>

class GLFWwindow;
class Zoom;
class Pan;
class Rotate;
class AddObstruction;
class RemoveObstruction;
class MoveObstruction;
class PauseSimulation; 
class PauseRayTracing;
class SetSimulationScale;
class SetTimestepsPerFrame;
class SetInletVelocity;
class SetContourMode;
class SetContourMinMax;
class SetSurfaceShadingMode;

namespace Shizuku{
    namespace Flow{
        class Flow;
    }
}

enum ContourMode;
enum SurfaceShadingMode;

using namespace Shizuku::Core;

class Window
{
private:
    std::shared_ptr<Zoom> m_zoom;
    std::shared_ptr<Pan> m_pan;
    std::shared_ptr<Rotate> m_rotate;
    std::shared_ptr<AddObstruction> m_addObstruction;
    std::shared_ptr<RemoveObstruction> m_removeObstruction;
    std::shared_ptr<MoveObstruction> m_moveObstruction;
    std::shared_ptr<PauseSimulation> m_pauseSimulation;
    std::shared_ptr<PauseRayTracing> m_pauseRayTracing;
    std::shared_ptr<SetSimulationScale> m_setSimulationScale;
    std::shared_ptr<SetTimestepsPerFrame> m_timestepsPerFrame;
    std::shared_ptr<SetInletVelocity> m_setVelocity;
    std::shared_ptr<SetContourMode> m_setContourMode;
    std::shared_ptr<SetContourMinMax> m_setContourMinMax;
    std::shared_ptr<SetSurfaceShadingMode> m_setSurfaceShadingMode;
    FpsTracker m_fpsTracker;
    GLFWwindow* m_window;
    Rect<int> m_size;

    float m_simulationScale;
    int m_timesteps;
    float m_velocity;
    float m_viscosity;
    ContourMode m_contourMode;
    MinMax<float> m_contourMinMax;
    bool m_paused;
    SurfaceShadingMode m_shadingMode;
    bool m_rayTracingPaused;

    bool m_firstUIDraw;

    std::shared_ptr<Shizuku::Flow::Flow> m_flow;
public:
    Window();
    void SetGraphics(std::shared_ptr<Shizuku::Flow::Flow> flow);
    void RegisterCommands();
    void RegisterGlfwInputs();
    void Resize(Rect<int> size);
    void Display();
    void InitializeGlfw();
    void InitializeImGui();

    void GlfwResize(GLFWwindow* window, int width, int height);
    void GlfwMouseButton(const int button, const int state, const int mod);
    void MouseMotion(const int x, const int y);
    void GlfwMouseWheel(double xwheel, double ywheel);
    void GlfwKeyboard(int key, int scancode, int action, int mode);
    void GlfwUpdateWindowTitle(const float fps, const Rect<int> &domainSize, const int tSteps);

    static Window& Instance()
    {
        static Window s_window = Window();
        return s_window;
    }

private:
    float GetFloatCoordX(const int x);
    float GetFloatCoordY(const int y);

    void Draw3D();
    void DrawUI();

    void TogglePaused();
    void SetPaused(const bool p_paused);
    void SetRayTracingPaused(const bool p_paused);
};