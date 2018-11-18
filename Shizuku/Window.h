#pragma once
#include "Shizuku.Core/Utilities/FpsTracker.h"
#include "Shizuku.Core/Rect.h"
#include <memory>

class GLFWwindow;
class GraphicsManager;
class Zoom;
class Pan;
class Rotate;
class AddObstruction;
class RemoveObstruction;
class MoveObstruction;
class PauseSimulation; 
class SetSimulationScale;
class SetTimestepsPerFrame;
class SetInletVelocity;
class SetContourMode;

using namespace Shizuku::Core;

enum ContourMode;

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
    std::shared_ptr<SetSimulationScale> m_setSimulationScale;
    std::shared_ptr<SetTimestepsPerFrame> m_timestepsPerFrame;
    std::shared_ptr<SetInletVelocity> m_setVelocity;
    std::shared_ptr<SetContourMode> m_setContourMode;
    FpsTracker m_fpsTracker;
    GLFWwindow* m_window;
    Rect<int> m_size;

    float m_simulationScale;
    int m_timesteps;
    float m_velocity;
    float m_viscosity;
    ContourMode m_contourMode;

    bool m_firstUIDraw;

    GraphicsManager* m_graphics;
public:
    Window();
    Window(GraphicsManager& graphics);
    void SetGraphicsManager(GraphicsManager& graphics);
    void RegisterCommands();
    void RegisterGlfwInputs();
    float GetFloatCoordX(const int x);
    float GetFloatCoordY(const int y);
    void InitializeGL();
    void GlfwResize(GLFWwindow* window, int width, int height);
    void Resize(Rect<int> size);
    void GlfwMouseButton(const int button, const int state, const int mod);
    void MouseMotion(const int x, const int y);
    void GlfwMouseWheel(double xwheel, double ywheel);
    void GlfwKeyboard(int key, int scancode, int action, int mode);
    void GlfwUpdateWindowTitle(const float fps, const Rect<int> &domainSize, const int tSteps);
    void Draw3D();
    void InitializeGlfw();
    void InitializeImGui();
    void GlfwDisplay();
    void DrawUI();

    static Window& Instance()
    {
        static Window s_window = Window();
        return s_window;
    }

private:
    void TogglePaused();
};