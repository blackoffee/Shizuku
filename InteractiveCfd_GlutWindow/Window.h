#pragma once
#include "Command/Zoom.h"
#include "Command/Pan.h"
#include "Command/Rotate.h"
#include "Command/AddObstruction.h"
#include "Command/RemoveObstruction.h"
#include "Command/MoveObstruction.h"
#include "Command/PauseSimulation.h"
#include "Command/SetSimulationScale.h"
#include "Command/SetTimestepsPerFrame.h"
#include "FpsTracker.h"
#include "Shizuku.Core/Rect.h"

class GLFWwindow;
class GraphicsManager;

using namespace Shizuku::Core;

class Window
{
private:
    Zoom m_zoom;
    Pan m_pan;
    Rotate m_rotate;
    AddObstruction m_addObstruction;
    RemoveObstruction m_removeObstruction;
    MoveObstruction m_moveObstruction;
    PauseSimulation m_pauseSimulation;
    SetSimulationScale m_setSimulationScale;
    SetTimestepsPerFrame m_timestepsPerFrame;
    int m_leftPanelWidth;
    int m_leftPanelHeight;
    FpsTracker m_fpsTracker;
    GLFWwindow* m_window;
    Rect<int> m_size;

    float m_simulationScale;
    int m_timesteps;

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