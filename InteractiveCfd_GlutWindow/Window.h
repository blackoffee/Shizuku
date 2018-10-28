#pragma once
#include "Command/Zoom.h"
#include "Command/Pan.h"
#include "Command/Rotate.h"
#include "Command/AddObstruction.h"
#include "Command/RemoveObstruction.h"
#include "Command/MoveObstruction.h"
#include "Command/PauseSimulation.h"
#include "FpsTracker.h"

class Domain;
class GLFWwindow;
class GraphicsManager;

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
    int m_leftPanelWidth;
    int m_leftPanelHeight;
    FpsTracker m_fpsTracker;
    GLFWwindow* m_window;

    GraphicsManager* m_graphics;
public:
    Window();
    Window(GraphicsManager& graphics);
    void SetGraphicsManager(GraphicsManager& graphics);
    void RegisterCommands();
    float GetFloatCoordX(const int x);
    float GetFloatCoordY(const int y);
    void InitializeGL();
    void GlfwResize(GLFWwindow* window, int width, int height);
    void Resize(const int width, const int height);
    void MouseButton(const int button, const int state,
        const int x, const int y);
    void GlfwMouseButton(const int button, const int state, const int mod);
    void MouseMotion(const int x, const int y);
    void Keyboard(const unsigned char key,
        const int /*x*/, const int /*y*/);
    void MouseWheel(const int button, const int direction,
        const int x, const int y);
    void GlfwMouseWheel(double xwheel, double ywheel);
    void UpdateWindowTitle(const float fps, Domain &domain, const int tSteps);
    void GlfwUpdateWindowTitle(const float fps, Domain &domain, const int tSteps);
    void DrawLoop();
    void GlfwDrawLoop();
    void InitializeGLUT(int argc, char **argv);
    void InitializeGlfw(int argc, char **argv);
    void Display();
    void GlfwDisplay();

    static Window& Instance()
    {
        static Window s_window = Window();
        return s_window;
    }
};