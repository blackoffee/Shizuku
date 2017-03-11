#pragma once
#include <typeinfo>
#include "main.h"
#include "command.h"

class Window
{
private:
    static Panel* m_windowPanel;
    static Panel* m_currentPanel;
    static int m_previousMouseX;
    static int m_previousMouseY;
    static int m_currentMouseButton;
    static Zoom m_zoom;
    static Pan m_pan;
    static Rotate m_rotate;
    static ButtonPress m_buttonPress;
    static SliderDrag m_sliderDrag;
    static AddObstruction m_addObstruction;
    static RemoveObstruction m_removeObstruction;
    static MoveObstruction m_moveObstruction;
    static int m_leftPanelWidth;
    static int m_leftPanelHeight;
public:
    Window();
    Window(const int width, const int height);

    Panel* GetWindowPanel();
    static float GetFloatCoordX(const int x);
    static float GetFloatCoordY(const int y);
    void InitializeGL();
    //static void timerEvent(int value);
    void Resize(const int width, const int height);

    void MouseButton(const int button, const int state,
        const int x, const int y);

    void MouseMotion(const int x, const int y);
    void Keyboard(const unsigned char key,
        const int /*x*/, const int /*y*/);
    void MouseWheel(const int button, const int direction,
        const int x, const int y);

    void DrawLoop();
    void InitializeGLUT(int argc, char **argv);
    void Display();

    static Window& Instance()
    {
        static Window s_window = Window(1200,300);
        return s_window;
    }
};


void ResizeWrapper(const int x, const int y);
void MouseButtonWrapper(const int button, const int state, const int x, const int y);
void MouseMotionWrapper(const int x, const int y);
void MouseWheelWrapper(const int button, const int direction, const int x, const int y);
void KeyboardWrapper(const unsigned char key, const int x, const int y);

void DrawLoopWrapper();