#pragma once
#include <typeinfo>
#include "main.h"
#include "command.h"

class Window
{
private:
    Panel* m_windowPanel;
    Panel* m_currentPanel;
    int m_previousMouseX;
    int m_previousMouseY;
    int m_currentMouseButton;
    Zoom m_zoom;
    Pan m_pan;
    Rotate m_rotate;
    ButtonPress m_buttonPress;
    SliderDrag m_sliderDrag;
    AddObstruction m_addObstruction;
    RemoveObstruction m_removeObstruction;
    MoveObstruction m_moveObstruction;
    int m_leftPanelWidth;
    int m_leftPanelHeight;
public:
    Window();
    Panel* GetWindowPanel();
    float GetFloatCoordX(const int x);
    float GetFloatCoordY(const int y);
    void InitializeGL();
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
        static Window s_window = Window();
        return s_window;
    }
};

void ResizeWrapper(const int x, const int y);
void MouseButtonWrapper(const int button, const int state, const int x, const int y);
void MouseMotionWrapper(const int x, const int y);
void MouseWheelWrapper(const int button, const int direction, const int x, const int y);
void KeyboardWrapper(const unsigned char key, const int x, const int y);
void DrawLoopWrapper();