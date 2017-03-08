#include <GLEW/glew.h>
#include <GLUT/freeglut.h>

#include <stdio.h>
#include <iostream>
#include <ostream>
#include <fstream>
#include <time.h>
#include <algorithm>

#include "main.h"

const int g_leftPanelWidth(350);
const int g_leftPanelHeight(500);


class Command
{
    Panel* m_rootPanel;
public:
    enum State {ACTIVE, UNACTIVE};
    State m_state;
    Command();
    void Start();
    void Track();
    void End();
    void Initialize(Panel &rootPanel);
    Panel* GetRootPanel();
    GraphicsManager* GetGraphicsManager();
};

Command::Command()
{
}

void Command::Initialize(Panel &rootPanel)
{
    m_rootPanel = &rootPanel;
}

Panel* Command::GetRootPanel()
{
    return m_rootPanel;
}

GraphicsManager* Command::GetGraphicsManager()
{
    return m_rootPanel->GetPanel("Graphics")->GetGraphicsManager();
}


class Window;

class Zoom : public Command
{
public:
    Zoom();
    void Start(Panel &rootPanel, const int dir, const float mag);
};

Zoom::Zoom()
{
}

void Zoom::Start(Panel &rootPanel, const int dir, const float mag)
{
    GetGraphicsManager()->Zoom(dir, mag);
}


class Pan : public Command
{
    float m_initialX;
    float m_initialY;
public:
    Pan();
    void Start(const float initialX, const float initialY);
    void Track(const float currentX, const float currentY);
    void End();
};

Pan::Pan()
{
    m_state = UNACTIVE;
}

void Pan::Start(const float initialX, const float initialY)
{
    m_state = ACTIVE;
    m_initialX = initialX;
    m_initialY = initialY;
}

void Pan::Track(const float currentX, const float currentY)
{
    float dx = currentX - m_initialX;
    float dy = currentY - m_initialY;
    if (m_state == ACTIVE)
    {
        GetGraphicsManager()->Pan(dx, dy);
    }
    m_initialX = currentX;
    m_initialY = currentY;
}

void Pan::End()
{
    m_state = UNACTIVE;
}

class Rotate : public Command
{
    float m_initialX;
    float m_initialY;
public:
    Rotate();
    void Start(const float initialX, const float initialY);
    void Track(const float currentX, const float currentY);
    void End();
};

Rotate::Rotate()
{
    m_state = UNACTIVE;
}

void Rotate::Start(const float initialX, const float initialY)
{
    m_state = ACTIVE;
    m_initialX = initialX;
    m_initialY = initialY;
}

void Rotate::Track(const float currentX, const float currentY)
{
    float dx = (currentX - m_initialX)*45.f;
    float dy = (currentY - m_initialY)*45.f;
    if (m_state == ACTIVE)
    {
        GetGraphicsManager()->Rotate(dx, dy);
    }
    m_initialX = currentX;
    m_initialY = currentY;
}

void Rotate::End()
{
    m_state = UNACTIVE;
}


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
public:
    Window();
    Window(const int width, const int height);

    Panel* GetWindowPanel();
    static float GetFloatCoordX(const int x);
    static float GetFloatCoordY(const int y);
    void InitializeGL();
    static void timerEvent(int value);
    static void Resize(const int width, const int height);

    static void MouseButton(const int button, const int state,
        const int x, const int y);

    static void MouseMotion(const int x, const int y);
    static void Keyboard(const unsigned char key,
        const int /*x*/, const int /*y*/);
    static void MouseWheel(const int button, const int direction,
        const int x, const int y);

    static void DrawLoop();
    void InitializeGLUT(int argc, char **argv);
    void Display();
};

Panel* Window::m_currentPanel = NULL;
int Window::m_previousMouseX = 0;
int Window::m_previousMouseY = 0;
int Window::m_currentMouseButton = 0;
Panel* Window::m_windowPanel = new Panel;
Zoom Window::m_zoom = Zoom();
Pan Window::m_pan = Pan();
Rotate Window::m_rotate = Rotate();



Window::Window()
{
    m_zoom.Initialize(*m_windowPanel);
    m_pan.Initialize(*m_windowPanel);
    m_rotate.Initialize(*m_windowPanel);
}
Window::Window(const int width, const int height)
{
    m_zoom.Initialize(*m_windowPanel);
    m_pan.Initialize(*m_windowPanel);
    m_rotate.Initialize(*m_windowPanel);
}


Panel* Window::GetWindowPanel()
{
    return m_windowPanel;
}

float Window::GetFloatCoordX(const int x)
{
    int width = m_windowPanel->GetWidth();
    return static_cast<float>(x)/width*2.f - 1.f;
}
float Window::GetFloatCoordY(const int y)
{
    int height = m_windowPanel->GetHeight();
    return static_cast<float>(y)/height*2.f - 1.f;
}

void Window::InitializeGL()
{
    glEnable(GL_LIGHT0);
    glewInit();
    glViewport(0,0,800,600);
}

void Window::timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

void Window::Resize(const int width, const int height)
{
    float scaleUp = m_windowPanel->GetPanel("Graphics")->GetGraphicsManager()->GetScaleFactor();
    int windowWidth = m_windowPanel->GetWidth();
    int windowHeight = m_windowPanel->GetHeight();
    UpdateDomainDimensionsBasedOnWindowSize(*m_windowPanel, g_leftPanelHeight, g_leftPanelWidth);

    RectInt rect = { 200, 100, width, height };
    m_windowPanel->SetSize_Absolute(rect);
    rect = { 0, windowHeight - g_leftPanelHeight, g_leftPanelWidth, g_leftPanelHeight };
    m_windowPanel->GetPanel("CDV")->SetSize_Absolute(rect);
    rect = { g_leftPanelWidth, 0, windowWidth - g_leftPanelWidth, windowHeight };
    m_windowPanel->GetPanel("Graphics")->SetSize_Absolute(rect);
    m_windowPanel->UpdateAll();

    glViewport(0, 0, windowWidth, windowHeight);
}
void Window::MouseButton(const int button, const int state,
    const int x, const int y)
{
    int windowHeight = m_windowPanel->GetHeight();
    float xf = GetFloatCoordX(x);
    float yf = GetFloatCoordY(windowHeight-y);
    m_currentPanel = GetPanelThatPointIsIn(m_windowPanel, xf, yf);
    int mod = glutGetModifiers();
    if (m_currentPanel != NULL)
    {
        if (m_currentPanel->GetGraphicsManager() != NULL)
        {
            if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN
                && mod == GLUT_ACTIVE_CTRL)
            {
                m_pan.Start(xf, yf);
            }
            else if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN)
            {
                m_rotate.Start(xf, yf);
            }
            else
            {
                m_pan.End();
                m_rotate.End();
            }
        }
    }

}
void Window::MouseMotion(const int x, const int y)
{
    int windowHeight = m_windowPanel->GetHeight();
    float xf = GetFloatCoordX(x);
    float yf = GetFloatCoordY(windowHeight-y);

    if (m_currentPanel != NULL)
    {
        if (m_currentPanel->GetGraphicsManager() != NULL)
        {
            m_pan.Track(xf, yf);
            m_rotate.Track(xf, yf);
        }
    }


}
void Window::Keyboard(const unsigned char key,
    const int /*x*/, const int /*y*/)
{
    
}
void Window::MouseWheel(const int button, const int direction,
    const int x, const int y)
{
    float xf = intCoordToFloatCoord(x, m_windowPanel->GetWidth());
    float yf = intCoordToFloatCoord(y, m_windowPanel->GetHeight());
    Panel* panel = GetPanelThatPointIsIn(m_windowPanel, xf, yf);
    if (panel->GetGraphicsManager() != NULL)
    {
        m_zoom.Start(*m_windowPanel, direction, 0.3f);
    }
}

void Window::DrawLoop()
{
    GraphicsManager* graphicsManager = m_windowPanel->GetPanel("Graphics")->GetGraphicsManager();
    CudaLbm* cudaLbm = graphicsManager->GetCudaLbm();
    graphicsManager->UpdateViewTransformations();
    graphicsManager->UpdateGraphicsInputs();

    Resize(m_windowPanel->GetWidth(), m_windowPanel->GetHeight());

    graphicsManager->CenterGraphicsViewToGraphicsPanel(g_leftPanelWidth);

    graphicsManager->GetCudaLbm()->UpdateDeviceImage();
    graphicsManager->RunCuda();

    bool renderFloor = graphicsManager->ShouldRenderFloor();
    Graphics* graphics = graphicsManager->GetGraphics();
    Domain domain = *cudaLbm->GetDomain();
    graphics->RenderVbo(renderFloor, domain);

    Draw2D(*m_windowPanel);

    glutSwapBuffers();
}

void Window::InitializeGLUT(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB|GLUT_DEPTH|GLUT_DOUBLE);
    int width = m_windowPanel->GetWidth();
    int height = m_windowPanel->GetHeight();
    glutInitWindowSize(width,height);
    glutInitWindowPosition(50,30);

    glutCreateWindow("New Window management");

    glutReshapeFunc(&Window::Resize);
    glutMouseFunc(&Window::MouseButton);
    glutMotionFunc(&Window::MouseMotion);
    glutKeyboardFunc(&Window::Keyboard);
    glutMouseWheelFunc(&Window::MouseWheel);

    glutDisplayFunc(&Window::DrawLoop);
    glutTimerFunc(REFRESH_DELAY, &Window::timerEvent, 0);

}

void Window::Display()
{
    glutMainLoop();
}




int main(int argc, char **argv)
{
    int initialWindowWidth = 1200;
    int initialWindowHeight = g_leftPanelHeight+100;
    Window window(initialWindowWidth,initialWindowHeight);
    Panel* windowPanel = window.GetWindowPanel();

    SetUpWindow(*windowPanel);

    window.InitializeGLUT(argc, argv);
    window.InitializeGL();

    GraphicsManager* graphicsManager = window.GetWindowPanel()
        ->GetPanel("Graphics")->GetGraphicsManager();
    Graphics* graphics = graphicsManager->GetGraphics();
    SetUpGLInterop(*windowPanel);
    SetUpCUDA(*windowPanel);

    window.Display();


    return 0;
}