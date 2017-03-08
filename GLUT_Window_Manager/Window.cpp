#include "Window.h"

int Window::m_leftPanelWidth(350);
int Window::m_leftPanelHeight(500);
Panel* Window::m_currentPanel = NULL;
int Window::m_previousMouseX = 0;
int Window::m_previousMouseY = 0;
int Window::m_currentMouseButton = 0;
Panel* Window::m_windowPanel = new Panel;
Zoom Window::m_zoom = Zoom();
Pan Window::m_pan = Pan();
Rotate Window::m_rotate = Rotate();
ButtonPress Window::m_buttonPress = ButtonPress();
SliderDrag Window::m_sliderDrag = SliderDrag();

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
    UpdateDomainDimensionsBasedOnWindowSize(*m_windowPanel, m_leftPanelHeight, m_leftPanelWidth);

    RectInt rect = { 200, 100, width, height };
    m_windowPanel->SetSize_Absolute(rect);
    rect = { 0, windowHeight - m_leftPanelHeight, m_leftPanelWidth, m_leftPanelHeight };
    m_windowPanel->GetPanel("CDV")->SetSize_Absolute(rect);
    rect = { m_leftPanelWidth, 0, windowWidth - m_leftPanelWidth, windowHeight };
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
    if (m_currentPanel == NULL)
    {
        return;
    }
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
            m_sliderDrag.End();
        }
    }
    else
    {
        std::string panelType = typeid(*m_currentPanel).name();
        if (panelType == "class Button")
        {
            Button* buttonPanel = dynamic_cast<Button*>(m_currentPanel);
            if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
            {
                m_buttonPress.Start(buttonPanel);
            }
            else if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
            {
                m_buttonPress.End(buttonPanel);
            }
        }
        else if (panelType == "class SliderBar")
        {
            SliderBar* sliderBar = dynamic_cast<SliderBar*>(m_currentPanel);
            if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
            {
                m_sliderDrag.Start(sliderBar, xf, yf);
            }
            else if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
            {
                m_sliderDrag.End();
            }
        }
    }

}
void Window::MouseMotion(const int x, const int y)
{
    int windowHeight = m_windowPanel->GetHeight();
    float xf = GetFloatCoordX(x);
    float yf = GetFloatCoordY(windowHeight-y);
    if (m_currentPanel == NULL)
    {
        return;
    }
    if (m_currentPanel->GetGraphicsManager() != NULL)
    {
        m_pan.Track(xf, yf);
        m_rotate.Track(xf, yf);
    }
    else
    {
        m_sliderDrag.Track(xf, yf);
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
        m_zoom.Start(direction, 0.3f);
    }
}

void Window::DrawLoop()
{
    GraphicsManager* graphicsManager = m_windowPanel->GetPanel("Graphics")->GetGraphicsManager();
    CudaLbm* cudaLbm = graphicsManager->GetCudaLbm();
    graphicsManager->UpdateViewTransformations();
    graphicsManager->UpdateGraphicsInputs();

    Resize(m_windowPanel->GetWidth(), m_windowPanel->GetHeight());

    graphicsManager->CenterGraphicsViewToGraphicsPanel(m_leftPanelWidth);

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


