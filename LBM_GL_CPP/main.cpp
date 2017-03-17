#include "Layout.h"
#include "Panel.h"
#include "Mouse.h"
#include "kernel.h"
#include "FpsTracker.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_cuda_gl.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GLEW/glew.h>
#include <GLUT/freeglut.h>
#include <ostream>
#include <fstream>
#include <time.h>
#include <algorithm>

extern const int g_leftPanelWidth(350);
extern const int g_leftPanelHeight(500);

FpsTracker g_fpsTracker;

Panel Window;
Mouse theMouse;

const int g_glutMouseYOffset = 10; //hack to get better mouse precision

void Init()
{
    glEnable(GL_LIGHT0);
    glewInit();
    int windowWidth = Window.GetWidth();
    int windowHeight = Window.GetHeight();
    glViewport(0,0,windowWidth,windowHeight);

}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

/*----------------------------------------------------------------------------------------
 *	Mouse interactions
 */

void MouseButton(int button, int state, int x, int y)
{
    theMouse.Click(x, theMouse.m_winH-y-g_glutMouseYOffset, button, state);
}

void MouseMotion(int x, int y)
{
    int windowWidth = Window.GetWidth();
    int windowHeight = Window.GetHeight();
    if (x >= 0 && x <= windowWidth && y>=0 && y<=windowHeight)
    {
        theMouse.Move(x, theMouse.m_winH-y-g_glutMouseYOffset);
    }
}

void MouseWheel(int button, int dir, int x, int y)
{
    theMouse.Wheel(button, dir, x, y);
}

void Keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
    case (' ') :
        GraphicsManager* graphicsManager = Window.GetPanel("Graphics")->GetGraphicsManager();
        CudaLbm* cudaLbm = graphicsManager->GetCudaLbm();
        bool currentPausedState = cudaLbm->IsPaused();
        cudaLbm->SetPausedState(!currentPausedState);
        break;
    }
}

void Resize(int windowWidth, int windowHeight)
{
    Layout::UpdateDomainDimensionsBasedOnWindowSize(Window, g_leftPanelHeight, g_leftPanelWidth);

    theMouse.m_winW = windowWidth;
    theMouse.m_winH = windowHeight;

    RectInt rect = { 200, 100, windowWidth, windowHeight };
    Window.SetSize_Absolute(rect);
    rect = { 0, windowHeight - g_leftPanelHeight, g_leftPanelWidth, g_leftPanelHeight };
    Window.GetPanel("CDV")->SetSize_Absolute(rect);
    rect = { g_leftPanelWidth, 0, windowWidth - g_leftPanelWidth, windowHeight };
    Window.GetPanel("Graphics")->SetSize_Absolute(rect);
    Window.UpdateAll();

    glViewport(0, 0, windowWidth, windowHeight);

    //UpdateDeviceImage();

    GraphicsManager *graphicsManager = Window.GetPanel("Graphics")->GetGraphicsManager();
    graphicsManager->GetCudaLbm()->UpdateDeviceImage();

}

void UpdateLbmInputs(CudaLbm &cudaLbm, Panel &rootPanel)
{
    //get simulation inputs
    float u = rootPanel.GetSlider("Slider_InletV")->m_sliderBar1->GetValue();
    float omega = rootPanel.GetSlider("Slider_Visc")->m_sliderBar1->GetValue();
    cudaLbm.SetInletVelocity(u);
    cudaLbm.SetOmega(omega);
}

void CheckGLError()
{
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cout << "OpenGL error: " << err << std::endl;
    }
}

void UpdateWindowTitle(const float fps, Domain &domain)
{
    char fpsReport[256];
    int xDim = domain.GetXDim();
    int yDim = domain.GetYDim();
    sprintf(fpsReport, 
        "Interactive CFD running at: %i timesteps/frame at %3.1f fps = %3.1f timesteps/second on %ix%i mesh",
        TIMESTEPS_PER_FRAME, fps, TIMESTEPS_PER_FRAME*fps, xDim, yDim);
    glutSetWindowTitle(fpsReport);
}

void Draw()
{
    g_fpsTracker.Tick();
    
    GraphicsManager* graphicsManager = Window.GetPanel("Graphics")->GetGraphicsManager();
    CudaLbm* cudaLbm = graphicsManager->GetCudaLbm();
    graphicsManager->UpdateGraphicsInputs();

    Resize(Window.GetWidth(), Window.GetHeight());

    graphicsManager->CenterGraphicsViewToGraphicsPanel(g_leftPanelWidth);
    graphicsManager->UpdateViewTransformations();

    graphicsManager->GetCudaLbm()->UpdateDeviceImage();
    graphicsManager->RunCuda();
    graphicsManager->RunComputeShader();
    graphicsManager->RenderVboUsingShaders();

    CheckGLError();

    Layout::Draw2D(Window);

    glutSwapBuffers();

    //Compute and display FPS
    g_fpsTracker.Tock();
    float fps = g_fpsTracker.GetFps();
    Domain domain = *cudaLbm->GetDomain();
    UpdateWindowTitle(fps, domain);
}

int main(int argc,char **argv)
{
    Layout::SetUpWindow(Window);
    theMouse.SetBasePanel(&Window);

    glutInit(&argc,argv);

    glutInitDisplayMode(GLUT_RGB|GLUT_DEPTH|GLUT_DOUBLE);
    int windowWidth = Window.GetWidth();
    int windowHeight = Window.GetHeight();
    glutInitWindowSize(windowWidth,windowHeight);
    glutInitWindowPosition(200,100);
    glutCreateWindow("Interactive CFD");

    glutDisplayFunc(Draw);
    glutReshapeFunc(Resize);
    glutMouseFunc(MouseButton);
    glutMotionFunc(MouseMotion);
    glutKeyboardFunc(Keyboard);
    glutMouseWheelFunc(MouseWheel);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    Init();
    GraphicsManager* graphicsManager = Window.GetPanel("Graphics")->GetGraphicsManager();
    graphicsManager->SetUpGLInterop();
    graphicsManager->SetUpCuda();
    graphicsManager->SetUpShaders();

    glutMainLoop();

    return 0;
}