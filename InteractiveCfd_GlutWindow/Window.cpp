#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_glfw.h"
#include "../imgui/imgui_impl_opengl3.h"
#include "Window.h"
#include "Graphics/GraphicsManager.h"
#include "Graphics/CudaLbm.h"
#include "../Shizuku.Core/Ogl/Ogl.h"
#include "../Shizuku.Core/Ogl/Shader.h"


#include <GLFW/glfw3.h>
#include <typeinfo>
#include <memory>


namespace
{
    void ResizeWrapper(GLFWwindow* window, int width, int height)
    {
        Window::Instance().Resize(Rect<int>(width, height));
    }

    void MouseButtonWrapper(GLFWwindow* window, int button, int state, int mods)
    {
        Window::Instance().GlfwMouseButton(button, state, mods);
    }

    void MouseMotionWrapper(GLFWwindow* window, double x, double y)
    {
        Window::Instance().MouseMotion(x, y);
    }

    void MouseWheelWrapper(GLFWwindow* window, double xwheel, double ywheel)
    {
        Window::Instance().GlfwMouseWheel(xwheel, ywheel);
    }

    void KeyboardWrapper(GLFWwindow* window, int key, int scancode, int action, int mode)
    {
        Window::Instance().GlfwKeyboard(key, scancode, action, mode);
    }

    void GLAPIENTRY MessageCallback( GLenum source,
                     GLenum type,
                     GLuint id,
                     GLenum severity,
                     GLsizei length,
                     const GLchar* message,
                     const void* userParam )
    {
      printf( "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
               ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
                type, severity, message );
    }
}

Window::Window() : 
    m_zoom(Zoom(*m_graphics)),
    m_pan(Pan(*m_graphics)),
    m_rotate(Rotate(*m_graphics)),
    m_addObstruction(AddObstruction(*m_graphics)),
    m_removeObstruction(RemoveObstruction(*m_graphics)),
    m_moveObstruction(MoveObstruction(*m_graphics)),
    m_pauseSimulation(PauseSimulation(*m_graphics))
{
}

void Window::SetGraphicsManager(GraphicsManager &graphics)
{
    m_graphics = &graphics;
}

void Window::RegisterCommands()
{
    m_zoom = Zoom(*m_graphics);
    m_pan = Pan(*m_graphics);
    m_rotate = Rotate(*m_graphics);
    m_addObstruction = AddObstruction(*m_graphics);
    m_removeObstruction = RemoveObstruction(*m_graphics);
    m_moveObstruction = MoveObstruction(*m_graphics);
    m_pauseSimulation = PauseSimulation(*m_graphics);
}

float Window::GetFloatCoordX(const int x)
{
    return static_cast<float>(x)/m_graphics->GetViewport().Width*2.f - 1.f;
}
float Window::GetFloatCoordY(const int y)
{
    return static_cast<float>(y)/m_graphics->GetViewport().Height*2.f - 1.f;
}

void Window::InitializeGL()
{
    glEnable(GL_LIGHT0);
    glewInit();
    glViewport(0,0,m_size.Width,m_size.Height);
}

void Window::GlfwResize(GLFWwindow* window, int width, int height)
{
    Resize(Rect<int>(width, height));
}

void Window::Resize(Rect<int> size)
{
    m_size = size;
    m_graphics->SetViewport(size);
}

void Window::GlfwMouseButton(const int button, const int state, const int mod)
{
    double x, y;
    glfwGetCursorPos(m_window, &x, &y);

    float xf = GetFloatCoordX(x);
    float yf = GetFloatCoordY(m_size.Height - y);
    if (button == GLFW_MOUSE_BUTTON_MIDDLE && state == GLFW_PRESS
        && mod == GLFW_MOD_CONTROL)
    {
        m_pan.Start(xf, yf);
    }
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE && state == GLFW_PRESS)
    {
        m_rotate.Start(xf, yf);
        m_removeObstruction.Start(xf, yf);
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && state == GLFW_PRESS)
    {
        m_addObstruction.Start(xf, yf);
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_PRESS)
    {
        m_moveObstruction.Start(xf, yf);
    }
    else
    {
        m_pan.End();
        m_rotate.End();
        m_moveObstruction.End();
        m_removeObstruction.End(xf, yf);
    }
}

void Window::MouseMotion(const int x, const int y)
{
    float xf = GetFloatCoordX(x);
    float yf = GetFloatCoordY(m_size.Height - y);
    m_pan.Track(xf, yf);
    m_rotate.Track(xf, yf);
    m_moveObstruction.Track(xf, yf);
}

void Window::GlfwMouseWheel(double xwheel, double ywheel)
{
    double x, y;
    glfwGetCursorPos(m_window, &x, &y);
    const int dir = ywheel > 0 ? 1 : 0;
    m_zoom.Start(dir, 0.3f);
}

void Window::GlfwKeyboard(int key, int scancode, int action, int mode)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        if (m_graphics->GetCudaLbm()->IsPaused())
        {
            m_pauseSimulation.End();
        }
        else
        {
            m_pauseSimulation.Start();
        }
    }
}

void Window::GlfwUpdateWindowTitle(const float fps, const Rect<int> &domainSize, const int tSteps)
{
    char fpsReport[256];
    int xDim = domainSize.Width;
    int yDim = domainSize.Height;
    sprintf_s(fpsReport, 
        "Shizuku Flow running at: %i timesteps/frame at %3.1f fps = %3.1f timesteps/second on %ix%i mesh",
        tSteps, fps, TIMESTEPS_PER_FRAME*fps, xDim, yDim);
    glfwSetWindowTitle(m_window, fpsReport);
}

void Window::GlfwDrawLoop()
{
    m_fpsTracker.Tick();
    GraphicsManager* graphicsManager = m_graphics;
    graphicsManager->UpdateGraphicsInputs();
//    graphicsManager->GetCudaLbm()->UpdateDeviceImage();

//    graphicsManager->RunSimulation();

//    // render caustic floor to texture
//    graphicsManager->RenderFloorToTexture();

//    graphicsManager->RunSurfaceRefraction();

    //int viewX, viewY;
    //m_graphics->GetViewport(viewX, viewY);
    //ResizeWrapper(m_window, viewX, viewY);

//    graphicsManager->UpdateViewMatrices();
//    graphicsManager->UpdateViewTransformations();


    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    graphicsManager->RenderVbo();

    m_fpsTracker.Tock();

    CudaLbm* cudaLbm = graphicsManager->GetCudaLbm();
    const int tStepsPerFrame = graphicsManager->GetCudaLbm()->GetTimeStepsPerFrame();
    GlfwUpdateWindowTitle(m_fpsTracker.GetFps(), cudaLbm->GetDomainSize(), tStepsPerFrame);

}

void Window::InitializeGlfw()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(m_size.Width, m_size.Height, "Shizuku", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    m_window = window;

    glfwSetKeyCallback(window, KeyboardWrapper);
    glfwSetWindowSizeCallback(window, ResizeWrapper); //glfw is in C so cannot bind instance method...
    glfwSetCursorPosCallback(window, MouseMotionWrapper);
    glfwSetMouseButtonCallback(window, MouseButtonWrapper);
    glfwSetScrollCallback(window, MouseWheelWrapper);

    glewExperimental = GL_TRUE;
    glewInit();

    glEnable              ( GL_DEBUG_OUTPUT );
    glDebugMessageCallback(MessageCallback, 0);
}

void Window::GlfwDisplay()
{
//    // Setup Dear ImGui binding
//    IMGUI_CHECKVERSION();
//    ImGui::CreateContext();
//    ImGuiIO& io = ImGui::GetIO(); (void)io;
//    const char* glsl_version = "#version 430 core";
//    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
//    ImGui_ImplOpenGL3_Init(glsl_version);
//    ImGui::StyleColorsDark();
    while (!glfwWindowShouldClose(m_window))
    {
        glfwPollEvents();
        GlfwDrawLoop();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        bool test;
        ImGui::ShowDemoWindow(&test);
        glfwSwapBuffers(m_window);
    }
    glfwTerminate();
}


