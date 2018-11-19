#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_glfw.h"
#include "Window.h"
#include "../imgui/imgui_impl_opengl3.h"

#include "Shizuku.Flow/Graphics/GraphicsManager.h"
#include "Shizuku.Flow/Graphics/CudaLbm.h"

#include "Shizuku.Flow/Command/Zoom.h"
#include "Shizuku.Flow/Command/Pan.h"
#include "Shizuku.Flow/Command/Rotate.h"
#include "Shizuku.Flow/Command/AddObstruction.h"
#include "Shizuku.Flow/Command/RemoveObstruction.h"
#include "Shizuku.Flow/Command/MoveObstruction.h"
#include "Shizuku.Flow/Command/PauseSimulation.h"
#include "Shizuku.Flow/Command/PauseRayTracing.h"
#include "Shizuku.Flow/Command/SetSimulationScale.h"
#include "Shizuku.Flow/Command/SetTimestepsPerFrame.h"
#include "Shizuku.Flow/Command/SetInletVelocity.h"
#include "Shizuku.Flow/Command/SetContourMode.h"
#include "Shizuku.Flow/Command/SetContourMinMax.h"

#include "Shizuku.Core/Ogl/Ogl.h"
#include "Shizuku.Core/Ogl/Shader.h"

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

    const char* MakeReadableString(const ContourMode contour)
    {
        switch (contour)
        {
        case ContourMode::VelocityMagnitude:
            return "Velocity.mag";
        case ContourMode::VelocityU:
            return "Velocity.u";
        case ContourMode::VelocityV:
            return "Velocity.v";
        case ContourMode::Pressure:
            return "Pressure";
        case ContourMode::StrainRate:
            return "Strain Rate";
        case ContourMode::Water:
            return "Water";
        default:
            throw "Unexpected contour mode";
        }
    }

}

Window::Window() : 
    m_simulationScale(2.5f),
    m_velocity(0.05f),
    m_timesteps(6),
    m_contourMode(ContourMode::Water),
    m_firstUIDraw(true),
    m_contourMinMax(0.0f, 1.0f),
    m_paused(false)
{
}

void Window::SetGraphicsManager(GraphicsManager &graphics)
{
    m_graphics = &graphics;
}

void Window::RegisterCommands()
{
    m_zoom = std::make_shared<Zoom>(*m_graphics);
    m_pan = std::make_shared<Pan>(*m_graphics);
    m_rotate = std::make_shared<Rotate>(*m_graphics);
    m_addObstruction = std::make_shared<AddObstruction>(*m_graphics);
    m_removeObstruction = std::make_shared<RemoveObstruction>(*m_graphics);
    m_moveObstruction = std::make_shared<MoveObstruction>(*m_graphics);
    m_pauseSimulation = std::make_shared<PauseSimulation>(*m_graphics);
    m_pauseRayTracing = std::make_shared<PauseRayTracing>(*m_graphics);
    m_setSimulationScale = std::make_shared<SetSimulationScale>(*m_graphics);
    m_timestepsPerFrame = std::make_shared<SetTimestepsPerFrame>(*m_graphics);
    m_setVelocity = std::make_shared<SetInletVelocity>(*m_graphics);
    m_setContourMode = std::make_shared<SetContourMode>(*m_graphics);
    m_setContourMinMax = std::make_shared<SetContourMinMax>(*m_graphics);

    m_setSimulationScale->Start(m_simulationScale);
    m_timestepsPerFrame->Start(m_timesteps);
    m_setVelocity->Start(m_velocity);
    m_setContourMode->Start(m_contourMode);
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


void Window::InitializeImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    const char* glsl_version = "#version 430 core";
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    ImGui::StyleColorsDark();
}

void Window::RegisterGlfwInputs()
{
    //glfw is in C so cannot bind instance method...
    glfwSetKeyCallback(m_window, KeyboardWrapper);
    glfwSetWindowSizeCallback(m_window, ResizeWrapper); 
    glfwSetCursorPosCallback(m_window, MouseMotionWrapper);
    glfwSetMouseButtonCallback(m_window, MouseButtonWrapper);
    glfwSetScrollCallback(m_window, MouseWheelWrapper);
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
        m_pan->Start(xf, yf);
    }
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE && state == GLFW_PRESS)
    {
        m_rotate->Start(xf, yf);
        m_removeObstruction->Start(xf, yf);
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && state == GLFW_PRESS)
    {
        m_addObstruction->Start(xf, yf);
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_PRESS)
    {
        m_moveObstruction->Start(xf, yf);
    }
    else
    {
        m_pan->End();
        m_rotate->End();
        m_moveObstruction->End();
        m_removeObstruction->End(xf, yf);
    }
}

void Window::MouseMotion(const int x, const int y)
{
    float xf = GetFloatCoordX(x);
    float yf = GetFloatCoordY(m_size.Height - y);
    m_pan->Track(xf, yf);
    m_rotate->Track(xf, yf);
    m_moveObstruction->Track(xf, yf);
}

void Window::GlfwMouseWheel(double xwheel, double ywheel)
{
    double x, y;
    glfwGetCursorPos(m_window, &x, &y);
    const int dir = ywheel > 0 ? 1 : 0;
    m_zoom->Start(dir, 0.6f);
}

void Window::GlfwKeyboard(int key, int scancode, int action, int mode)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        TogglePaused();
    }
}

void Window::TogglePaused()
{
    if (m_graphics->GetCudaLbm()->IsPaused())
    {
        m_pauseSimulation->End();
    }
    else
    {
        m_pauseSimulation->Start();
    }
}

void Window::SetPaused(const bool p_paused)
{

    if (m_graphics->GetCudaLbm()->IsPaused() && !p_paused)
    {
        m_pauseSimulation->End();
    }
    else if (!m_graphics->GetCudaLbm()->IsPaused() && p_paused)
    {
        m_pauseSimulation->Start();
    }
}

void Window::SetRayTracingPaused(const bool p_paused)
{
    if (m_graphics->IsRayTracingPaused() && !p_paused)
        m_pauseRayTracing->End();
    else if (!m_graphics->IsRayTracingPaused() && p_paused)
        m_pauseRayTracing->Start();
}

void Window::GlfwUpdateWindowTitle(const float fps, const Rect<int> &domainSize, const int tSteps)
{
    char fpsReport[256];
    int xDim = domainSize.Width;
    int yDim = domainSize.Height;
    sprintf_s(fpsReport, 
        "Shizuku Flow running at: %i timesteps/frame at %3.1f fps = %3.1f timesteps/second on %ix%i mesh",
        tSteps, fps, m_timesteps*fps, xDim, yDim);
    glfwSetWindowTitle(m_window, fpsReport);
}

void Window::Draw3D()
{
    m_fpsTracker.Tick();
    GraphicsManager* graphicsManager = m_graphics;
    graphicsManager->UpdateGraphicsInputs();
    graphicsManager->GetCudaLbm()->UpdateDeviceImage();

    graphicsManager->RunSimulation();

    graphicsManager->RenderFloorToTexture();

    graphicsManager->RunSurfaceRefraction();

    graphicsManager->UpdateViewMatrices();
    graphicsManager->UpdateViewTransformations();

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

    glewExperimental = GL_TRUE;
    glewInit();

    glEnable              ( GL_DEBUG_OUTPUT );
    glDebugMessageCallback(MessageCallback, 0);
}

void Window::DrawUI()
{
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 3.0f);
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (m_firstUIDraw)
    {
        ImGui::SetNextWindowSize(ImVec2(350,200));
        ImGui::SetNextWindowPos(ImVec2(430,20));
    }

    ImGui::Begin("Settings");
    {
        const ContourMode contour = m_contourMode;
        const char* currentItem = MakeReadableString(contour);
        if (ImGui::BeginCombo("Contour Mode", currentItem)) 
        {
            for (int i = 0; i < static_cast<int>(ContourMode::NUMB_CONTOUR_MODE); ++i)
            {
                const ContourMode contourItem = static_cast<ContourMode>(i);
                bool isSelected = (contourItem == contour);
                if (ImGui::Selectable(MakeReadableString(contourItem), isSelected))
                {
                    currentItem = MakeReadableString(contourItem);
                    m_contourMode = contourItem;
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (contour != m_contourMode)
            m_setContourMode->Start(m_contourMode);

        if (m_contourMode != ContourMode::Water)
        {
            MinMax<float> maxRange;
            const char* format = "%.3f";
            switch (m_contourMode){
            case VelocityMagnitude:
            case VelocityU:
                maxRange = MinMax<float>(0.f, 2.5f*m_velocity);
                break;
            case VelocityV:
                maxRange = MinMax<float>(-m_velocity, m_velocity);
                break;
            case Pressure:
                maxRange = MinMax<float>(0.99f, 1.01f);
                break;
            case StrainRate:
                maxRange = MinMax<float>(0.f, m_velocity*m_velocity);
                format = "%.5f";
                break;
            }

            const MinMax<float> oldMinMax = m_contourMinMax;
            m_contourMinMax.Clamp(maxRange);
            float minMax[2] = { m_contourMinMax.Min, m_contourMinMax.Max };
            ImGui::SliderFloat2("Contour Range", minMax, maxRange.Min, maxRange.Max, format);
            m_contourMinMax = MinMax<float>(minMax[0], minMax[1]);
            if (m_contourMinMax != oldMinMax)
                m_setContourMinMax->Start(m_contourMinMax);
        }

        const float oldScale = m_simulationScale;
        ImGui::SliderFloat("Scale", &m_simulationScale, 4.0f, 1.0f, "%.3f");
        if (oldScale != m_simulationScale)
            m_setSimulationScale->Start(m_simulationScale);

        const float oldTimesteps = m_timesteps;
        ImGui::SliderInt("Timesteps/Frame", &m_timesteps, 2, 30);
        if (oldTimesteps != m_timesteps)
            m_timestepsPerFrame->Start(m_timesteps);

        const float oldVel = m_velocity;
        ImGui::SliderFloat("Velocity", &m_velocity, 0.0f, 0.12f, "%.3f");
        if (oldVel != m_velocity)
            m_setVelocity->Start(m_velocity);

        ImGui::Spacing();

        const bool oldPaused = m_paused;
        if (ImGui::Checkbox("Pause Simulation", &m_paused) && m_paused != oldPaused)
            SetPaused(m_paused);

        const bool oldRayTracingPaused = m_rayTracingPaused;
        if (ImGui::Checkbox("Pause Ray Tracing", &m_rayTracingPaused) && m_rayTracingPaused != oldRayTracingPaused)
            SetRayTracingPaused(m_rayTracingPaused);
    }
    ImGui::End();

    ImGui::Render();

    int display_w, display_h;
    glfwMakeContextCurrent(m_window);
    glfwGetFramebufferSize(m_window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    m_firstUIDraw = false;
}

void Window::GlfwDisplay()
{
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    while (!glfwWindowShouldClose(m_window))
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glfwPollEvents();

        Draw3D();

        DrawUI();

        glfwSwapBuffers(m_window);
    }
    glfwTerminate();
}


