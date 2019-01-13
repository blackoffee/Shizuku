#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_glfw.h"
#include "Window.h"
#include "../imgui/imgui_impl_opengl3.h"

#include "Shizuku.Flow/Command/Zoom.h"
#include "Shizuku.Flow/Command/Pan.h"
#include "Shizuku.Flow/Command/Rotate.h"
#include "Shizuku.Flow/Command/AddObstruction.h"
#include "Shizuku.Flow/Command/MoveObstruction.h"
#include "Shizuku.Flow/Command/PreSelectObstruction.h"
#include "Shizuku.Flow/Command/AddPreSelectionToSelection.h"
#include "Shizuku.Flow/Command/DeleteSelectedObstructions.h"
#include "Shizuku.Flow/Command/PauseSimulation.h"
#include "Shizuku.Flow/Command/PauseRayTracing.h"
#include "Shizuku.Flow/Command/SetSimulationScale.h"
#include "Shizuku.Flow/Command/SetTimestepsPerFrame.h"
#include "Shizuku.Flow/Command/SetInletVelocity.h"
#include "Shizuku.Flow/Command/SetContourMode.h"
#include "Shizuku.Flow/Command/SetContourMinMax.h"
#include "Shizuku.Flow/Command/SetSurfaceShadingMode.h"
#include "Shizuku.Flow/Command/SetWaterDepth.h"
#include "Shizuku.Flow/Command/RestartSimulation.h"
#include "Shizuku.Flow/Command/SetFloorWireframeVisibility.h"
#include "Shizuku.Flow/Command/Parameter/VelocityParameter.h"
#include "Shizuku.Flow/Command/Parameter/ScaleParameter.h"
#include "Shizuku.Flow/Command/Parameter/ModelSpacePointParameter.h"
#include "Shizuku.Flow/Command/Parameter/ScreenPointParameter.h"
#include "Shizuku.Flow/Command/Parameter/MinMaxParameter.h"
#include "Shizuku.Flow/Command/Parameter/DepthParameter.h"
#include "Shizuku.Flow/Command/Parameter/VisibilityParameter.h"

#include "Shizuku.Flow/Query.h"
#include "Shizuku.Flow/Flow.h"
#include "Shizuku.Flow/TimerKey.h"

#include "Shizuku.Core/Ogl/Ogl.h"
#include "Shizuku.Core/Ogl/Shader.h"

#include <GLFW/glfw3.h>

#include <boost/any.hpp>
#include <boost/none.hpp>

#include <iostream>
#include <typeinfo>
#include <memory>

using namespace Shizuku::Presentation;
using namespace Shizuku::Flow;

namespace
{
    void ResizeWrapper(GLFWwindow* window, int width, int height)
    {
        Window::Instance().Resize(Rect<int>(width, height));
    }

    void MouseButtonWrapper(GLFWwindow* window, int button, int state, int mods)
    {
        Window::Instance().MouseButton(button, state, mods);
    }

    void MouseMotionWrapper(GLFWwindow* window, double x, double y)
    {
        Window::Instance().MouseMotion(x, y);
    }

    void MouseWheelWrapper(GLFWwindow* window, double xwheel, double ywheel)
    {
        Window::Instance().MouseWheel(xwheel, ywheel);
    }

    void KeyboardWrapper(GLFWwindow* window, int key, int scancode, int action, int mode)
    {
        Window::Instance().Keyboard(key, scancode, action, mode);
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

    const char* MakeReadableString(const SurfaceShadingMode p_shadingMode)
    {
        switch (p_shadingMode)
        {
        case SurfaceShadingMode::RayTracing:
            return "Ray Tracing";
        case SurfaceShadingMode::SimplifiedRayTracing:
            return "Simplified Ray Tracing";
        case SurfaceShadingMode::Phong:
            return "Phong";
        default:
            throw "Unexpected contour mode";
        }
    }

    namespace TimeHistoryProviders{
        static float SolveFluid(void* data, int i) { return TimeHistory::Instance(TimerKey::SolveFluid).DataProvider(data, i); }
        static float PrepareFloor(void* data, int i) { return TimeHistory::Instance(TimerKey::PrepareFloor).DataProvider(data, i); }
        static float PrepareSurf(void* data, int i) { return TimeHistory::Instance(TimerKey::PrepareSurface).DataProvider(data, i); }
        static float ProcessFloor(void* data, int i) { return TimeHistory::Instance(TimerKey::ProcessFloor).DataProvider(data, i); }
        static float ProcessSurf(void* data, int i) { return TimeHistory::Instance(TimerKey::ProcessSurface).DataProvider(data, i); }
    }

    void CreateHistoryPlotLines(Query& p_query, const TimerKey p_key, const char* p_label)
    {
        const double time = p_query.GetTime(p_key);
        TimeHistory::Instance(p_key).Append(time);

        float(*provider) (void*, int) = NULL;
        switch (p_key)
        {
            case TimerKey::SolveFluid:
                provider = TimeHistoryProviders::SolveFluid;
                break;
            case TimerKey::PrepareSurface:
                provider = TimeHistoryProviders::PrepareSurf;
                break;
            case TimerKey::PrepareFloor:
                provider = TimeHistoryProviders::PrepareFloor;
                break;
            case TimerKey::ProcessSurface:
                provider = TimeHistoryProviders::ProcessSurf;
                break;
            case TimerKey::ProcessFloor:
                provider = TimeHistoryProviders::ProcessFloor;
                break;
            default:
                throw "Unexpected TimerKey";
        }

        const MinMax<double> minMax = TimeHistory::Instance(p_key).MinMax();
        char timeStr[64];
        sprintf_s(timeStr, "%f", time);
        const int chartHeight = 64;
        ImGui::PlotLines(p_label, provider, NULL, (TimeHistory::Instance(p_key).Size()), 0, timeStr,
            minMax.Min, minMax.Max, ImVec2(0, chartHeight));
    }

    float ScaleFromResolution(const float p_res)
    {
        return -p_res + 2.f;
    }
}

Window::Window() : 
    m_resolution(0.5f),
    m_velocity(0.05f),
    m_timesteps(6),
    m_contourMode(ContourMode::Water),
    m_firstUIDraw(true),
    m_contourMinMax(0.0f, 1.0f),
    m_depth(0.5f),
    m_paused(false),
    m_diagEnabled(false),
    //m_history(20),
    m_shadingMode(SurfaceShadingMode::RayTracing)
{
}

void Window::SetGraphics(std::shared_ptr<Shizuku::Flow::Flow> flow)
{
    m_flow = flow;
    m_query = std::make_shared<Query>(*flow);
}

void Window::RegisterCommands()
{
    m_zoom = std::make_shared<Zoom>(*m_flow);
    m_pan = std::make_shared<Pan>(*m_flow);
    m_rotate = std::make_shared<Rotate>(*m_flow);
    m_addObstruction = std::make_shared<AddObstruction>(*m_flow);
    m_moveObstruction = std::make_shared<MoveObstruction>(*m_flow);
    m_preSelectObst = std::make_shared<PreSelectObstruction>(*m_flow);
    m_addPreSelectionToSelection = std::make_shared<AddPreSelectionToSelection>(*m_flow);
	m_deleteSelectedObstructions = std::make_shared<DeleteSelectedObstructions>(*m_flow);
    m_pauseSimulation = std::make_shared<PauseSimulation>(*m_flow);
    m_pauseRayTracing = std::make_shared<PauseRayTracing>(*m_flow);
    m_restartSimulation = std::make_shared<RestartSimulation>(*m_flow);
    m_setSimulationScale = std::make_shared<SetSimulationScale>(*m_flow);
    m_timestepsPerFrame = std::make_shared<SetTimestepsPerFrame>(*m_flow);
    m_setVelocity = std::make_shared<SetInletVelocity>(*m_flow);
    m_setContourMode = std::make_shared<SetContourMode>(*m_flow);
    m_setContourMinMax = std::make_shared<SetContourMinMax>(*m_flow);
    m_setSurfaceShadingMode = std::make_shared<SetSurfaceShadingMode>(*m_flow);
    m_setDepth = std::make_shared<SetWaterDepth>(*m_flow);
    m_setFloorWireframeVisibility = std::make_shared<SetFloorWireframeVisibility>(*m_flow);
}

void Window::ApplyInitialFlowSettings()
{
    m_setSimulationScale->Start(boost::any(ScaleParameter(ScaleFromResolution(m_resolution))));
    m_timestepsPerFrame->Start(m_timesteps);
    m_setVelocity->Start(boost::any(VelocityParameter(m_velocity)));
    m_setContourMode->Start(m_contourMode);
    m_setSurfaceShadingMode->Start(m_shadingMode);
    m_setDepth->Start(boost::any(DepthParameter(m_depth)));

	//TODO: need mode switching
	m_preSelectObst->Start(boost::none);
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

void Window::Resize(GLFWwindow* window, int width, int height)
{
    Resize(Rect<int>(width, height));
}

void Window::Resize(const Rect<int>& size)
{
    m_size = size;
    m_flow->Resize(size);
}

void Window::EnableDebug()
{
    m_debug = true;
}

void Window::EnableDiagnostics()
{
    m_diagEnabled = true;
}

void Window::MouseButton(const int button, const int state, const int mod)
{
    double x, y;
    glfwGetCursorPos(m_window, &x, &y);
    const Types::Point<int> screenPos(x, m_size.Height - 1 - y);
    const ScreenPointParameter param(screenPos);

    if (button == GLFW_MOUSE_BUTTON_MIDDLE && state == GLFW_PRESS
        && mod == GLFW_MOD_CONTROL)
    {
        m_pan->Start(param);
    }
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE && state == GLFW_PRESS)
    {
        m_rotate->Start(param);
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && state == GLFW_PRESS)
    {
        m_addObstruction->Start(ModelSpacePointParameter(m_query->ProbeModelSpaceCoord(screenPos)));
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_PRESS)
    {
		m_addPreSelectionToSelection->Start(boost::none);

        //m_moveObstruction->Start(param);
    }
    else
    {
        m_pan->End(boost::none);
        m_rotate->End(boost::none);
        m_moveObstruction->End(boost::none);
		m_addPreSelectionToSelection->End(boost::none);
    }
}

void Window::MouseMotion(const int x, const int y)
{
    const ScreenPointParameter param(Types::Point<int>(x, m_size.Height - 1 - y));
    m_pan->Track(param);
    m_rotate->Track(param);
    m_moveObstruction->Track(param);
	m_preSelectObst->Track(param);
}

void Window::MouseWheel(double xwheel, double ywheel)
{
    double x, y;
    glfwGetCursorPos(m_window, &x, &y);
    const int dir = ywheel > 0 ? 1 : 0;
    m_zoom->Start(dir, 0.6f);
}

void Window::Keyboard(int key, int scancode, int action, int mode)
{
	switch (key)
	{
	case GLFW_KEY_SPACE:
		if (action == GLFW_RELEASE)
			TogglePaused();
		break;
	case GLFW_KEY_DELETE:
		if (action == GLFW_PRESS)
			m_deleteSelectedObstructions->Start(boost::none);
		else if (action == GLFW_RELEASE)
			m_deleteSelectedObstructions->End(boost::none);
	}
}

void Window::TogglePaused()
{
    m_paused = !m_paused;
    m_pauseSimulation->Start(boost::any(bool(m_paused)));
}

void Window::UpdateWindowTitle(const float fps, const Rect<int> &domainSize, const int tSteps)
{
    char fpsReport[256];
    const int xDim = domainSize.Width;
    const int yDim = domainSize.Height;
    sprintf_s(fpsReport, 
        "Shizuku Flow running at: %i timesteps/frame at %3.1f fps = %3.1f timesteps/second on %ix%i mesh",
        tSteps, fps, m_timesteps*fps, xDim, yDim);
    glfwSetWindowTitle(m_window, fpsReport);
}

void Window::Draw3D()
{
    m_fpsTracker.Tick();

    m_flow->Update();

	m_flow->SetUpFrame();

    m_flow->Draw3D();

    m_fpsTracker.Tock();

    UpdateWindowTitle(m_fpsTracker.GetFps(), m_query->SimulationDomain(), m_timesteps);
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

    if (m_debug)
    {
        glEnable              ( GL_DEBUG_OUTPUT );
        glDebugMessageCallback(MessageCallback, 0);
    }
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
        ImGui::SetNextWindowSize(ImVec2(350,270));
        ImGui::SetNextWindowPos(ImVec2(5,5));
        //! HACK - Adding obst has to happen after lbm dimensions are set. Need to split up Flow initilization sequence
        m_addObstruction->Start(ModelSpacePointParameter(Point<float>(-0.2f, 0.2f)));
        m_addObstruction->Start(ModelSpacePointParameter(Point<float>(-0.1f, -0.3f)));
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
                maxRange = MinMax<float>(0.f, 4.f*m_velocity);
                break;
            case VelocityU:
                maxRange = MinMax<float>(-1.5f*m_velocity, 4.f*m_velocity);
                break;
            case VelocityV:
                maxRange = MinMax<float>(-2.f*m_velocity, 2.f*m_velocity);
                break;
            case Pressure:
                maxRange = MinMax<float>(0.98f, 1.02f);
                break;
            case StrainRate:
                maxRange = MinMax<float>(0.f, 2.f*m_velocity*m_velocity);
                format = "%.5f";
                break;
            }

            const MinMax<float> oldMinMax = m_contourMinMax;
            m_contourMinMax.Clamp(maxRange);
            float minMax[2] = { m_contourMinMax.Min, m_contourMinMax.Max };
            ImGui::SliderFloat2("Contour Range", minMax, maxRange.Min, maxRange.Max, format);
            m_contourMinMax = MinMax<float>(minMax[0], minMax[1]);
            if (m_contourMinMax != oldMinMax)
                m_setContourMinMax->Start(boost::any(MinMaxParameter(m_contourMinMax)));
        }

        const float oldRes = m_resolution;
        ImGui::SliderFloat("Resolution", &m_resolution, 0.0f, 1.0f, "%.2f");
        if (oldRes != m_resolution)
            m_setSimulationScale->Start(boost::any(ScaleParameter(ScaleFromResolution(m_resolution))));

        const float oldTimesteps = m_timesteps;
        ImGui::SliderInt("Timesteps/Frame", &m_timesteps, 2, 30);
        if (oldTimesteps != m_timesteps)
            m_timestepsPerFrame->Start(m_timesteps);

        const float oldVel = m_velocity;
        ImGui::SliderFloat("Velocity", &m_velocity, 0.0f, 0.12f, "%.3f");
        if (oldVel != m_velocity)
            m_setVelocity->Start(boost::any(VelocityParameter(m_velocity)));

        const float oldDepth = m_depth;
        ImGui::SliderFloat("Depth", &m_depth, 0.0f, 5.f, "%.2f");
        if (oldDepth != m_depth)
            m_setDepth->Start(boost::any(DepthParameter(m_depth)));

        if (m_debug)
        {
            const SurfaceShadingMode shadingMode = m_shadingMode;
            const char* currentShadingItem = MakeReadableString(shadingMode);
            if (ImGui::BeginCombo("Shading Mode", currentShadingItem)) 
            {
                for (int i = 0; i < static_cast<int>(SurfaceShadingMode::NUMB_SHADING_MODE); ++i)
                {
                    const SurfaceShadingMode shadingItem = static_cast<SurfaceShadingMode>(i);
                    bool isSelected = (shadingItem == shadingMode);
                    if (ImGui::Selectable(MakeReadableString(shadingItem), isSelected))
                    {
                        currentShadingItem = MakeReadableString(shadingItem);
                        m_shadingMode = shadingItem;
                    }
                    if (isSelected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            if (shadingMode != m_shadingMode)
                m_setSurfaceShadingMode->Start(m_shadingMode);
        }

        ImGui::Spacing();

        if (ImGui::Button("Restart Simulation"))
            m_restartSimulation->Start();

        const bool oldPaused = m_paused;
        if (ImGui::Checkbox("Pause Simulation", &m_paused) && m_paused != oldPaused)
            m_pauseSimulation->Start(boost::any(bool(m_paused)));

        const bool oldRayTracingPaused = m_rayTracingPaused;
        if (ImGui::Checkbox("Pause Ray Tracing", &m_rayTracingPaused) && m_rayTracingPaused != oldRayTracingPaused)
            m_pauseRayTracing->Start(boost::any(bool(m_rayTracingPaused)));

        const bool oldFloorWireframe = m_floorWireframeVisible;
        if (ImGui::Checkbox("Show caustics mesh", &m_floorWireframeVisible) && m_floorWireframeVisible != oldFloorWireframe)
            m_setFloorWireframeVisibility->Start(boost::any(VisibilityParameter(m_floorWireframeVisible)));
    }
    ImGui::End();


    if (m_diagEnabled)
    {
        if (m_firstUIDraw)
        {
            ImGui::SetNextWindowSize(ImVec2(370,250));
            ImGui::SetNextWindowPos(ImVec2(m_size.Width-370-5,5));
        }

        ImGui::Begin("Diagnostics");
        {
            CreateHistoryPlotLines(*m_query, TimerKey::SolveFluid, "Solve Fluid");
            CreateHistoryPlotLines(*m_query, TimerKey::PrepareSurface, "Prepare Surface");
            CreateHistoryPlotLines(*m_query, TimerKey::PrepareFloor, "Prepare Floor");
            //CreateHistoryPlotLines(*m_query, TimerKey::ProcessSurface, "Process Surface");
            //CreateHistoryPlotLines(*m_query, TimerKey::ProcessFloor, "Process Floor");
        }
        ImGui::End();
    }

    ImGui::Render();

    int display_w, display_h;
    glfwMakeContextCurrent(m_window);
    glfwGetFramebufferSize(m_window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    m_firstUIDraw = false;

}

void Window::Display()
{
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    while (!glfwWindowShouldClose(m_window))
    {
        glfwPollEvents();

        Draw3D();

        DrawUI();

        glfwSwapBuffers(m_window);
    }
    glfwTerminate();
}


