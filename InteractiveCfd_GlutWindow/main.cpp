#include "Window.h"
#include "Graphics/GraphicsManager.h"

int main(int argc, char **argv)
{
    GraphicsManager* graphicsManager = &(GraphicsManager());

    //Layout::SetUpWindow(*windowPanel, *graphicsManager);
    //GraphicsManager* graphicsManager = windowPanel->GetPanel("Graphics")->GetGraphicsManager();

    graphicsManager->SetViewport(500, 500);
    graphicsManager->SetContourVar(ContourVariable::WATER_RENDERING);

    //Window window = Window(*graphicsManager);
    Window::Instance().SetGraphicsManager(*graphicsManager);
    Window::Instance().RegisterCommands();

    bool glfw = true;
    if (glfw)
    {
        Window::Instance().InitializeGlfw(argc, argv);

        //graphicsManager->UseCuda(false);
        graphicsManager->SetUpGLInterop();
        graphicsManager->SetUpCuda();
        graphicsManager->SetUpShaders();

        Window::Instance().GlfwDisplay();
    }
    else{
        Window::Instance().InitializeGLUT(argc, argv);
        Window::Instance().InitializeGL();

        //graphicsManager->UseCuda(false);
        graphicsManager->SetUpGLInterop();
        graphicsManager->SetUpCuda();
        graphicsManager->SetUpShaders();

        Window::Instance().Display();
    }

    return 0;
}