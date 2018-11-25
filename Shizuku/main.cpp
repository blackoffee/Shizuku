#include "Window.h"
#include "Shizuku.Flow/Flow.h"
#include "Shizuku.Flow/Graphics/GraphicsManager.h"

using namespace Shizuku::Flow;

int main(int argc, char **argv)
{
    Flow flow = Flow();
    GraphicsManager graphics = *flow.Graphics();


    Rect<int> windowSize = Rect<int>(800, 600);
    //GraphicsManager* graphicsManager = &(GraphicsManager());

    //graphicsManager->SetContourVar(ContourVariable::WATER_RENDERING);

    Window::Instance().SetGraphicsManager(graphics);
    Window::Instance().RegisterCommands();

    Window::Instance().Resize(windowSize);
    Window::Instance().InitializeGlfw();
    Window::Instance().InitializeImGui();
    Window::Instance().RegisterGlfwInputs();

    flow.Resize(windowSize);
    flow.Initialize();
    //graphicsManager->SetViewport(windowSize);
    //graphicsManager->UseCuda(false);

    Window::Instance().GlfwDisplay();

    return 0;
}