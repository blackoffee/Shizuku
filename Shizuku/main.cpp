#include "Window.h"
#include "Shizuku.Flow/Flow.h"
#include <memory>

using namespace Shizuku::Flow;

int main(int argc, char **argv)
{
    std::shared_ptr<Flow> flow = std::make_shared<Flow>();

    Rect<int> windowSize = Rect<int>(800, 600);

    Window::Instance().SetGraphics(flow);
    Window::Instance().RegisterCommands();

    Window::Instance().Resize(windowSize);
    Window::Instance().InitializeGlfw();
    Window::Instance().InitializeImGui();
    Window::Instance().RegisterGlfwInputs();

    flow->Initialize();

    Window::Instance().GlfwDisplay();

    return 0;
}