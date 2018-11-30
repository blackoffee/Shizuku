#include "Window.h"
#include "Shizuku.Flow/Flow.h"
#include <string.h>
#include <memory>

using namespace Shizuku::Presentation;
using namespace Shizuku::Flow;

int main(int argc, char **argv)
{
    std::shared_ptr<Flow> flow = std::make_shared<Flow>();

    Rect<int> windowSize = Rect<int>(800, 600);

    Window::Instance().SetGraphics(flow);
    Window::Instance().RegisterCommands();

    Window::Instance().Resize(windowSize);
    bool debug(false);
    for (int i = 0; i < argc; ++i)
    {
        if (strcmp(argv[i],"-d") == 0)
            debug = true;
    }

    Window::Instance().InitializeGlfw(debug);
    Window::Instance().InitializeImGui();
    Window::Instance().RegisterGlfwInputs();

    flow->Initialize();

    Window::Instance().Display();

    return 0;
}