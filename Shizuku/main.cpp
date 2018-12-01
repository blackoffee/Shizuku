#include "Window.h"
#include "Shizuku.Flow/Flow.h"
#include <string.h>
#include <memory>

using namespace Shizuku::Presentation;
using namespace Shizuku::Flow;

int main(int argc, char **argv)
{
    bool debug(false);
    bool diag(false);
    for (int i = 0; i < argc; ++i)
    {
        if (strcmp(argv[i], "-d") == 0)
            debug = true;
        else if (strcmp(argv[i], "-v") == 0)
            diag = true;
    }

    std::shared_ptr<Flow> flow = std::make_shared<Flow>();

    Rect<int> windowSize = Rect<int>(1000, 700);

    Window::Instance().SetGraphics(flow);
    Window::Instance().RegisterCommands();

    Window::Instance().Resize(windowSize);
    Window::Instance().InitializeGlfw(debug);
    Window::Instance().InitializeImGui();
    Window::Instance().RegisterGlfwInputs();

    if (diag)
        Window::Instance().EnableDiagnostics();

    flow->Initialize();

    Window::Instance().Display();

    return 0;
}