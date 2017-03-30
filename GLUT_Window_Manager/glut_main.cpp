#include "Window.h"
#include "Layout.h"
#include "Panel.h"
#include "GraphicsManager.h"
#include "Command.h"
#include <GLEW/glew.h>
#include <GLUT/freeglut.h>

int main(int argc, char **argv)
{
    Panel* windowPanel = Window::Instance().GetWindowPanel();

    Layout::SetUpWindow(*windowPanel);
    GraphicsManager* graphicsManager = windowPanel->GetPanel("Graphics")->GetGraphicsManager();

    Window::Instance().InitializeGLUT(argc, argv);
    Window::Instance().InitializeGL();

    graphicsManager->UseCuda(false);
    graphicsManager->SetUpGLInterop();
    graphicsManager->SetUpCuda();
    graphicsManager->SetUpShaders();

    Window::Instance().Display();

    return 0;
}