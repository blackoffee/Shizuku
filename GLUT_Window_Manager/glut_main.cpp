#include <GLEW/glew.h>
#include <GLUT/freeglut.h>

#include <stdio.h>
#include <iostream>
#include <ostream>
#include <fstream>
#include <time.h>
#include <algorithm>

#include "main.h"
#include "Window.h"
#include "Command.h"

int main(int argc, char **argv)
{
    Panel* windowPanel = Window::Instance().GetWindowPanel();

    SetUpWindow(*windowPanel);

    Window::Instance().InitializeGLUT(argc, argv);
    Window::Instance().InitializeGL();

    SetUpGLInterop(*windowPanel);
    SetUpCUDA(*windowPanel);

    Window::Instance().Display();

    return 0;
}