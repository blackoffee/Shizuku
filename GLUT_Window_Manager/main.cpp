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
    int initialWindowWidth = 1200;
    int initialWindowHeight = 600;
    Window window(initialWindowWidth,initialWindowHeight);
    Panel* windowPanel = window.GetWindowPanel();

    SetUpWindow(*windowPanel);

    window.InitializeGLUT(argc, argv);
    window.InitializeGL();

    SetUpGLInterop(*windowPanel);
    SetUpCUDA(*windowPanel);

    window.Display();

    return 0;
}