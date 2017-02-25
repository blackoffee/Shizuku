#include <GLEW/glew.h>
#include <GLUT/freeglut.h>

#include <stdio.h>
#include <iostream>
#include <ostream>
#include <fstream>
#include <time.h>
#include <algorithm>

#include "Panel.h"

const int g_leftPanelWidth(350);
const int g_leftPanelHeight(500);


Panel Window;



void InitializeGL()
{
    glEnable(GL_LIGHT0);
    glewInit();
    glViewport(0,0,800,600);
}

void InitializeGLUT(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB|GLUT_DEPTH|GLUT_DOUBLE);

    glutInitWindowSize(1200,g_leftPanelHeight+100);
    glutInitWindowPosition(50,30);

    glutCreateWindow("New Window management");
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

void SetUpWindow()
{
    int windowWidth = 1200;
    int windowHeight = g_leftPanelHeight+100;

    Window.SetSize_Absolute(RectInt(200, 100, windowWidth, windowHeight));
    Window.m_draw = false;
    Window.SetName("Main Window");

    Panel* CDV = Window.CreateSubPanel(RectInt(0, 0, g_leftPanelWidth, g_leftPanelHeight), Panel::DEF_ABS,
        "CDV", Color(Color::DARK_GRAY));
    Panel* outputsPanel = CDV->CreateSubPanel(RectFloat(-1.f,  -0.9f, 2.f, 0.5f), Panel::DEF_REL,
        "Outputs", Color(Color::DARK_GRAY));
    Panel* inputsPanel  = CDV->CreateSubPanel(RectFloat(-1.f, -0.4f, 2.f, 0.6f), Panel::DEF_REL,
        "Inputs", Color(Color::DARK_GRAY));
    Panel* drawingPanel = CDV->CreateSubPanel(RectFloat(-1.f,  0.2f, 2.f, 0.8f), Panel::DEF_REL,
        "Drawing", Color(Color::DARK_GRAY));
    Panel* viewModePanel = CDV->CreateSubPanel(RectFloat(-1.f,  -1.f, 2.f, 0.1f), Panel::DEF_REL,
        "ViewMode", Color(Color::DARK_GRAY));


    outputsPanel->CreateButton(RectFloat(-0.9f, -0.2f +0.12f, 0.85f, 0.4f),
        Panel::DEF_REL, "X Velocity", Color(Color::GRAY));
    outputsPanel->CreateButton(RectFloat(-0.9f, -0.6f +0.08f, 0.85f, 0.4f),
        Panel::DEF_REL, "Velocity Magnitude", Color(Color::GRAY));
    outputsPanel->CreateButton(RectFloat(-0.9f, -1.f  +0.04f, 0.85f, 0.4f),
        Panel::DEF_REL, "StrainRate", Color(Color::GRAY));
    outputsPanel->CreateButton(RectFloat(0.05f, -0.2f +0.12f, 0.85f, 0.4f),
        Panel::DEF_REL, "Y Velocity", Color(Color::GRAY));
    outputsPanel->CreateButton(RectFloat(0.05f, -0.6f +0.08f, 0.85f, 0.4f),
        Panel::DEF_REL, "Pressure"  , Color(Color::GRAY));
    outputsPanel->CreateButton(RectFloat(0.05f, -1.f  +0.04f, 0.85f, 0.4f),
        Panel::DEF_REL, "Water Rendering", Color(Color::GRAY));

    viewModePanel->CreateButton(RectFloat(-0.9f , -1.f  +0.04f, 0.35f, 2.f),
        Panel::DEF_REL, "3D", Color(Color::GRAY));
    viewModePanel->CreateButton(RectFloat(-0.50f, -1.f  +0.04f, 0.35f, 2.f),
        Panel::DEF_REL, "2D", Color(Color::GRAY));

    Window.CreateSubPanel(RectInt(g_leftPanelWidth, 0, windowWidth-g_leftPanelWidth, windowHeight),
        Panel::DEF_ABS, "Graphics", Color(Color::RED));
    Window.GetPanel("Graphics")->m_draw = false;
    Window.GetPanel("Graphics")->CreateGraphicsManager();
    //Window.GetPanel("Graphics")->m_graphicsManager->SetObstructionsPointer(&g_obstructions[0]);
    //float scaleUp = Window.GetPanel("Graphics")->m_graphicsManager->GetScaleFactor();

//    UpdateDomainDimensionsBasedOnWindowSize(g_leftPanelHeight, g_leftPanelWidth,
//        windowWidth, windowHeight, scaleUp);


    float sliderH = 1.4f/3.f/2.f;
    float sliderBarW = 0.1f;
    float sliderBarH = 2.f;

    inputsPanel->CreateSubPanel(RectFloat(-0.9f, -1.f+0.4f+0.16f+sliderH*5, 0.5f, sliderH),
        Panel::DEF_REL, "Label_InletV", Color(Color::DARK_GRAY));
    inputsPanel->CreateSubPanel(RectFloat(-0.9f, -1.f+0.4f+0.12f+sliderH*3, 0.5f, sliderH),
        Panel::DEF_REL, "Label_Visc", Color(Color::DARK_GRAY));
    inputsPanel->CreateSubPanel(RectFloat(-0.9f, -1.f+0.4f+0.08f+sliderH, 0.5f, sliderH),
        Panel::DEF_REL, "Label_Resolution", Color(Color::DARK_GRAY));
    inputsPanel->CreateSlider(RectFloat(-0.9f, -1.f+0.4f+0.16f+sliderH*4, 1.8f, sliderH),
        Panel::DEF_REL, "Slider_InletV", Color(Color::LIGHT_GRAY));
    inputsPanel->CreateSlider(RectFloat(-0.9f, -1.f+0.4f+0.12f+sliderH*2, 1.8f, sliderH),
        Panel::DEF_REL, "Slider_Visc", Color(Color::LIGHT_GRAY));
    inputsPanel->CreateSlider(RectFloat(-0.9f, -1.f+0.4f+0.08f, 1.8f, sliderH),
        Panel::DEF_REL, "Slider_Resolution", Color(Color::LIGHT_GRAY));
    inputsPanel->CreateButton(RectFloat(-0.9f, -1.f+0.09f , 1.8f, 0.3f ),
        Panel::DEF_REL, "Initialize", Color(Color::GRAY));

    Window.GetPanel("Label_InletV")->SetDisplayText("Inlet Velocity");
    Window.GetSlider("Slider_InletV")->CreateSliderBar(RectFloat(0.7f, -sliderBarH*0.5f, sliderBarW, sliderBarH),
        Panel::DEF_REL, "SliderBar_InletV", Color(Color::GRAY));
    Window.GetSlider("Slider_InletV")->SetMaxValue(0.125f);
    Window.GetSlider("Slider_InletV")->SetMinValue(0.f);
    Window.GetSlider("Slider_InletV")->m_sliderBar1->UpdateValue();

    Window.GetPanel("Label_Visc")->SetDisplayText("Viscosity");
    Window.GetSlider("Slider_Visc")->CreateSliderBar(RectFloat(-0.85f, -sliderBarH*0.5f, sliderBarW, sliderBarH),
        Panel::DEF_REL, "SliderBar_Visc", Color(Color::GRAY));
    Window.GetSlider("Slider_Visc")->SetMaxValue(1.8f);
    Window.GetSlider("Slider_Visc")->SetMinValue(1.99f);
    Window.GetSlider("Slider_Visc")->m_sliderBar1->UpdateValue();

    Window.GetPanel("Label_Resolution")->SetDisplayText("Resolution");
    Window.GetSlider("Slider_Resolution")->CreateSliderBar(RectFloat(-0.3f, -sliderBarH*0.5f, sliderBarW, sliderBarH),
        Panel::DEF_REL, "SliderBar_Resolution", Color(Color::GRAY));
    Window.GetSlider("Slider_Resolution")->SetMaxValue(1.f);
    Window.GetSlider("Slider_Resolution")->SetMinValue(6.f);
    Window.GetSlider("Slider_Resolution")->m_sliderBar1->UpdateValue();


    std::string VarName = "Velocity Magnitude";
    std::string labelName = "Label_"+VarName;
    std::string sliderName = VarName;
    std::string sliderBarName1 = VarName+"Max";
    std::string sliderBarName2 = VarName+"Min";
    RectFloat contourSliderPosition{-0.9f, 0.2f+0.16f+(0.64f-sliderH*2)*0.5f, 1.8f, sliderH};
    outputsPanel->CreateSubPanel(RectFloat{-0.9f, 0.2f+0.16f+(0.64f-sliderH*2)*0.5f+sliderH, 0.5f, sliderH}
        , Panel::DEF_REL, "Label_Contour", Color(Color::DARK_GRAY));
    Window.GetPanel("Label_Contour")->SetDisplayText("Contour Color");
    float contourSliderBarWidth = 0.1f;
    float contourSliderBarHeight = 2.f;
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-1.f, -1, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.65f, -1, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    Window.GetSlider(sliderName)->SetMaxValue(INITIAL_UMAX*2.f);
    Window.GetSlider(sliderName)->SetMinValue(0.f);
    Window.GetSlider(sliderName)->m_sliderBar1->SetForegroundColor(Color::BLUE);
    Window.GetSlider(sliderName)->m_sliderBar2->SetForegroundColor(Color::WHITE);
    Window.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    Window.GetSlider(sliderName)->m_sliderBar2->UpdateValue();

    VarName = "X Velocity";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.85f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.65f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    Window.GetSlider(sliderName)->SetMaxValue(INITIAL_UMAX*1.8f);
    Window.GetSlider(sliderName)->SetMinValue(-INITIAL_UMAX*1.f);
    Window.GetSlider(sliderName)->m_sliderBar1->SetForegroundColor(Color::BLUE);
    Window.GetSlider(sliderName)->m_sliderBar2->SetForegroundColor(Color::WHITE);
    Window.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    Window.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    Window.GetSlider(sliderName)->Hide();

    VarName = "Y Velocity";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.65f, -1.f, contourSliderBarWidth,
        contourSliderBarHeight), Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.65f-contourSliderBarWidth*0.5f, -1.f,
        contourSliderBarWidth, contourSliderBarHeight), Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    Window.GetSlider(sliderName)->SetMaxValue(INITIAL_UMAX*1.f);
    Window.GetSlider(sliderName)->SetMinValue(-INITIAL_UMAX*1.f);
    Window.GetSlider(sliderName)->m_sliderBar1->SetForegroundColor(Color::BLUE);
    Window.GetSlider(sliderName)->m_sliderBar2->SetForegroundColor(Color::WHITE);
    Window.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    Window.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    Window.GetSlider(sliderName)->Hide();

    VarName = "StrainRate";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.9f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(0.35f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    Window.GetSlider(sliderName)->SetMaxValue(INITIAL_UMAX*0.1f);
    Window.GetSlider(sliderName)->SetMinValue(0.f);
    Window.GetSlider(sliderName)->m_sliderBar1->SetForegroundColor(Color::BLUE);
    Window.GetSlider(sliderName)->m_sliderBar2->SetForegroundColor(Color::WHITE);
    Window.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    Window.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    Window.GetSlider(sliderName)->Hide();

    VarName = "Pressure";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.45f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.45f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    Window.GetSlider(sliderName)->SetMaxValue(1.05f);
    Window.GetSlider(sliderName)->SetMinValue(0.95f);
    Window.GetSlider(sliderName)->m_sliderBar1->SetForegroundColor(Color::BLUE);
    Window.GetSlider(sliderName)->m_sliderBar2->SetForegroundColor(Color::WHITE);
    Window.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    Window.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    Window.GetSlider(sliderName)->Hide();

    VarName = "Water Rendering";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.45f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.45f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    Window.GetSlider(sliderName)->SetMaxValue(1.05f);
    Window.GetSlider(sliderName)->SetMinValue(0.95f);
    Window.GetSlider(sliderName)->m_sliderBar1->SetForegroundColor(Color::BLUE);
    Window.GetSlider(sliderName)->m_sliderBar2->SetForegroundColor(Color::WHITE);
    Window.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    Window.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    Window.GetSlider(sliderName)->Hide();


    //Drawing panel
    Panel* drawingPreview = Window.GetPanel("Drawing")->CreateSubPanel(RectFloat(-0.5f, -1.f, 1.5f, 1.5f),
        Panel::DEF_REL, "DrawingPreview", Color(Color::DARK_GRAY));
    Panel* drawingButtons = Window.GetPanel("Drawing")->CreateSubPanel(RectFloat(-0.9f, -1.f, 0.4f, 1.5f),
        Panel::DEF_REL, "DrawingButtons", Color(Color::DARK_GRAY));

    drawingPanel->CreateSlider(RectFloat(-0.9f, 0.9f-sliderH*0.75f*2,1.8f, sliderH*0.75f), Panel::DEF_REL,
        "Slider_Size", Color(Color::LIGHT_GRAY));
    drawingPanel->CreateSubPanel(RectFloat(-0.9f,0.9f-sliderH*0.75f, 0.5f, sliderH*0.75f), Panel::DEF_REL,
        "Label_Size", Color(Color::DARK_GRAY));
    Window.GetPanel("Label_Size")->SetDisplayText("Size");

    float leftEnd = -0.9f;
    float width = 1.8f;
    float buttonSpacing = 0.0f;
    drawingButtons->CreateButton(RectFloat(-0.9f, 0.7f-0.04f , 1.8f, 0.3f ), Panel::DEF_REL,
        "Square"    , Color(Color::GRAY));
    drawingButtons->CreateButton(RectFloat(-0.9f, 0.4f-0.08f , 1.8f, 0.3f ), Panel::DEF_REL,
        "Circle"    , Color(Color::GRAY));
    drawingButtons->CreateButton(RectFloat(-0.9f, 0.1f-0.12f , 1.8f, 0.3f ), Panel::DEF_REL,
        "Hor. Line" , Color(Color::GRAY));
    drawingButtons->CreateButton(RectFloat(-0.9f,-0.2f-0.16f , 1.8f, 0.3f ), Panel::DEF_REL,
        "Vert. Line", Color(Color::GRAY));
    Window.GetSlider("Slider_Size")->CreateSliderBar(RectFloat(-0.2f,-sliderBarH*0.5f, sliderBarW, sliderBarH),
        Panel::DEF_REL, "SliderBar_Size", Color(Color::GRAY));
    Window.GetSlider("Slider_Size")->SetMaxValue(15.f);
    Window.GetSlider("Slider_Size")->SetMinValue(1.f);
    Window.GetSlider("Slider_Size")->m_sliderBar1->UpdateValue();
    float currentObstSize = Window.GetSlider("Slider_Size")->m_sliderBar1->GetValue();
    //Window.GetPanel("Graphics")->m_graphicsManager->SetCurrentObstSize(currentObstSize);


//    SetUpButtons();
//    WaterRenderingButtonCallBack(); //default is water rendering
//    SquareButtonCallBack(); //default is square shape
//    ThreeDButtonCallBack();
}



void Draw()
{
    glOrtho(-1, 1, -1, 1, -100, 20);
    Window.DrawAll();

    glutSwapBuffers();
}

int main(int argc, char **argv)
{

    SetUpWindow();

    InitializeGLUT(argc, argv);
    InitializeGL();


    
    glutDisplayFunc(Draw);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);


    glutMainLoop();


    return 0;
}