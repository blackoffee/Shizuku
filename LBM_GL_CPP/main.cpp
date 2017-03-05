#include <GLEW/glew.h>
#include <GLUT/freeglut.h>
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_cuda_gl.h"

#include <stdio.h>
#include <iostream>
#include <ostream>
#include <fstream>
#include <time.h>
#include <algorithm>

#include "main.h"
#include "kernel.h"
#include "Mouse.h"
#include "Panel.h"
#include "common.h"
#include "Domain.h"
#include "FpsTracker.h"


const int g_leftPanelWidth(350);
const int g_leftPanelHeight(500);

FpsTracker g_fpsTracker;

//simulation inputs
Domain g_simDomain;

Panel Window;
Mouse theMouse;

const int g_glutMouseYOffset = 10; //hack to get better mouse precision

void Init()
{
    glEnable(GL_LIGHT0);
    glewInit();
    int windowWidth = Window.GetWidth();
    int windowHeight = Window.GetHeight();
    glViewport(0,0,windowWidth,windowHeight);

}

void SetUpWindow(Panel &rootPanel)
{
    int windowWidth = 1200;
    int windowHeight = g_leftPanelHeight+100;

    rootPanel.SetSize_Absolute(RectInt(200, 100, windowWidth, windowHeight));
    rootPanel.m_draw = false;
    rootPanel.SetName("Main Window");
    theMouse.SetBasePanel(&rootPanel);

    Panel* CDV = rootPanel.CreateSubPanel(RectInt(0, 0, g_leftPanelWidth, g_leftPanelHeight), Panel::DEF_ABS,
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

    rootPanel.CreateSubPanel(RectInt(g_leftPanelWidth, 0, windowWidth-g_leftPanelWidth, windowHeight),
        Panel::DEF_ABS, "Graphics", Color(Color::RED));
    rootPanel.GetPanel("Graphics")->m_draw = false;
    rootPanel.GetPanel("Graphics")->CreateGraphicsManager();
    float scaleUp = rootPanel.GetPanel("Graphics")->GetGraphicsManager()->GetScaleFactor();

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

    rootPanel.GetPanel("Label_InletV")->SetDisplayText("Inlet Velocity");
    rootPanel.GetSlider("Slider_InletV")->CreateSliderBar(RectFloat(0.7f, -sliderBarH*0.5f, sliderBarW, sliderBarH),
        Panel::DEF_REL, "SliderBar_InletV", Color(Color::GRAY));
    rootPanel.GetSlider("Slider_InletV")->SetMaxValue(0.125f);
    rootPanel.GetSlider("Slider_InletV")->SetMinValue(0.f);
    rootPanel.GetSlider("Slider_InletV")->m_sliderBar1->UpdateValue();

    rootPanel.GetPanel("Label_Visc")->SetDisplayText("Viscosity");
    rootPanel.GetSlider("Slider_Visc")->CreateSliderBar(RectFloat(-0.85f, -sliderBarH*0.5f, sliderBarW, sliderBarH),
        Panel::DEF_REL, "SliderBar_Visc", Color(Color::GRAY));
    rootPanel.GetSlider("Slider_Visc")->SetMaxValue(1.8f);
    rootPanel.GetSlider("Slider_Visc")->SetMinValue(1.99f);
    rootPanel.GetSlider("Slider_Visc")->m_sliderBar1->UpdateValue();

    rootPanel.GetPanel("Label_Resolution")->SetDisplayText("Resolution");
    rootPanel.GetSlider("Slider_Resolution")->CreateSliderBar(RectFloat(-0.3f, -sliderBarH*0.5f, sliderBarW, sliderBarH),
        Panel::DEF_REL, "SliderBar_Resolution", Color(Color::GRAY));
    rootPanel.GetSlider("Slider_Resolution")->SetMaxValue(1.f);
    rootPanel.GetSlider("Slider_Resolution")->SetMinValue(6.f);
    rootPanel.GetSlider("Slider_Resolution")->m_sliderBar1->UpdateValue();


    std::string VarName = "Velocity Magnitude";
    std::string labelName = "Label_"+VarName;
    std::string sliderName = VarName;
    std::string sliderBarName1 = VarName+"Max";
    std::string sliderBarName2 = VarName+"Min";
    RectFloat contourSliderPosition{-0.9f, 0.2f+0.16f+(0.64f-sliderH*2)*0.5f, 1.8f, sliderH};
    outputsPanel->CreateSubPanel(RectFloat{-0.9f, 0.2f+0.16f+(0.64f-sliderH*2)*0.5f+sliderH, 0.5f, sliderH}
        , Panel::DEF_REL, "Label_Contour", Color(Color::DARK_GRAY));
    rootPanel.GetPanel("Label_Contour")->SetDisplayText("Contour Color");
    float contourSliderBarWidth = 0.1f;
    float contourSliderBarHeight = 2.f;
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    rootPanel.GetSlider(sliderName)->CreateSliderBar(RectFloat(-1.f, -1, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    rootPanel.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.65f, -1, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    rootPanel.GetSlider(sliderName)->SetMaxValue(INITIAL_UMAX*2.f);
    rootPanel.GetSlider(sliderName)->SetMinValue(0.f);
    rootPanel.GetSlider(sliderName)->m_sliderBar1->SetForegroundColor(Color::BLUE);
    rootPanel.GetSlider(sliderName)->m_sliderBar2->SetForegroundColor(Color::WHITE);
    rootPanel.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    rootPanel.GetSlider(sliderName)->m_sliderBar2->UpdateValue();

    VarName = "X Velocity";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    rootPanel.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.85f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    rootPanel.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.65f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    rootPanel.GetSlider(sliderName)->SetMaxValue(INITIAL_UMAX*1.8f);
    rootPanel.GetSlider(sliderName)->SetMinValue(-INITIAL_UMAX*1.f);
    rootPanel.GetSlider(sliderName)->m_sliderBar1->SetForegroundColor(Color::BLUE);
    rootPanel.GetSlider(sliderName)->m_sliderBar2->SetForegroundColor(Color::WHITE);
    rootPanel.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    rootPanel.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    rootPanel.GetSlider(sliderName)->Hide();

    VarName = "Y Velocity";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    rootPanel.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.65f, -1.f, contourSliderBarWidth,
        contourSliderBarHeight), Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    rootPanel.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.65f-contourSliderBarWidth*0.5f, -1.f,
        contourSliderBarWidth, contourSliderBarHeight), Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    rootPanel.GetSlider(sliderName)->SetMaxValue(INITIAL_UMAX*1.f);
    rootPanel.GetSlider(sliderName)->SetMinValue(-INITIAL_UMAX*1.f);
    rootPanel.GetSlider(sliderName)->m_sliderBar1->SetForegroundColor(Color::BLUE);
    rootPanel.GetSlider(sliderName)->m_sliderBar2->SetForegroundColor(Color::WHITE);
    rootPanel.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    rootPanel.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    rootPanel.GetSlider(sliderName)->Hide();

    VarName = "StrainRate";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    rootPanel.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.9f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    rootPanel.GetSlider(sliderName)->CreateSliderBar(RectFloat(0.35f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    rootPanel.GetSlider(sliderName)->SetMaxValue(INITIAL_UMAX*0.1f);
    rootPanel.GetSlider(sliderName)->SetMinValue(0.f);
    rootPanel.GetSlider(sliderName)->m_sliderBar1->SetForegroundColor(Color::BLUE);
    rootPanel.GetSlider(sliderName)->m_sliderBar2->SetForegroundColor(Color::WHITE);
    rootPanel.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    rootPanel.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    rootPanel.GetSlider(sliderName)->Hide();

    VarName = "Pressure";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    rootPanel.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.45f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    rootPanel.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.45f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    rootPanel.GetSlider(sliderName)->SetMaxValue(1.05f);
    rootPanel.GetSlider(sliderName)->SetMinValue(0.95f);
    rootPanel.GetSlider(sliderName)->m_sliderBar1->SetForegroundColor(Color::BLUE);
    rootPanel.GetSlider(sliderName)->m_sliderBar2->SetForegroundColor(Color::WHITE);
    rootPanel.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    rootPanel.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    rootPanel.GetSlider(sliderName)->Hide();

    VarName = "Water Rendering";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    rootPanel.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.45f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    rootPanel.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.45f, -1.f, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    rootPanel.GetSlider(sliderName)->SetMaxValue(1.05f);
    rootPanel.GetSlider(sliderName)->SetMinValue(0.95f);
    rootPanel.GetSlider(sliderName)->m_sliderBar1->SetForegroundColor(Color::BLUE);
    rootPanel.GetSlider(sliderName)->m_sliderBar2->SetForegroundColor(Color::WHITE);
    rootPanel.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    rootPanel.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    rootPanel.GetSlider(sliderName)->Hide();


    //Drawing panel
    Panel* drawingPreview = rootPanel.GetPanel("Drawing")->CreateSubPanel(RectFloat(-0.5f, -1.f, 1.5f, 1.5f),
        Panel::DEF_REL, "DrawingPreview", Color(Color::DARK_GRAY));
    Panel* drawingButtons = rootPanel.GetPanel("Drawing")->CreateSubPanel(RectFloat(-0.9f, -1.f, 0.4f, 1.5f),
        Panel::DEF_REL, "DrawingButtons", Color(Color::DARK_GRAY));

    drawingPanel->CreateSlider(RectFloat(-0.9f, 0.9f-sliderH*0.75f*2,1.8f, sliderH*0.75f), Panel::DEF_REL,
        "Slider_Size", Color(Color::LIGHT_GRAY));
    drawingPanel->CreateSubPanel(RectFloat(-0.9f,0.9f-sliderH*0.75f, 0.5f, sliderH*0.75f), Panel::DEF_REL,
        "Label_Size", Color(Color::DARK_GRAY));
    rootPanel.GetPanel("Label_Size")->SetDisplayText("Size");

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
    rootPanel.GetSlider("Slider_Size")->CreateSliderBar(RectFloat(-0.2f,-sliderBarH*0.5f, sliderBarW, sliderBarH),
        Panel::DEF_REL, "SliderBar_Size", Color(Color::GRAY));
    rootPanel.GetSlider("Slider_Size")->SetMaxValue(15.f);
    rootPanel.GetSlider("Slider_Size")->SetMinValue(1.f);
    rootPanel.GetSlider("Slider_Size")->m_sliderBar1->UpdateValue();
    float currentObstSize = rootPanel.GetSlider("Slider_Size")->m_sliderBar1->GetValue();
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetCurrentObstSize(currentObstSize);


    SetUpButtons(rootPanel);
    WaterRenderingButtonCallBack(rootPanel); //default is water rendering
    SquareButtonCallBack(rootPanel); //default is square shape
    ThreeDButtonCallBack(rootPanel);
}

/*----------------------------------------------------------------------------------------
 *	Button setup
 */

Slider* GetCurrentContourSlider(Panel &rootPanel)
{
    if (rootPanel.GetSlider("Velocity Magnitude")->m_draw == true)
        return rootPanel.GetSlider("Velocity Magnitude");
    else if (rootPanel.GetSlider("X Velocity")->m_draw == true)
        return rootPanel.GetSlider("X Velocity");
    else if (rootPanel.GetSlider("Y Velocity")->m_draw == true) 
        return rootPanel.GetSlider("Y Velocity");
    else if (rootPanel.GetSlider("StrainRate")->m_draw == true) 
        return rootPanel.GetSlider("StrainRate");
    else if (rootPanel.GetSlider("Pressure")->m_draw == true) 
        return rootPanel.GetSlider("Pressure");
    else if (rootPanel.GetSlider("Water Rendering")->m_draw == true) 
        return rootPanel.GetSlider("Water Rendering");
}


void InitializeButtonCallBack(Panel &rootPanel)
{
    GraphicsManager* graphicsManager = rootPanel.GetPanel("Graphics")->GetGraphicsManager();
    Graphics* graphics = graphicsManager->GetGraphics();
    cudaGraphicsResource* cudaSolutionField = graphics->GetCudaSolutionGraphicsResource();
    float4 *dptr;
    cudaGraphicsMapResources(1, &cudaSolutionField, 0);
    size_t num_bytes,num_bytes2;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cudaSolutionField);

    CudaLbm* cudaLbm = graphicsManager->GetCudaLbm();

    float* fA_d = cudaLbm->GetFA();
    float* fB_d = cudaLbm->GetFB();
    int* im_d = cudaLbm->GetImage();

    float u = rootPanel.GetSlider("Slider_InletV")->m_sliderBar1->GetValue();
    Domain* domain = cudaLbm->GetDomain();
    InitializeDomain(dptr, fA_d, im_d, u, *domain);
}

void VelMagButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* contourButtons = rootPanel.GetButtonGroup("ContourButtons");
    contourButtons->ExclusiveEnable(rootPanel.GetButton("Velocity Magnitude"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetContourVar(VEL_MAG);
}

void VelXButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* contourButtons = rootPanel.GetButtonGroup("ContourButtons");
    contourButtons->ExclusiveEnable(rootPanel.GetButton("X Velocity"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetContourVar(VEL_U);
}

void VelYButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* contourButtons = rootPanel.GetButtonGroup("ContourButtons");
    contourButtons->ExclusiveEnable(rootPanel.GetButton("Y Velocity"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetContourVar(VEL_V);
}

void StrainRateButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* contourButtons = rootPanel.GetButtonGroup("ContourButtons");
    contourButtons->ExclusiveEnable(rootPanel.GetButton("StrainRate"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetContourVar(STRAIN_RATE);
}

void PressureButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* contourButtons = rootPanel.GetButtonGroup("ContourButtons");
    contourButtons->ExclusiveEnable(rootPanel.GetButton("Pressure"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetContourVar(PRESSURE);
}

void WaterRenderingButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* contourButtons = rootPanel.GetButtonGroup("ContourButtons");
    contourButtons->ExclusiveEnable(rootPanel.GetButton("Water Rendering"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetContourVar(WATER_RENDERING);
}

void SquareButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* shapeButtons = rootPanel.GetButtonGroup("ShapeButtons");
    shapeButtons->ExclusiveEnable(rootPanel.GetButton("Square"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetCurrentObstShape(Obstruction::SQUARE);
}

void CircleButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* shapeButtons = rootPanel.GetButtonGroup("ShapeButtons");
    shapeButtons->ExclusiveEnable(rootPanel.GetButton("Circle"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetCurrentObstShape(Obstruction::CIRCLE);
}

void HorLineButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* shapeButtons = rootPanel.GetButtonGroup("ShapeButtons");
    shapeButtons->ExclusiveEnable(rootPanel.GetButton("Hor. Line"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetCurrentObstShape(Obstruction::HORIZONTAL_LINE);
}

void VertLineButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* shapeButtons = rootPanel.GetButtonGroup("ShapeButtons");
    shapeButtons->ExclusiveEnable(rootPanel.GetButton("Vert. Line"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetCurrentObstShape(Obstruction::VERTICAL_LINE);
}

void ThreeDButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* viewModeButtons = rootPanel.GetButtonGroup("ViewModeButtons");
    viewModeButtons->ExclusiveEnable(rootPanel.GetButton("3D"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetViewMode(THREE_DIMENSIONAL);
}

void TwoDButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* viewModeButtons = rootPanel.GetButtonGroup("ViewModeButtons");
    viewModeButtons->ExclusiveEnable(rootPanel.GetButton("2D"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetViewMode(TWO_DIMENSIONAL);
}

void SetUpButtons(Panel &rootPanel)
{
    rootPanel.GetButton("Initialize")->SetCallback(InitializeButtonCallBack);
    rootPanel.GetButton("Velocity Magnitude")->SetCallback(VelMagButtonCallBack);
    rootPanel.GetButton("X Velocity")->SetCallback(VelXButtonCallBack);
    rootPanel.GetButton("Y Velocity")->SetCallback(VelYButtonCallBack);
    rootPanel.GetButton("StrainRate")->SetCallback(StrainRateButtonCallBack);
    rootPanel.GetButton("Pressure"  )->SetCallback(PressureButtonCallBack);
    rootPanel.GetButton("Water Rendering")->SetCallback(WaterRenderingButtonCallBack);

    std::vector<Button*> buttons = {
        rootPanel.GetButton("Velocity Magnitude"),
        rootPanel.GetButton("X Velocity"),
        rootPanel.GetButton("Y Velocity"),
        rootPanel.GetButton("StrainRate"),
        rootPanel.GetButton("Pressure"),
        rootPanel.GetButton("Water Rendering") };
    ButtonGroup* contourButtonGroup = rootPanel.CreateButtonGroup("ContourButtons", buttons);

    //Shape buttons
    rootPanel.GetButton("Square")->SetCallback(SquareButtonCallBack);
    rootPanel.GetButton("Circle")->SetCallback(CircleButtonCallBack);
    rootPanel.GetButton("Hor. Line")->SetCallback(HorLineButtonCallBack);
    rootPanel.GetButton("Vert. Line")->SetCallback(VertLineButtonCallBack);

    std::vector<Button*> buttons2 = {
        rootPanel.GetButton("Square"),
        rootPanel.GetButton("Circle"),
        rootPanel.GetButton("Hor. Line"),
        rootPanel.GetButton("Vert. Line") };
    ButtonGroup* shapeButtonGroup = rootPanel.CreateButtonGroup("ShapeButtons", buttons2);

    rootPanel.GetButton("3D")->SetCallback(ThreeDButtonCallBack);
    rootPanel.GetButton("2D")->SetCallback(TwoDButtonCallBack);
    
    std::vector<Button*> buttons3 = {
        rootPanel.GetButton("2D"),
        rootPanel.GetButton("3D")
    };
    ButtonGroup* viewModeButtonGroup = rootPanel.CreateButtonGroup("ViewModeButtons", buttons3);

}

void DrawShapePreview(Panel &rootPanel)
{
    Panel* previewPanel = rootPanel.GetPanel("DrawingPreview");
    float centerX = previewPanel->GetRectFloatAbs().GetCentroidX();
    float centerY = previewPanel->GetRectFloatAbs().GetCentroidY();
    int windowWidth = rootPanel.GetWidth();
    int windowHeight = rootPanel.GetHeight();
    float graphicsToWindowScaleFactor = static_cast<float>(windowWidth)/
        rootPanel.GetPanel("Graphics")->GetRectIntAbs().m_w;

    GraphicsManager *graphicsManager = rootPanel.GetPanel("Graphics")->GetGraphicsManager();
    CudaLbm* cudaLbm = graphicsManager->GetCudaLbm();
    int xDimVisible = cudaLbm->GetDomain()->GetXDimVisible();
    int yDimVisible = cudaLbm->GetDomain()->GetYDimVisible();
    float currentSize = rootPanel.GetSlider("Slider_Size")->m_sliderBar1->GetValue();
    int graphicsWindowWidth = rootPanel.GetPanel("Graphics")->GetRectIntAbs().m_w;
    int graphicsWindowHeight = rootPanel.GetPanel("Graphics")->GetRectIntAbs().m_h;
    int r1ix = currentSize*static_cast<float>(graphicsWindowWidth) / (xDimVisible); //r1x in pixels
    int r1iy = currentSize*static_cast<float>(graphicsWindowHeight) / (yDimVisible); //r1x in pixels
    float r1fx = static_cast<float>(r1ix) / windowWidth*2.f;
    float r1fy = static_cast<float>(r1iy) / windowHeight*2.f;

    Obstruction::Shape currentShape = graphicsManager->GetCurrentObstShape();
    glColor3f(0.8f,0.8f,0.8f);
    switch (currentShape)
    {
    case Obstruction::CIRCLE:
    {
        glBegin(GL_TRIANGLE_FAN);
        int circleResolution = 20;
        glVertex2f(centerX, centerY);
        for (int i = 0; i <= circleResolution; i++)
        {
            glVertex2f(centerX + r1fx*cos(i*2.f*PI/circleResolution),
                        centerY + r1fy*sin(i*2.f*PI/circleResolution));
        }
        glEnd();
        break;
    }
    case Obstruction::SQUARE:
    {
        glBegin(GL_QUADS);
            glVertex2f(centerX - r1fx, centerY + r1fy);
            glVertex2f(centerX - r1fx, centerY - r1fy);
            glVertex2f(centerX + r1fx, centerY - r1fy);
            glVertex2f(centerX + r1fx, centerY + r1fy);
        glEnd();
        break;
    }
    case Obstruction::HORIZONTAL_LINE:
    {
        r1fy = static_cast<float>(LINE_OBST_WIDTH) / windowHeight*2.f;
        glBegin(GL_QUADS);
            glVertex2f(centerX - r1fx*2.f, centerY + r1fy);
            glVertex2f(centerX - r1fx*2.f, centerY - r1fy);
            glVertex2f(centerX + r1fx*2.f, centerY - r1fy);
            glVertex2f(centerX + r1fx*2.f, centerY + r1fy);
        glEnd();
        break;
    }
    case Obstruction::VERTICAL_LINE:
    {
        r1fx = static_cast<float>(LINE_OBST_WIDTH) / windowWidth*2.f;
        glBegin(GL_QUADS);
            glVertex2f(centerX - r1fx, centerY + r1fy*2.f);
            glVertex2f(centerX - r1fx, centerY - r1fy*2.f);
            glVertex2f(centerX + r1fx, centerY - r1fy*2.f);
            glVertex2f(centerX + r1fx, centerY + r1fy*2.f);
        glEnd();
        break;
    }
    }
}

/*----------------------------------------------------------------------------------------
 *	GL Interop Functions
 */

void SetUpGLInterop(Panel &rootPanel)
{
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
    unsigned int solutionMemorySize = MAX_XDIM*MAX_YDIM * 4 * sizeof(float);
    unsigned int floorSize = MAX_XDIM*MAX_YDIM * 4 * sizeof(float);
    GraphicsManager* graphicsManager = rootPanel.GetPanel("Graphics")->GetGraphicsManager();
    graphicsManager->GetGraphics()->SetUpGLInterOp(solutionMemorySize+floorSize);
}

void CleanUpGLInterop(Panel &rootPanel)
{
    GraphicsManager* graphicsManager = rootPanel.GetPanel("Graphics")->GetGraphicsManager();
    graphicsManager->GetGraphics()->CleanUpGLInterOp();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

/*----------------------------------------------------------------------------------------
 *	CUDA calls
 */
void SetUpCUDA(Panel &rootPanel)
{
    float4 rayCastIntersect{ 0, 0, 0, 1e6 };

    GraphicsManager* graphicsManager = rootPanel.GetPanel("Graphics")->GetGraphicsManager();
    cudaMalloc((void **)&graphicsManager->m_rayCastIntersect_d, sizeof(float4));
    cudaMemcpy(graphicsManager->m_rayCastIntersect_d, &rayCastIntersect, sizeof(float4), cudaMemcpyHostToDevice);

    CudaLbm* cudaLbm = graphicsManager->GetCudaLbm();
    cudaLbm->AllocateDeviceMemory();
    cudaLbm->InitializeDeviceMemory();

    float u = rootPanel.GetSlider("Slider_InletV")->m_sliderBar1->GetValue();

    float* fA_d = cudaLbm->GetFA();
    float* fB_d = cudaLbm->GetFB();
    int* im_d = cudaLbm->GetImage();
    float* floor_d = cudaLbm->GetFloorTemp();
    Obstruction* obst_d = cudaLbm->GetDeviceObst();

    Graphics* graphics = graphicsManager->GetGraphics();
    cudaGraphicsResource* cudaSolutionField = graphics->GetCudaSolutionGraphicsResource();
    float4 *dptr;
    cudaGraphicsMapResources(1, &cudaSolutionField, 0);
    size_t num_bytes,num_bytes2;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cudaSolutionField);

    Domain* domain = cudaLbm->GetDomain();
    InitializeDomain(dptr, fA_d, im_d, u, *domain);
    InitializeDomain(dptr, fB_d, im_d, u, *domain);

    InitializeFloor(dptr, floor_d, *domain);

    cudaGraphicsUnmapResources(1, &cudaSolutionField, 0);

}

void Draw2D(Panel &rootPanel)
{
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1,1,-1,1,-100,20);
    rootPanel.DrawAll();
    DrawShapePreview(rootPanel);
}


/*----------------------------------------------------------------------------------------
 *	Mouse interactions
 */

void MouseButton(int button, int state, int x, int y)
{
    theMouse.Click(x, theMouse.m_winH-y-g_glutMouseYOffset, button, state);
}

void MouseMotion(int x, int y)
{
    int windowWidth = Window.GetWidth();
    int windowHeight = Window.GetHeight();
    if (x >= 0 && x <= windowWidth && y>=0 && y<=windowHeight)
    {
        theMouse.Move(x, theMouse.m_winH-y-g_glutMouseYOffset);
    }
}


void MouseWheel(int button, int dir, int x, int y)
{
    theMouse.Wheel(button, dir, x, y);
}

void Keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
    case (' ') :
        GraphicsManager* graphicsManager = Window.GetPanel("Graphics")->GetGraphicsManager();
        CudaLbm* cudaLbm = graphicsManager->GetCudaLbm();
        bool currentPausedState = cudaLbm->IsPaused();
        cudaLbm->SetPausedState(!currentPausedState);
        break;
    }
}

void UpdateWindowDimensionsBasedOnAspectRatio(int& heightOut, int& widthOut, int area,
    int leftPanelHeight, int leftPanelWidth, int xDim, int yDim, float scaleUp)
{
    float aspectRatio = static_cast<float>(xDim) / yDim;
    float leftPanelW = static_cast<float>(leftPanelWidth);
    heightOut = scaleUp*(-scaleUp*leftPanelW+sqrt(scaleUp*scaleUp*leftPanelW*leftPanelW
        +scaleUp*scaleUp*4*aspectRatio*area))/(scaleUp*scaleUp*2.f*aspectRatio);
    heightOut = std::max(heightOut, leftPanelHeight);
    widthOut = heightOut*aspectRatio+leftPanelW;
}

void UpdateDomainDimensionsBasedOnWindowSize(Panel &rootPanel, int leftPanelHeight, int leftPanelWidth)
{
    GraphicsManager *graphicsManager = rootPanel.GetPanel("Graphics")->GetGraphicsManager();
    float scaleUp = graphicsManager->GetScaleFactor();
    int windowWidth = rootPanel.GetWidth();
    int windowHeight = rootPanel.GetHeight();
    int xDimVisible = static_cast<float>(windowWidth - leftPanelWidth) / scaleUp;
    int yDimVisible = ceil(static_cast<float>(windowHeight) / scaleUp);
    CudaLbm* cudaLbm = graphicsManager->GetCudaLbm();
    cudaLbm->GetDomain()->SetXDimVisible(xDimVisible);
    cudaLbm->GetDomain()->SetYDimVisible(yDimVisible);
}

void Resize(int windowWidth, int windowHeight)
{
    UpdateDomainDimensionsBasedOnWindowSize(Window, g_leftPanelHeight, g_leftPanelWidth);

    theMouse.m_winW = windowWidth;
    theMouse.m_winH = windowHeight;

    RectInt rect = { 200, 100, windowWidth, windowHeight };
    Window.SetSize_Absolute(rect);
    rect = { 0, windowHeight - g_leftPanelHeight, g_leftPanelWidth, g_leftPanelHeight };
    Window.GetPanel("CDV")->SetSize_Absolute(rect);
    rect = { g_leftPanelWidth, 0, windowWidth - g_leftPanelWidth, windowHeight };
    Window.GetPanel("Graphics")->SetSize_Absolute(rect);
    Window.UpdateAll();

    glViewport(0, 0, windowWidth, windowHeight);

    //UpdateDeviceImage();

    GraphicsManager *graphicsManager = Window.GetPanel("Graphics")->GetGraphicsManager();
    graphicsManager->GetCudaLbm()->UpdateDeviceImage();

}

void UpdateLbmInputs(CudaLbm &cudaLbm, Panel &rootPanel)
{
    //get simulation inputs
    float u = rootPanel.GetSlider("Slider_InletV")->m_sliderBar1->GetValue();
    float omega = rootPanel.GetSlider("Slider_Visc")->m_sliderBar1->GetValue();
    cudaLbm.SetInletVelocity(u);
    cudaLbm.SetOmega(omega);
}

void CheckGLError()
{
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cout << "OpenGL error: " << err << std::endl;
    }
}

void UpdateWindowTitle(const int fps, Domain &domain)
{
    char fpsReport[256];
    int xDim = domain.GetXDim();
    int yDim = domain.GetYDim();
    sprintf(fpsReport, 
        "Interactive CFD running at: %i timesteps/frame at %3.1f fps = %3.1f timesteps/second on %ix%i mesh",
        TIMESTEPS_PER_FRAME, fps, TIMESTEPS_PER_FRAME*fps, xDim, yDim);
    glutSetWindowTitle(fpsReport);
}

void Draw()
{
    g_fpsTracker.Tick();
    
    GraphicsManager* graphicsManager = Window.GetPanel("Graphics")->GetGraphicsManager();
    CudaLbm* cudaLbm = graphicsManager->GetCudaLbm();
    graphicsManager->UpdateViewTransformations();
    graphicsManager->UpdateGraphicsInputs();

    Resize(Window.GetWidth(), Window.GetHeight());

    graphicsManager->CenterGraphicsViewToGraphicsPanel(g_leftPanelWidth);

    graphicsManager->GetCudaLbm()->UpdateDeviceImage();
    graphicsManager->RunCuda();

    bool renderFloor = graphicsManager->ShouldRenderFloor();
    Graphics* graphics = graphicsManager->GetGraphics();
    Domain domain = *cudaLbm->GetDomain();
    graphics->RenderVbo(renderFloor, domain);

    CheckGLError();

    Draw2D(Window);

    glutSwapBuffers();

    //Compute and display FPS
    g_fpsTracker.Tock();
    float fps = g_fpsTracker.GetFps();
    UpdateWindowTitle(fps, domain);
}

int main(int argc,char **argv)
{
    SetUpWindow(Window);

    glutInit(&argc,argv);

    glutInitDisplayMode(GLUT_RGB|GLUT_DEPTH|GLUT_DOUBLE);
    int windowWidth = Window.GetWidth();
    int windowHeight = Window.GetHeight();
    glutInitWindowSize(windowWidth,windowHeight);
    glutInitWindowPosition(200,100);
    glutCreateWindow("Interactive CFD");

    glutDisplayFunc(Draw);
    glutReshapeFunc(Resize);
    glutMouseFunc(MouseButton);
    glutMotionFunc(MouseMotion);
    glutKeyboardFunc(Keyboard);
    glutMouseWheelFunc(MouseWheel);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    Init();
    SetUpGLInterop(Window);
    SetUpCUDA(Window);

    glutMainLoop();

    return 0;
}