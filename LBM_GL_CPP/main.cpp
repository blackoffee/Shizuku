#include <GLEW/glew.h>
#include <GLUT/freeglut.h>
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_cuda_gl.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <stdio.h>
#include <iostream>
#include <ostream>
#include <fstream>
#include <time.h>
#include <algorithm>

#include "kernel.h"
#include "Mouse.h"
#include "Panel.h"
#include "common.h"
#include "Domain.h"
#include "FpsTracker.h"
#include "Shader.h"

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

const int g_leftPanelWidth(350);
const int g_leftPanelHeight(500);

FpsTracker g_fpsTracker;

//simulation inputs
Domain g_simDomain;

Obstruction g_obstructions[MAXOBSTS];

Panel Window;
Mouse theMouse;

ButtonGroup contourButtons;
ButtonGroup shapeButtons;
ButtonGroup viewButtons;

//GL buffers
GLuint g_vaoSolutionField;
GLuint g_vboSolutionField;
GLuint g_elementArrayIndexBuffer;
cudaGraphicsResource *g_cudaSolutionField;

float* g_fA_d;
float* g_fB_d;
float* g_floor_d;
int* g_im_d;
Obstruction* g_obst_d;

const int g_glutMouseYOffset = 10; //hack to get better mouse precision

ShaderProgram g_shader;


// forward declarations
void SetUpButtons();
void WaterRenderingButtonCallBack();
void SquareButtonCallBack();
void ThreeDButtonCallBack();

void UpdateWindowDimensionsBasedOnAspectRatio(int& heightOut, int& widthOut, int area,
    int leftPanelHeight, int leftPanelWidth, int xDim, int yDim, float scaleUp);
void UpdateDomainDimensionsBasedOnWindowSize(int leftPanelHeight, int leftPanelWidth,
    int windowWidth, int windowHeight, float scaleUp);

void Init()
{
    glEnable(GL_LIGHT0);
    glewInit();
    int windowWidth = Window.GetWidth();
    int windowHeight = Window.GetHeight();
    glViewport(0,0,windowWidth,windowHeight);

}


void CompileShaderPrograms()
{
    g_shader.Init();
    g_shader.CreateShader("VertexShader.glsl", GL_VERTEX_SHADER);
    g_shader.CreateShader("FragmentShader.glsl", GL_FRAGMENT_SHADER);
}



void SetUpWindow()
{
    int windowWidth = 1200;
    int windowHeight = g_leftPanelHeight+100;

    Window.SetSize_Absolute(RectInt(200, 100, windowWidth, windowHeight));
    Window.m_draw = false;
    Window.SetName("Main Window");
    theMouse.SetBasePanel(&Window);

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
    Window.GetPanel("Graphics")->m_graphicsManager->SetObstructionsPointer(&g_obstructions[0]);
    float scaleUp = Window.GetPanel("Graphics")->m_graphicsManager->GetScaleFactor();

    UpdateDomainDimensionsBasedOnWindowSize(g_leftPanelHeight, g_leftPanelWidth,
        windowWidth, windowHeight, scaleUp);


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

    Window.GetPanel("Label_InletV")->m_displayText = "Inlet Velocity";
    Window.GetSlider("Slider_InletV")->CreateSliderBar(RectFloat(0.7f, -sliderBarH*0.5f, sliderBarW, sliderBarH),
        Panel::DEF_REL, "SliderBar_InletV", Color(Color::GRAY));
    Window.GetSlider("Slider_InletV")->m_maxValue = 0.125f;
    Window.GetSlider("Slider_InletV")->m_minValue = 0.f;
    Window.GetSlider("Slider_InletV")->m_sliderBar1->UpdateValue();

    Window.GetPanel("Label_Visc")->m_displayText = "Viscosity";
    Window.GetSlider("Slider_Visc")->CreateSliderBar(RectFloat(-0.85f, -sliderBarH*0.5f, sliderBarW, sliderBarH),
        Panel::DEF_REL, "SliderBar_Visc", Color(Color::GRAY));
    Window.GetSlider("Slider_Visc")->m_maxValue = 1.8f;
    Window.GetSlider("Slider_Visc")->m_minValue = 1.99f;
    Window.GetSlider("Slider_Visc")->m_sliderBar1->UpdateValue();

    Window.GetPanel("Label_Resolution")->m_displayText = "Resolution";
    Window.GetSlider("Slider_Resolution")->CreateSliderBar(RectFloat(-0.3f, -sliderBarH*0.5f, sliderBarW, sliderBarH),
        Panel::DEF_REL, "SliderBar_Resolution", Color(Color::GRAY));
    Window.GetSlider("Slider_Resolution")->m_maxValue = 1.f;
    Window.GetSlider("Slider_Resolution")->m_minValue = 6.f;
    Window.GetSlider("Slider_Resolution")->m_sliderBar1->UpdateValue();


    std::string VarName = "Velocity Magnitude";
    std::string labelName = "Label_"+VarName;
    std::string sliderName = VarName;
    std::string sliderBarName1 = VarName+"Max";
    std::string sliderBarName2 = VarName+"Min";
    RectFloat contourSliderPosition{-0.9f, 0.2f+0.16f+(0.64f-sliderH*2)*0.5f, 1.8f, sliderH};
    outputsPanel->CreateSubPanel(RectFloat{-0.9f, 0.2f+0.16f+(0.64f-sliderH*2)*0.5f+sliderH, 0.5f, sliderH}
        , Panel::DEF_REL, "Label_Contour", Color(Color::DARK_GRAY));
    Window.GetPanel("Label_Contour")->m_displayText = "Contour Color";
    float contourSliderBarWidth = 0.1f;
    float contourSliderBarHeight = 2.f;
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-1.f, -1, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.65f, -1, contourSliderBarWidth, contourSliderBarHeight),
        Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    Window.GetSlider(sliderName)->m_maxValue = INITIAL_UMAX*2.f;
    Window.GetSlider(sliderName)->m_minValue = 0.f;
    Window.GetSlider(sliderName)->m_sliderBar1->m_foregroundColor = Color::BLUE;
    Window.GetSlider(sliderName)->m_sliderBar2->m_foregroundColor = Color::WHITE;
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
    Window.GetSlider(sliderName)->m_maxValue = INITIAL_UMAX*1.8f;
    Window.GetSlider(sliderName)->m_minValue = -INITIAL_UMAX*1.f;
    Window.GetSlider(sliderName)->m_sliderBar1->m_foregroundColor = Color::BLUE;
    Window.GetSlider(sliderName)->m_sliderBar2->m_foregroundColor = Color::WHITE;
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
    Window.GetSlider(sliderName)->m_maxValue = INITIAL_UMAX*1.f;
    Window.GetSlider(sliderName)->m_minValue = -INITIAL_UMAX*1.f;
    Window.GetSlider(sliderName)->m_sliderBar1->m_foregroundColor = Color::BLUE;
    Window.GetSlider(sliderName)->m_sliderBar2->m_foregroundColor = Color::WHITE;
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
    Window.GetSlider(sliderName)->m_maxValue = INITIAL_UMAX*0.1f;
    Window.GetSlider(sliderName)->m_minValue = 0.f;
    Window.GetSlider(sliderName)->m_sliderBar1->m_foregroundColor = Color::BLUE;
    Window.GetSlider(sliderName)->m_sliderBar2->m_foregroundColor = Color::WHITE;
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
    Window.GetSlider(sliderName)->m_maxValue = 1.05f;
    Window.GetSlider(sliderName)->m_minValue = 0.95f;
    Window.GetSlider(sliderName)->m_sliderBar1->m_foregroundColor = Color::BLUE;
    Window.GetSlider(sliderName)->m_sliderBar2->m_foregroundColor = Color::WHITE;
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
    Window.GetSlider(sliderName)->m_maxValue = 1.05f;
    Window.GetSlider(sliderName)->m_minValue = 0.95f;
    Window.GetSlider(sliderName)->m_sliderBar1->m_foregroundColor = Color::BLUE;
    Window.GetSlider(sliderName)->m_sliderBar2->m_foregroundColor = Color::WHITE;
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
    Window.GetPanel("Label_Size")->m_displayText = "Size";

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
    Window.GetSlider("Slider_Size")->m_maxValue = 15.f;
    Window.GetSlider("Slider_Size")->m_minValue = 1.f;
    Window.GetSlider("Slider_Size")->m_sliderBar1->UpdateValue();
    float currentObstSize = Window.GetSlider("Slider_Size")->m_sliderBar1->GetValue();
    Window.GetPanel("Graphics")->m_graphicsManager->SetCurrentObstSize(currentObstSize);


    SetUpButtons();
    WaterRenderingButtonCallBack(); //default is water rendering
    SquareButtonCallBack(); //default is square shape
    ThreeDButtonCallBack();
}

/*----------------------------------------------------------------------------------------
 *	Button setup
 */

Slider* GetCurrentContourSlider()
{
    if (Window.GetSlider("Velocity Magnitude")->m_draw == true)
        return Window.GetSlider("Velocity Magnitude");
    else if (Window.GetSlider("X Velocity")->m_draw == true)
        return Window.GetSlider("X Velocity");
    else if (Window.GetSlider("Y Velocity")->m_draw == true) 
        return Window.GetSlider("Y Velocity");
    else if (Window.GetSlider("StrainRate")->m_draw == true) 
        return Window.GetSlider("StrainRate");
    else if (Window.GetSlider("Pressure")->m_draw == true) 
        return Window.GetSlider("Pressure");
    else if (Window.GetSlider("Water Rendering")->m_draw == true) 
        return Window.GetSlider("Water Rendering");
}


void InitializeButtonCallBack()
{
    float4 *dptr;
    cudaGraphicsMapResources(1, &g_cudaSolutionField, 0);
    size_t num_bytes,num_bytes2;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, g_cudaSolutionField);

    float u = Window.GetSlider("Slider_InletV")->m_sliderBar1->GetValue();
    InitializeDomain(dptr, g_fA_d, g_im_d, u, g_simDomain);
}

void VelMagButtonCallBack()
{
    contourButtons.ExclusiveEnable(Window.GetButton("Velocity Magnitude"));
    Window.GetPanel("Graphics")->m_graphicsManager->SetContourVar(VEL_MAG);
}

void VelXButtonCallBack()
{
    contourButtons.ExclusiveEnable(Window.GetButton("X Velocity"));
    Window.GetPanel("Graphics")->m_graphicsManager->SetContourVar(VEL_U);
}

void VelYButtonCallBack()
{
    contourButtons.ExclusiveEnable(Window.GetButton("Y Velocity"));
    Window.GetPanel("Graphics")->m_graphicsManager->SetContourVar(VEL_V);
}

void StrainRateButtonCallBack()
{
    contourButtons.ExclusiveEnable(Window.GetButton("StrainRate"));
    Window.GetPanel("Graphics")->m_graphicsManager->SetContourVar(STRAIN_RATE);
}

void PressureButtonCallBack()
{
    contourButtons.ExclusiveEnable(Window.GetButton("Pressure"));
    Window.GetPanel("Graphics")->m_graphicsManager->SetContourVar(PRESSURE);
}

void WaterRenderingButtonCallBack()
{
    contourButtons.ExclusiveEnable(Window.GetButton("Water Rendering"));
    Window.GetPanel("Graphics")->m_graphicsManager->SetContourVar(WATER_RENDERING);
}

void SquareButtonCallBack()
{
    shapeButtons.ExclusiveEnable(Window.GetButton("Square"));
    Window.GetPanel("Graphics")->m_graphicsManager->SetCurrentObstShape(Obstruction::SQUARE);
}

void CircleButtonCallBack()
{
    shapeButtons.ExclusiveEnable(Window.GetButton("Circle"));
    Window.GetPanel("Graphics")->m_graphicsManager->SetCurrentObstShape(Obstruction::CIRCLE);
}

void HorLineButtonCallBack()
{
    shapeButtons.ExclusiveEnable(Window.GetButton("Hor. Line"));
    Window.GetPanel("Graphics")->m_graphicsManager->SetCurrentObstShape(Obstruction::HORIZONTAL_LINE);
}

void VertLineButtonCallBack()
{
    shapeButtons.ExclusiveEnable(Window.GetButton("Vert. Line"));
    Window.GetPanel("Graphics")->m_graphicsManager->SetCurrentObstShape(Obstruction::VERTICAL_LINE);
}

void ThreeDButtonCallBack()
{
    viewButtons.ExclusiveEnable(Window.GetButton("3D"));
    Window.GetPanel("Graphics")->m_graphicsManager->SetViewMode(THREE_DIMENSIONAL);
}

void TwoDButtonCallBack()
{
    viewButtons.ExclusiveEnable(Window.GetButton("2D"));
    Window.GetPanel("Graphics")->m_graphicsManager->SetViewMode(TWO_DIMENSIONAL);
}

void SetUpButtons()
{
    Window.GetButton("Initialize")->m_callBack = InitializeButtonCallBack;
    Window.GetButton("Velocity Magnitude")->m_callBack = VelMagButtonCallBack;
    Window.GetButton("X Velocity")->m_callBack = VelXButtonCallBack;
    Window.GetButton("Y Velocity")->m_callBack = VelYButtonCallBack;
    Window.GetButton("StrainRate")->m_callBack = StrainRateButtonCallBack;
    Window.GetButton("Pressure"  )->m_callBack = PressureButtonCallBack;
    Window.GetButton("Water Rendering")->m_callBack = WaterRenderingButtonCallBack;

    std::vector<Button*> buttons = {
        Window.GetButton("Velocity Magnitude"),
        Window.GetButton("X Velocity"),
        Window.GetButton("Y Velocity"),
        Window.GetButton("StrainRate"),
        Window.GetButton("Pressure"),
        Window.GetButton("Water Rendering") };
    contourButtons = ButtonGroup(buttons);

    //Shape buttons
    Window.GetButton("Square")->m_callBack = SquareButtonCallBack;
    Window.GetButton("Circle")->m_callBack = CircleButtonCallBack;
    Window.GetButton("Hor. Line")->m_callBack = HorLineButtonCallBack;
    Window.GetButton("Vert. Line")->m_callBack = VertLineButtonCallBack;

    std::vector<Button*> buttons2 = {
        Window.GetButton("Square"),
        Window.GetButton("Circle"),
        Window.GetButton("Hor. Line"),
        Window.GetButton("Vert. Line") };
    shapeButtons = ButtonGroup(buttons2);

    Window.GetButton("3D")->m_callBack = ThreeDButtonCallBack;
    Window.GetButton("2D")->m_callBack = TwoDButtonCallBack;
    
    std::vector<Button*> buttons3 = {
        Window.GetButton("2D"),
        Window.GetButton("3D")
    };
    viewButtons = ButtonGroup(buttons3);

}

void DrawShapePreview()
{
    Panel* previewPanel = Window.GetPanel("DrawingPreview");
    float centerX = previewPanel->GetRectFloatAbs().GetCentroidX();
    float centerY = previewPanel->GetRectFloatAbs().GetCentroidY();
    int windowWidth = Window.GetWidth();
    int windowHeight = Window.GetHeight();
    float graphicsToWindowScaleFactor = static_cast<float>(windowWidth)/
        Window.GetPanel("Graphics")->GetRectIntAbs().m_w;

    int xDimVisible = g_simDomain.GetXDimVisible();
    int yDimVisible = g_simDomain.GetYDimVisible();
    float currentSize = Window.GetSlider("Slider_Size")->m_sliderBar1->GetValue();
    int graphicsWindowWidth = Window.GetPanel("Graphics")->GetRectIntAbs().m_w;
    int graphicsWindowHeight = Window.GetPanel("Graphics")->GetRectIntAbs().m_h;
    int r1ix = currentSize*static_cast<float>(graphicsWindowWidth) / (xDimVisible); //r1x in pixels
    int r1iy = currentSize*static_cast<float>(graphicsWindowHeight) / (yDimVisible); //r1x in pixels
    float r1fx = static_cast<float>(r1ix) / windowWidth*2.f;
    float r1fy = static_cast<float>(r1iy) / windowHeight*2.f;

    GraphicsManager *graphicsManager = Window.GetPanel("Graphics")->m_graphicsManager;
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
void CreateVBO(GLuint *vbo, cudaGraphicsResource **vbo_res, unsigned int size, unsigned int vbo_res_flags)
{

    glGenVertexArrays(1, &g_vaoSolutionField);
    glBindVertexArray(g_vaoSolutionField);

    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);
}

void DeleteVBO(GLuint *vbo, cudaGraphicsResource *vbo_res)
{
    cudaGraphicsUnregisterResource(vbo_res);
    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}


void GenerateIndexListForSurfaceAndFloor(GLuint &arrayIndexBuffer)
{
    int numberOfElements = (MAX_XDIM - 1)*(MAX_YDIM - 1);
    int numberOfNodes = MAX_XDIM*MAX_YDIM;
    GLuint* elementIndices = new GLuint[numberOfElements * 4 * 2];
    for (int j = 0; j < MAX_YDIM-1; j++){
        for (int i = 0; i < MAX_XDIM-1; i++){
            //going clockwise, since y orientation will be flipped when rendered
            elementIndices[j*(MAX_XDIM-1)*4+i*4+0] = (i)+(j)*MAX_XDIM;
            elementIndices[j*(MAX_XDIM-1)*4+i*4+1] = (i+1)+(j)*MAX_XDIM;
            elementIndices[j*(MAX_XDIM-1)*4+i*4+2] = (i+1)+(j+1)*MAX_XDIM;
            elementIndices[j*(MAX_XDIM-1)*4+i*4+3] = (i)+(j+1)*MAX_XDIM;
        }
    }
    for (int j = 0; j < MAX_YDIM-1; j++){
        for (int i = 0; i < MAX_XDIM-1; i++){
            //going clockwise, since y orientation will be flipped when rendered
            elementIndices[numberOfElements*4+j*(MAX_XDIM-1)*4+i*4+0] = numberOfNodes+(i)+(j)*MAX_XDIM;
            elementIndices[numberOfElements*4+j*(MAX_XDIM-1)*4+i*4+1] = numberOfNodes+(i+1)+(j)*MAX_XDIM;
            elementIndices[numberOfElements*4+j*(MAX_XDIM-1)*4+i*4+2] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;
            elementIndices[numberOfElements*4+j*(MAX_XDIM-1)*4+i*4+3] = numberOfNodes+(i)+(j+1)*MAX_XDIM;
        }
    }

    glGenBuffers(1, &arrayIndexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, arrayIndexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*numberOfElements*4*2, elementIndices, GL_DYNAMIC_DRAW);
    free(elementIndices);
}

void CleanUpIndexList(GLuint &arrayIndexBuffer){
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &arrayIndexBuffer);
}


void SetUpGLInterop()
{
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
    GenerateIndexListForSurfaceAndFloor(g_elementArrayIndexBuffer);
    unsigned int solutionMemorySize = MAX_XDIM*MAX_YDIM * 4 * sizeof(float);
    unsigned int floorSize = MAX_XDIM*MAX_YDIM * 4 * sizeof(float);
    CreateVBO(&g_vboSolutionField, &g_cudaSolutionField, solutionMemorySize+floorSize,
        cudaGraphicsMapFlagsWriteDiscard);

}

void CleanUpGLInterop()
{
    CleanUpIndexList(g_elementArrayIndexBuffer);
    DeleteVBO(&g_vboSolutionField, g_cudaSolutionField);
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

//BC function for host side
int ImageFcn_h(int x, int y, Obstruction* obstructions){
    int xDim = g_simDomain.GetXDim();
    int yDim = g_simDomain.GetYDim();
    //if(y == 0 || x == XDIM-1 || y == YDIM-1)
    if (x < 0.1f)
        return 3;//west
    else if ((xDim - x) < 1.1f)
        return 2;//east
    else if ((yDim - y) < 1.1f)
        return 11;//11;//xsymmetry top
    else if (y < 0.1f)
        return 12;//12;//xsymmetry bottom
    return 0;
}

void UpdateDeviceImage()
{
    int domainSize = ceil(MAX_XDIM / BLOCKSIZEX)*BLOCKSIZEX*ceil(MAX_YDIM / BLOCKSIZEY)*BLOCKSIZEY;
    int* im_h = new int[domainSize];
    for (int i = 0; i < domainSize; i++)
    {
        int x = i%MAX_XDIM;
        int y = i/MAX_XDIM;
        im_h[i] = ImageFcn_h(x, y, g_obstructions);
    }
    size_t memsize_int = domainSize*sizeof(int);
    cudaMemcpy(g_im_d, im_h, memsize_int, cudaMemcpyHostToDevice);
    delete[] im_h;
}

void SetUpCUDA()
{
    size_t memsize, memsize_int, memsize_float, memsize_inputs;

    int domainSize = ceil(MAX_XDIM / BLOCKSIZEX)*BLOCKSIZEX*ceil(MAX_YDIM / BLOCKSIZEY)*BLOCKSIZEY;
    memsize = domainSize*sizeof(float)*9;
    memsize_int = domainSize*sizeof(int);
    memsize_float = domainSize*sizeof(float);
    memsize_inputs = sizeof(g_obstructions);

    float* fA_h = new float[domainSize * 9];
    float* fB_h = new float[domainSize * 9];
    float* floor_h = new float[domainSize];
    int* im_h = new int[domainSize];
    float4 rayCastIntersect{ 0, 0, 0, 1e6 };

    cudaMalloc((void **)&g_fA_d, memsize);
    cudaMalloc((void **)&g_fB_d, memsize);
    cudaMalloc((void **)&g_floor_d, memsize_float);
    cudaMalloc((void **)&g_im_d, memsize_int);
    cudaMalloc((void **)&g_obst_d, memsize_inputs);
    GraphicsManager* graphicsManager = Window.GetPanel("Graphics")->m_graphicsManager;
    cudaMalloc((void **)&graphicsManager->m_rayCastIntersect_d, sizeof(float4));

    for (int i = 0; i < domainSize*9; i++)
    {
        fA_h[i] = 0;
        fB_h[i] = 0;
    }
    for (int i = 0; i < MAXOBSTS; i++)
    {
        g_obstructions[i].r1 = 0;
        g_obstructions[i].x = 0;
        g_obstructions[i].y = -1000;
        g_obstructions[i].state = Obstruction::REMOVED;
    }	
    g_obstructions[0].r1 = 6.5;
    g_obstructions[0].x = 30;// g_xDim*0.2f;
    g_obstructions[0].y = 42;// g_yDim*0.3f;
    g_obstructions[0].u = 0;// g_yDim*0.3f;
    g_obstructions[0].v = 0;// g_yDim*0.3f;
    g_obstructions[0].shape = Obstruction::SQUARE;
    g_obstructions[0].state = Obstruction::NEW;

    g_obstructions[1].r1 = 4.5;
    g_obstructions[1].x = 30;// g_xDim*0.2f;
    g_obstructions[1].y = 100;// g_yDim*0.3f;
    g_obstructions[1].u = 0;// g_yDim*0.3f;
    g_obstructions[1].v = 0;// g_yDim*0.3f;
    g_obstructions[1].shape = Obstruction::VERTICAL_LINE;
    g_obstructions[1].state = Obstruction::NEW;

    for (int i = 0; i < domainSize; i++)
    {
        floor_h[i] = 0;
    }

    UpdateDeviceImage();
    
    cudaMemcpy(g_fA_d, fA_h, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(g_fB_d, fB_h, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(g_floor_d, floor_h, memsize_float, cudaMemcpyHostToDevice);
    cudaMemcpy(g_obst_d, g_obstructions, memsize_inputs, cudaMemcpyHostToDevice);
    cudaMemcpy(graphicsManager->m_rayCastIntersect_d, &rayCastIntersect, sizeof(float4), cudaMemcpyHostToDevice);

    delete[] fA_h;
    delete[] fB_h;
    delete[] floor_h;

    //writeInputs();
    float u = Window.GetSlider("Slider_InletV")->m_sliderBar1->GetValue();

    float4 *dptr;
    cudaGraphicsMapResources(1, &g_cudaSolutionField, 0);
    size_t num_bytes,num_bytes2;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, g_cudaSolutionField);

    InitializeDomain(dptr, g_fA_d, g_im_d, u, g_simDomain);
    InitializeDomain(dptr, g_fB_d, g_im_d, u, g_simDomain);

    InitializeFloor(dptr, g_floor_d, g_simDomain);

    cudaGraphicsUnmapResources(1, &g_cudaSolutionField, 0);

}

void RunCuda(struct cudaGraphicsResource **vbo_resource, float3 cameraPosition)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    cudaGraphicsMapResources(1, &g_cudaSolutionField, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, g_cudaSolutionField);

    float u = Window.GetSlider("Slider_InletV")->m_sliderBar1->GetValue();
    float omega = Window.GetSlider("Slider_Visc")->m_sliderBar1->GetValue();
    float contMin = GetCurrentContourSlider()->m_sliderBar1->GetValue();
    float contMax = GetCurrentContourSlider()->m_sliderBar2->GetValue();
    bool paused = Window.GetPanel("Graphics")->m_graphicsManager->IsPaused();
    ContourVariable contourVar = Window.GetPanel("Graphics")->m_graphicsManager->GetContourVar();
    ViewMode viewMode = Window.GetPanel("Graphics")->m_graphicsManager->GetViewMode();

    MarchSolution(g_fA_d, g_fB_d, g_im_d, g_obst_d, u, omega, TIMESTEPS_PER_FRAME/2, 
        g_simDomain, paused);
    UpdateSolutionVbo(dptr, g_fB_d, g_im_d, contourVar, contMin, contMax, viewMode,
        u, g_simDomain);
 
    SetObstructionVelocitiesToZero(g_obstructions, g_obst_d);

    if (viewMode == ViewMode::THREE_DIMENSIONAL || contourVar == ContourVariable::WATER_RENDERING)
    {
        LightSurface(dptr, g_obst_d, cameraPosition, g_simDomain);
    }
    LightFloor(dptr, g_floor_d, g_obst_d,cameraPosition, g_simDomain);
    CleanUpDeviceVBO(dptr, g_simDomain);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, &g_cudaSolutionField, 0);

}


void Draw2D()
{
    Window.DrawAll();
    DrawShapePreview();
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
        GraphicsManager *graphicsManager = Window.GetPanel("Graphics")->m_graphicsManager;
        graphicsManager->TogglePausedState();
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

void UpdateDomainDimensionsBasedOnWindowSize(int leftPanelHeight, int leftPanelWidth,
    int windowWidth, int windowHeight, float scaleUp)
{
    int xDimVisible = static_cast<float>(windowWidth - leftPanelWidth) / scaleUp;
    int yDimVisible = ceil(static_cast<float>(windowHeight) / scaleUp);
    g_simDomain.SetXDimVisible(xDimVisible);
    g_simDomain.SetYDimVisible(yDimVisible);
}

void Resize(int windowWidth, int windowHeight)
{
    float scaleUp = Window.GetPanel("Graphics")->m_graphicsManager->GetScaleFactor();
    UpdateDomainDimensionsBasedOnWindowSize(g_leftPanelHeight, g_leftPanelWidth,
        windowWidth, windowHeight, scaleUp);

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

    UpdateDeviceImage();

}

void Draw()
{
    g_fpsTracker.Tick();

    float scaleUp = Window.GetSlider("Slider_Resolution")->m_sliderBar1->GetValue();
    GraphicsManager *graphicsManager = Window.GetPanel("Graphics")->m_graphicsManager;
    graphicsManager->SetScaleFactor(scaleUp);

    int windowWidth = Window.GetWidth();
    int windowHeight = Window.GetHeight();
    Resize(windowWidth, windowHeight);

    int xDimVisible = g_simDomain.GetXDimVisible();
    int yDimVisible = g_simDomain.GetYDimVisible();
    float xTranslation = -((static_cast<float>(windowWidth)-xDimVisible*scaleUp)*0.5
        - static_cast<float>(g_leftPanelWidth)) / windowWidth*2.f;
    float yTranslation = -((static_cast<float>(windowHeight)-yDimVisible*scaleUp)*0.5)
        / windowHeight*2.f;

    //get view transformations
    float3 translateTransforms = graphicsManager->GetTranslationTransforms();
    float3 rotateTransforms = graphicsManager->GetRotationTransforms();
    float3 cameraPosition = { -xTranslation - translateTransforms.x, 
        -yTranslation - translateTransforms.y, +2 - translateTransforms.z };

    RunCuda(&g_cudaSolutionField, cameraPosition);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glTranslatef(xTranslation,yTranslation,0.f);
    glScalef((static_cast<float>(xDimVisible*scaleUp) / windowWidth),
        (static_cast<float>(yDimVisible*scaleUp) / windowHeight), 1.f);

    ViewMode viewMode = graphicsManager->GetViewMode();
    if (viewMode == ViewMode::TWO_DIMENSIONAL)
    {
        glOrtho(-1,1,-1,static_cast<float>(yDimVisible)/xDimVisible*2.f-1.f,-100,20);
    }
    else
    {
        gluPerspective(45.0, static_cast<float>(xDimVisible) / yDimVisible, 0.1, 10.0);
        glTranslatef(translateTransforms.x, translateTransforms.y, -2+translateTransforms.z);
        glRotatef(-rotateTransforms.x,1,0,0);
        glRotatef(rotateTransforms.z,0,0,1);
    }

    // Update transformation matrices in graphics manager for mouse ray casting
    graphicsManager->UpdateViewTransformations();
    // Update Obstruction size based on current slider value
    float currentObstSize = Window.GetSlider("Slider_Size")->m_sliderBar1->GetValue();
    graphicsManager->SetCurrentObstSize(currentObstSize);

    glm::vec4 viewportMatrix = graphicsManager->GetViewportMatrix();
    glm::mat4 modelMatrix = graphicsManager->GetModelMatrix();
    glm::mat4 projectionMatrix = graphicsManager->GetProjectionMatrix();



    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //Draw solution field
    glBindVertexArray(g_vaoSolutionField);
    glBindBuffer(GL_ARRAY_BUFFER, g_vboSolutionField);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_elementArrayIndexBuffer);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, 0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 16, (GLvoid*)(3*sizeof(GLfloat)));
    glEnableVertexAttribArray(1);

    //glVertexPointer(3, GL_FLOAT, 16, 0);
    //glEnableClientState(GL_VERTEX_ARRAY);
    //glColor3f(1.0, 0.0, 0.0);
    //glEnableClientState(GL_COLOR_ARRAY);
    //glColorPointer(4, GL_UNSIGNED_BYTE, 16, (char *)NULL + 12);
    ContourVariable contourVar = graphicsManager->GetContourVar();

    g_shader.Use();
    GLint viewportMatrixLocation = glGetUniformLocation(g_shader.ProgramID, "viewportMatrix");
    GLint modelMatrixLocation = glGetUniformLocation(g_shader.ProgramID, "modelMatrix");
    GLint projectionMatrixLocation = glGetUniformLocation(g_shader.ProgramID, "projectionMatrix");
    glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    if (viewMode == ViewMode::THREE_DIMENSIONAL || contourVar == ContourVariable::WATER_RENDERING)
    {
        //Draw floor
        glDrawElements(GL_QUADS, (MAX_XDIM - 1)*(yDimVisible - 1)*4, GL_UNSIGNED_INT, 
            BUFFER_OFFSET(sizeof(GLuint)*4*(MAX_XDIM - 1)*(MAX_YDIM - 1)));
    }
    //Draw water surface
    glDrawElements(GL_QUADS, (MAX_XDIM - 1)*(yDimVisible - 1)*4 , GL_UNSIGNED_INT, (GLvoid*)0);
    g_shader.Unset();
    //glDisableClientState(GL_VERTEX_ARRAY);
    glBindVertexArray(0);

    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cout << "OpenGL error: " << err << std::endl;
    }



    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1,1,-1,1,-100,20);

    Draw2D();

    glutSwapBuffers();

    //Compute and display FPS
    g_fpsTracker.Tock();
    float fps = g_fpsTracker.GetFps();
    char fpsReport[256];
    int xDim = g_simDomain.GetXDim();
    int yDim = g_simDomain.GetYDim();
    sprintf(fpsReport, 
        "Interactive CFD running at: %i timesteps/frame at %3.1f fps = %3.1f timesteps/second on %ix%i mesh",
        TIMESTEPS_PER_FRAME, fps, TIMESTEPS_PER_FRAME*fps, xDim, yDim);
    glutSetWindowTitle(fpsReport);


}

int main(int argc,char **argv)
{
    SetUpWindow();

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

    CompileShaderPrograms();

    SetUpGLInterop();
    SetUpCUDA();

    glutMainLoop();

    return 0;
}