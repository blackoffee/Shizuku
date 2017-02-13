#include <GL/glew.h>
#include <GL/freeglut.h>
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

#include "kernel.h"
#include "Mouse.h"
#include "Panel.h"
#include "common.h"
#include "SimulationParameters.h"
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

int winw, winh;
const int g_leftPanelWidth(350);
const int g_drawingPanelWidth(0);
const int g_leftPanelHeight(500);
const int g_drawingPanelHeight(0);
float g_initialScaleUp(1.f); 

int g_tStep = 15; //initial tstep value before adjustments
int g_fpsCount = 0;
int g_fpsLimit = 20;
float g_fps = 0;
float g_timeStepsPerSecond = 0;
clock_t g_timeBefore;


//simulation inputs
SimulationParameters g_simParams;
float g_uMax = 0.125f;

//view states
float g_contMin = 0.f;
float g_contMax = 0.1f;
ContourVariable g_contourVar;
ViewMode g_viewMode;


float4 d_rayCastIntersect;
float4 *d_rayCastIntersect_d;


//view transformations
float rotate_x = 60.f;
float rotate_z = 30.f;
float translate_x = 0.f;
float translate_y = 0.8f;
float translate_z = -0.2f;
int g_paused = 0;


Obstruction g_obstructions[MAXOBSTS];

Panel Window;
Mouse theMouse;

ButtonGroup contourButtons;
ButtonGroup shapeButtons;
ButtonGroup viewButtons;

//drawing modes
Obstruction::Shape g_currentShape=Obstruction::CIRCLE;
float g_currentSize = 5.f;

//GL buffers
GLuint g_vboSolutionField;
GLuint g_elementArrayIndexBuffer;
GLuint g_vboFloor;
GLuint g_elementArrayIndexFloorBuffer;
GLuint g_floorFrameBuffer;
GLuint g_floorTexture;
cudaGraphicsResource *g_cudaSolutionField;
cudaGraphicsResource *g_cudaFloor;

float* g_fA_h;
float* g_fA_d;
float* g_floor_h;
float* g_fB_h;
float* g_fB_d;
float* g_floor_d;
float* g_floorFiltered_d;
int* g_im_h;
int* g_im_d;
Obstruction* g_obst_d;

const int g_glutMouseYOffset = 10; //hack to get better mouse precision


// forward declarations
void SetUpButtons();
void WaterRenderingButtonCallBack();
void SquareButtonCallBack();
void ThreeDButtonCallBack();
void Resize(int w, int h);

void UpdateWindowDimensionsBasedOnAspectRatio(int& heightOut, int& widthOut, int area, int leftPanelHeight, int leftPanelWidth, int xDim, int yDim, float scaleUp);
void UpdateDomainDimensionsBasedOnWindowSize(int leftPanelHeight, int leftPanelWidth, int windowWidth, int windowHeight, float scaleUp);

void Init()
{
    glEnable(GL_LIGHT0);
    glewInit();
    glViewport(0,0,winw, winh);

}

void SetUpWindow()
{
    SimulationParameters_init(&g_simParams);

    winw = 1200;// g_xDim*g_initialScaleUp + g_leftPanelWidth;
    winh = g_leftPanelHeight + 100;// max(g_yDim*g_initialScaleUp, static_cast<float>(g_leftPanelHeight + 100));
    UpdateDomainDimensionsBasedOnWindowSize(g_leftPanelHeight, g_leftPanelWidth, winw, winh, g_initialScaleUp);


    Window.m_rectInt_abs = RectInt(200, 100, winw, winh);
    Window.m_rectFloat_abs = Window.RectIntAbsToRectFloatAbs();
    Window.m_draw = false;
    Window.m_name = "Main Window";
    Window.m_sizeDefinition = Panel::DEF_ABS;
    theMouse.SetBasePanel(&Window);
    theMouse.m_simScaleUp = g_initialScaleUp;

    Panel* CDV = Window.CreateSubPanel(RectInt(0, 0, g_leftPanelWidth, g_leftPanelHeight), Panel::DEF_ABS, "CDV", Color(Color::DARK_GRAY));
    Panel* outputsPanel = CDV->CreateSubPanel(RectFloat(-1.f,  -0.9f, 2.f, 0.5f), Panel::DEF_REL, "Outputs", Color(Color::DARK_GRAY));
    Panel* inputsPanel  = CDV->CreateSubPanel(RectFloat(-1.f, -0.4f, 2.f, 0.6f), Panel::DEF_REL, "Inputs", Color(Color::DARK_GRAY));
    Panel* drawingPanel = CDV->CreateSubPanel(RectFloat(-1.f,  0.2f, 2.f, 0.8f), Panel::DEF_REL, "Drawing", Color(Color::DARK_GRAY));
    Panel* viewModePanel = CDV->CreateSubPanel(RectFloat(-1.f,  -1.f, 2.f, 0.1f), Panel::DEF_REL, "ViewMode", Color(Color::DARK_GRAY));


    outputsPanel->CreateButton(RectFloat(-0.9f, -0.2f +0.12f, 0.85f, 0.4f), Panel::DEF_REL, "X Velocity", Color(Color::GRAY));
    outputsPanel->CreateButton(RectFloat(-0.9f, -0.6f +0.08f, 0.85f, 0.4f), Panel::DEF_REL, "Velocity Magnitude", Color(Color::GRAY));
    outputsPanel->CreateButton(RectFloat(-0.9f, -1.f  +0.04f, 0.85f, 0.4f), Panel::DEF_REL, "StrainRate", Color(Color::GRAY));
    outputsPanel->CreateButton(RectFloat(0.05f, -0.2f +0.12f, 0.85f, 0.4f), Panel::DEF_REL, "Y Velocity", Color(Color::GRAY));
    outputsPanel->CreateButton(RectFloat(0.05f, -0.6f +0.08f, 0.85f, 0.4f) ,Panel::DEF_REL, "Pressure"  , Color(Color::GRAY));
    outputsPanel->CreateButton(RectFloat(0.05f, -1.f  +0.04f, 0.85f, 0.4f), Panel::DEF_REL, "Water Rendering", Color(Color::GRAY));

    viewModePanel->CreateButton(RectFloat(-0.9f , -1.f  +0.04f, 0.35f, 2.f), Panel::DEF_REL, "3D", Color(Color::GRAY));
    viewModePanel->CreateButton(RectFloat(-0.50f, -1.f  +0.04f, 0.35f, 2.f), Panel::DEF_REL, "2D", Color(Color::GRAY));

    Window.CreateSubPanel(RectInt(g_leftPanelWidth, 0, winw-g_leftPanelWidth, winh), Panel::DEF_ABS, "Graphics", Color(Color::RED));
    Window.GetPanel("Graphics")->m_draw = false;
    Window.GetPanel("Graphics")->CreateGraphicsManager();
    Window.GetPanel("Graphics")->m_graphicsManager->m_obstructions = &g_obstructions[0];


    float sliderH = 1.4f/3.f/2.f;
    float sliderBarW = 0.1f;
    float sliderBarH = 2.f;

    inputsPanel->CreateSubPanel(RectFloat(-0.9f, -1.f+0.4f+0.16f+sliderH*5, 0.5f, sliderH), Panel::DEF_REL, "Label_InletV", Color(Color::DARK_GRAY));
    inputsPanel->CreateSubPanel(RectFloat(-0.9f, -1.f+0.4f+0.12f+sliderH*3, 0.5f, sliderH), Panel::DEF_REL, "Label_Visc", Color(Color::DARK_GRAY));
    inputsPanel->CreateSubPanel(RectFloat(-0.9f, -1.f+0.4f+0.08f+sliderH, 0.5f, sliderH), Panel::DEF_REL, "Label_Resolution", Color(Color::DARK_GRAY));
    inputsPanel->CreateSlider(RectFloat(-0.9f, -1.f+0.4f+0.16f+sliderH*4, 1.8f, sliderH), Panel::DEF_REL, "Slider_InletV", Color(Color::LIGHT_GRAY));
    inputsPanel->CreateSlider(RectFloat(-0.9f, -1.f+0.4f+0.12f+sliderH*2, 1.8f, sliderH), Panel::DEF_REL, "Slider_Visc", Color(Color::LIGHT_GRAY));
    inputsPanel->CreateSlider(RectFloat(-0.9f, -1.f+0.4f+0.08f, 1.8f, sliderH), Panel::DEF_REL, "Slider_Resolution", Color(Color::LIGHT_GRAY));
    inputsPanel->CreateButton(RectFloat(-0.9f, -1.f+0.09f , 1.8f, 0.3f ), Panel::DEF_REL, "Initialize", Color(Color::GRAY));

    Window.GetPanel("Label_InletV")->m_displayText = "Inlet Velocity";
    Window.GetSlider("Slider_InletV")->CreateSliderBar(RectFloat(0.7f, -sliderBarH*0.5f, sliderBarW, sliderBarH), Panel::DEF_REL, "SliderBar_InletV", Color(Color::GRAY));
    Window.GetSlider("Slider_InletV")->m_maxValue = 0.125f;
    Window.GetSlider("Slider_InletV")->m_minValue = 0.f;
    Window.GetSlider("Slider_InletV")->m_sliderBar1->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider("Slider_InletV")->m_sliderBar1->UpdateValue();

    Window.GetPanel("Label_Visc")->m_displayText = "Viscosity";
    Window.GetSlider("Slider_Visc")->CreateSliderBar(RectFloat(-0.85f, -sliderBarH*0.5f, sliderBarW, sliderBarH), Panel::DEF_REL, "SliderBar_Visc", Color(Color::GRAY));
    Window.GetSlider("Slider_Visc")->m_maxValue = 1.8f;
    Window.GetSlider("Slider_Visc")->m_minValue = 1.99f;
    Window.GetSlider("Slider_Visc")->m_sliderBar1->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider("Slider_Visc")->m_sliderBar1->UpdateValue();

    Window.GetPanel("Label_Resolution")->m_displayText = "Resolution";
    Window.GetSlider("Slider_Resolution")->CreateSliderBar(RectFloat(-0.3f, -sliderBarH*0.5f, sliderBarW, sliderBarH), Panel::DEF_REL, "SliderBar_Resolution", Color(Color::GRAY));
    Window.GetSlider("Slider_Resolution")->m_maxValue = 1.f;
    Window.GetSlider("Slider_Resolution")->m_minValue = 6.f;
    Window.GetSlider("Slider_Resolution")->m_sliderBar1->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider("Slider_Resolution")->m_sliderBar1->UpdateValue();


    std::string VarName = "Velocity Magnitude";
    std::string labelName = "Label_"+VarName;
    std::string sliderName = VarName;
    std::string sliderBarName1 = VarName+"Max";
    std::string sliderBarName2 = VarName+"Min";
    RectFloat contourSliderPosition{-0.9f, 0.2f+0.16f+(0.64f-sliderH*2)*0.5f, 1.8f, sliderH};
    outputsPanel->CreateSubPanel(RectFloat{-0.9f, 0.2f+0.16f+(0.64f-sliderH*2)*0.5f+sliderH, 0.5f, sliderH}, Panel::DEF_REL, "Label_Contour", Color(Color::DARK_GRAY));
    Window.GetPanel("Label_Contour")->m_displayText = "Contour Color";
    float contourSliderBarWidth = 0.1f;
    float contourSliderBarHeight = 2.f;
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-1.f, -1, contourSliderBarWidth, contourSliderBarHeight), Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.65f, -1, contourSliderBarWidth, contourSliderBarHeight), Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    Window.GetSlider(sliderName)->m_maxValue = g_uMax*2.f;
    Window.GetSlider(sliderName)->m_minValue = 0.f;
    Window.GetSlider(sliderName)->m_sliderBar1->m_foregroundColor = Color::BLUE;
    Window.GetSlider(sliderName)->m_sliderBar2->m_foregroundColor = Color::WHITE;
    Window.GetSlider(sliderName)->m_sliderBar1->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider(sliderName)->m_sliderBar2->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    Window.GetSlider(sliderName)->m_sliderBar2->UpdateValue();

    VarName = "X Velocity";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.85f, -1.f, contourSliderBarWidth, contourSliderBarHeight), Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.65f, -1.f, contourSliderBarWidth, contourSliderBarHeight), Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    Window.GetSlider(sliderName)->m_maxValue = g_uMax*1.8f;
    Window.GetSlider(sliderName)->m_minValue = -g_uMax*1.f;
    Window.GetSlider(sliderName)->m_sliderBar1->m_foregroundColor = Color::BLUE;
    Window.GetSlider(sliderName)->m_sliderBar2->m_foregroundColor = Color::WHITE;
    Window.GetSlider(sliderName)->m_sliderBar1->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider(sliderName)->m_sliderBar2->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    Window.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    Window.GetSlider(sliderName)->Hide();

    VarName = "Y Velocity";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.65f, -1.f, contourSliderBarWidth, contourSliderBarHeight), Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.65f-contourSliderBarWidth*0.5f, -1.f, contourSliderBarWidth, contourSliderBarHeight), Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    Window.GetSlider(sliderName)->m_maxValue = g_uMax*1.f;
    Window.GetSlider(sliderName)->m_minValue = -g_uMax*1.f;
    Window.GetSlider(sliderName)->m_sliderBar1->m_foregroundColor = Color::BLUE;
    Window.GetSlider(sliderName)->m_sliderBar2->m_foregroundColor = Color::WHITE;
    Window.GetSlider(sliderName)->m_sliderBar1->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider(sliderName)->m_sliderBar2->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    Window.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    Window.GetSlider(sliderName)->Hide();

    VarName = "StrainRate";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.9f, -1.f, contourSliderBarWidth, contourSliderBarHeight), Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(0.35f, -1.f, contourSliderBarWidth, contourSliderBarHeight), Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    Window.GetSlider(sliderName)->m_maxValue = g_uMax*0.1f;
    Window.GetSlider(sliderName)->m_minValue = 0.f;
    Window.GetSlider(sliderName)->m_sliderBar1->m_foregroundColor = Color::BLUE;
    Window.GetSlider(sliderName)->m_sliderBar2->m_foregroundColor = Color::WHITE;
    Window.GetSlider(sliderName)->m_sliderBar1->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider(sliderName)->m_sliderBar2->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    Window.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    Window.GetSlider(sliderName)->Hide();

    VarName = "Pressure";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.45f, -1.f, contourSliderBarWidth, contourSliderBarHeight), Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.45f, -1.f, contourSliderBarWidth, contourSliderBarHeight), Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    Window.GetSlider(sliderName)->m_maxValue = 1.05f;
    Window.GetSlider(sliderName)->m_minValue = 0.95f;
    Window.GetSlider(sliderName)->m_sliderBar1->m_foregroundColor = Color::BLUE;
    Window.GetSlider(sliderName)->m_sliderBar2->m_foregroundColor = Color::WHITE;
    Window.GetSlider(sliderName)->m_sliderBar1->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider(sliderName)->m_sliderBar2->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    Window.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    Window.GetSlider(sliderName)->Hide();

    VarName = "Water Rendering";
    labelName = "Label_"+VarName;
    sliderName = VarName;
    sliderBarName1 = VarName+"Max";
    sliderBarName2 = VarName+"Min";
    outputsPanel->CreateSlider(contourSliderPosition, Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-0.45f, -1.f, contourSliderBarWidth, contourSliderBarHeight), Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
    Window.GetSlider(sliderName)->CreateSliderBar(RectFloat( 0.45f, -1.f, contourSliderBarWidth, contourSliderBarHeight), Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
    Window.GetSlider(sliderName)->m_maxValue = 1.05f;
    Window.GetSlider(sliderName)->m_minValue = 0.95f;
    Window.GetSlider(sliderName)->m_sliderBar1->m_foregroundColor = Color::BLUE;
    Window.GetSlider(sliderName)->m_sliderBar2->m_foregroundColor = Color::WHITE;
    Window.GetSlider(sliderName)->m_sliderBar1->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider(sliderName)->m_sliderBar2->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
    Window.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
    Window.GetSlider(sliderName)->Hide();


    //Drawing panel
    //Window.CreateSubPanel(RectInt(g_leftPanelWidth, 0, g_drawingPanelWidth, g_drawingPanelHeight), Panel::DEF_ABS, "Drawing", Color(Color::DARK_GRAY));
    Panel* drawingPreview = Window.GetPanel("Drawing")->CreateSubPanel(RectFloat(-0.5f, -1.f, 1.5f, 1.5f), Panel::DEF_REL, "DrawingPreview", Color(Color::DARK_GRAY));
    Panel* drawingButtons = Window.GetPanel("Drawing")->CreateSubPanel(RectFloat(-0.9f, -1.f, 0.4f, 1.5f), Panel::DEF_REL, "DrawingButtons", Color(Color::DARK_GRAY));

    drawingPanel->CreateSlider(RectFloat(-0.9f, 0.9f-sliderH*0.75f*2,1.8f, sliderH*0.75f), Panel::DEF_REL, "Slider_Size", Color(Color::LIGHT_GRAY));
    drawingPanel->CreateSubPanel(RectFloat(-0.9f,0.9f-sliderH*0.75f, 0.5f, sliderH*0.75f), Panel::DEF_REL, "Label_Size", Color(Color::DARK_GRAY));
    Window.GetPanel("Label_Size")->m_displayText = "Size";

    float leftEnd = -0.9f;
    float width = 1.8f;
    float buttonSpacing = 0.0f;
    drawingButtons->CreateButton(RectFloat(-0.9f, 0.7f-0.04f , 1.8f, 0.3f ), Panel::DEF_REL, "Square"    , Color(Color::GRAY));
    drawingButtons->CreateButton(RectFloat(-0.9f, 0.4f-0.08f , 1.8f, 0.3f ), Panel::DEF_REL, "Circle"    , Color(Color::GRAY));
    drawingButtons->CreateButton(RectFloat(-0.9f, 0.1f-0.12f , 1.8f, 0.3f ), Panel::DEF_REL, "Hor. Line" , Color(Color::GRAY));
    drawingButtons->CreateButton(RectFloat(-0.9f,-0.2f-0.16f , 1.8f, 0.3f ), Panel::DEF_REL, "Vert. Line", Color(Color::GRAY));
    Window.GetSlider("Slider_Size")->CreateSliderBar(RectFloat(-0.2f,-sliderBarH*0.5f, sliderBarW, sliderBarH), Panel::DEF_REL, "SliderBar_Size", Color(Color::GRAY));
    Window.GetSlider("Slider_Size")->m_maxValue = 15.f;
    Window.GetSlider("Slider_Size")->m_minValue = 1.f;
    Window.GetSlider("Slider_Size")->m_sliderBar1->m_orientation = SliderBar::HORIZONTAL;
    Window.GetSlider("Slider_Size")->m_sliderBar1->UpdateValue();

    SetUpButtons();
    //VelMagButtonCallBack(); //default is vel mag contour
    WaterRenderingButtonCallBack(); //default is water rendering
    SquareButtonCallBack(); //default is square shape
    ThreeDButtonCallBack();
}

/*----------------------------------------------------------------------------------------
 *	Button setup
 */

Slider* GetCurrentContourSlider()
{
    if (Window.GetSlider("Velocity Magnitude")->m_draw == true) return Window.GetSlider("Velocity Magnitude");
    else if (Window.GetSlider("X Velocity")->m_draw == true) return Window.GetSlider("X Velocity");
    else if (Window.GetSlider("Y Velocity")->m_draw == true) return Window.GetSlider("Y Velocity");
    else if (Window.GetSlider("StrainRate")->m_draw == true) return Window.GetSlider("StrainRate");
    else if (Window.GetSlider("Pressure")->m_draw == true) return Window.GetSlider("Pressure");
    else if (Window.GetSlider("Water Rendering")->m_draw == true) return Window.GetSlider("Water Rendering");
}


void InitializeButtonCallBack()
{
    float4 *dptr;
    cudaGraphicsMapResources(1, &g_cudaSolutionField, 0);
    size_t num_bytes,num_bytes2;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, g_cudaSolutionField);

    float u = Window.GetSlider("Slider_InletV")->m_sliderBar1->GetValue();
    InitializeDomain(dptr, g_fA_d, g_im_d, u);
}

void VelMagButtonCallBack()
{
    contourButtons.ExclusiveEnable(Window.GetButton("Velocity Magnitude"));
    g_contourVar = VEL_MAG;
}

void VelXButtonCallBack()
{
    contourButtons.ExclusiveEnable(Window.GetButton("X Velocity"));
    g_contourVar = VEL_U;
}

void VelYButtonCallBack()
{
    contourButtons.ExclusiveEnable(Window.GetButton("Y Velocity"));
    g_contourVar = VEL_V;
}

void StrainRateButtonCallBack()
{
    contourButtons.ExclusiveEnable(Window.GetButton("StrainRate"));
    g_contourVar = STRAIN_RATE;
}

void PressureButtonCallBack()
{
    contourButtons.ExclusiveEnable(Window.GetButton("Pressure"));
    g_contourVar = PRESSURE;
}

void WaterRenderingButtonCallBack()
{
    contourButtons.ExclusiveEnable(Window.GetButton("Water Rendering"));
    g_contourVar = WATER_RENDERING;
}

void SquareButtonCallBack()
{
    shapeButtons.ExclusiveEnable(Window.GetButton("Square"));
    g_currentShape = Obstruction::SQUARE;
}

void CircleButtonCallBack()
{
    shapeButtons.ExclusiveEnable(Window.GetButton("Circle"));
    g_currentShape = Obstruction::CIRCLE;
}

void HorLineButtonCallBack()
{
    shapeButtons.ExclusiveEnable(Window.GetButton("Hor. Line"));
    g_currentShape = Obstruction::HORIZONTAL_LINE;
}

void VertLineButtonCallBack()
{
    shapeButtons.ExclusiveEnable(Window.GetButton("Vert. Line"));
    g_currentShape = Obstruction::VERTICAL_LINE;
}

void ThreeDButtonCallBack()
{
    viewButtons.ExclusiveEnable(Window.GetButton("3D"));
    g_viewMode = ViewMode::THREE_DIMENSIONAL;
}

void TwoDButtonCallBack()
{
    viewButtons.ExclusiveEnable(Window.GetButton("2D"));
    g_viewMode = ViewMode::TWO_DIMENSIONAL;
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
    float centerX = previewPanel->m_rectFloat_abs.GetCentroidX();
    float centerY = previewPanel->m_rectFloat_abs.GetCentroidY();
    float graphicsToWindowScaleFactor = static_cast<float>(winw)/Window.GetPanel("Graphics")->m_rectInt_abs.m_w;

    int xDimVisible = g_simParams.GetXDimVisible(&g_simParams);
    int yDimVisible = g_simParams.GetYDimVisible(&g_simParams);
    int r1ix = g_currentSize*static_cast<float>(Window.GetPanel("Graphics")->m_rectInt_abs.m_w) / (xDimVisible); //r1x in pixels
    int r1iy = g_currentSize*static_cast<float>(Window.GetPanel("Graphics")->m_rectInt_abs.m_h) / (yDimVisible); //r1x in pixels
    float r1fx = static_cast<float>(r1ix) / winw*2.f;
    float r1fy = static_cast<float>(r1iy) / winh*2.f;

    glColor3f(0.8f,0.8f,0.8f);
    switch (g_currentShape)
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
        r1fy = static_cast<float>(LINE_OBST_WIDTH) / winh*2.f;
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
        r1fx = static_cast<float>(LINE_OBST_WIDTH) / winw*2.f;
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
    GLuint* elementArrayIndices = new GLuint[(MAX_XDIM-1)*(MAX_YDIM-1) * 4 * 2];
    for (int j = 0; j < MAX_YDIM-1; j++){
        for (int i = 0; i < MAX_XDIM-1; i++){
            //going clockwise, since y orientation will be flipped when rendered
            elementArrayIndices[j*(MAX_XDIM-1)*4+i * 4 + 0] = (i)+(j)*MAX_XDIM;
            elementArrayIndices[j*(MAX_XDIM-1)*4+i * 4 + 1] = (i + 1) + (j)*MAX_XDIM;
            elementArrayIndices[j*(MAX_XDIM-1)*4+i * 4 + 2] = (i+1)+(j + 1)*MAX_XDIM;
            elementArrayIndices[j*(MAX_XDIM-1)*4+i * 4 + 3] = (i)+(j + 1)*MAX_XDIM;
        }
    }
    for (int j = 0; j < MAX_YDIM-1; j++){
        for (int i = 0; i < MAX_XDIM-1; i++){
            //going clockwise, since y orientation will be flipped when rendered
            elementArrayIndices[(MAX_XDIM-1)*(MAX_YDIM-1) * 4 + j*(MAX_XDIM-1)*4+i * 4 + 0] = (MAX_XDIM)*(MAX_YDIM) + (i)+(j)*MAX_XDIM;
            elementArrayIndices[(MAX_XDIM-1)*(MAX_YDIM-1) * 4 + j*(MAX_XDIM-1)*4+i * 4 + 1] = (MAX_XDIM)*(MAX_YDIM) + (i + 1) + (j)*MAX_XDIM;
            elementArrayIndices[(MAX_XDIM-1)*(MAX_YDIM-1) * 4 + j*(MAX_XDIM-1)*4+i * 4 + 2] = (MAX_XDIM)*(MAX_YDIM) + (i+1)+(j + 1)*MAX_XDIM;
            elementArrayIndices[(MAX_XDIM-1)*(MAX_YDIM-1) * 4 + j*(MAX_XDIM-1)*4+i * 4 + 3] = (MAX_XDIM)*(MAX_YDIM) + (i)+(j + 1)*MAX_XDIM;
        }
    }

    glGenBuffers(1, &arrayIndexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, arrayIndexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*(MAX_XDIM-1)*(MAX_YDIM-1)*4*2, elementArrayIndices, GL_DYNAMIC_DRAW);
    free(elementArrayIndices);
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
    CreateVBO(&g_vboSolutionField, &g_cudaSolutionField, solutionMemorySize+floorSize, cudaGraphicsMapFlagsWriteDiscard);

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
    int xDim = g_simParams.GetXDim(&g_simParams);
    int yDim = g_simParams.GetYDim(&g_simParams);
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
    int domainSize = ((MAX_XDIM + BLOCKSIZEX - 1) / BLOCKSIZEX)*(MAX_YDIM / BLOCKSIZEY)
                        *BLOCKSIZEX*BLOCKSIZEY;
    for (int i = 0; i < domainSize; i++)
    {
        int x = i%MAX_XDIM;
        int y = i/MAX_XDIM;
        g_im_h[i] = ImageFcn_h(x, y, g_obstructions);
    }
    size_t memsize_int = domainSize*sizeof(int);
    cudaMemcpy(g_im_d, g_im_h, memsize_int, cudaMemcpyHostToDevice);

}

void SetUpCUDA()
{
    size_t memsize, memsize_int, memsize_float, memsize_inputs, memsize_float2;
    g_uMax = 0.06f;

    int domainSize = ((MAX_XDIM + BLOCKSIZEX - 1) / BLOCKSIZEX)*(MAX_YDIM / BLOCKSIZEY)
                        *BLOCKSIZEX*BLOCKSIZEY;
    memsize = domainSize*sizeof(float)*9;
    memsize_int = domainSize*sizeof(int);
    memsize_float = domainSize*sizeof(float);
    memsize_float2 = domainSize*sizeof(float2);
    memsize_inputs = sizeof(g_obstructions);

    g_fA_h = (float *)malloc(memsize);
    g_fB_h = (float *)malloc(memsize);
    g_floor_h = (float *)malloc(memsize_float);
    g_im_h = (int *)malloc(memsize_int);
    d_rayCastIntersect = { 0, 0, 0, 1e6 };
    //obstructions = (input_values *)malloc(memsize_inputs);

    cudaMalloc((void **)&g_fA_d, memsize);
    cudaMalloc((void **)&g_fB_d, memsize);
    cudaMalloc((void **)&g_floor_d, memsize_float);
    cudaMalloc((void **)&g_floorFiltered_d, memsize_float);
    cudaMalloc((void **)&g_im_d, memsize_int);
    cudaMalloc((void **)&g_obst_d, memsize_inputs);
    cudaMalloc((void **)&d_rayCastIntersect_d, sizeof(float4));

    for (int i = 0; i < domainSize*9; i++)
    {
        g_fA_h[i] = 0;
        g_fB_h[i] = 0;
    }
    //u_max = UMAX;
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

//	for (int i = 0; i < domainSize; i++)
//	{
//		int x = i%MAX_XDIM;
//		int y = i/MAX_XDIM;
//		g_im_h[i] = ImageFcn_h(x, y, g_obstructions);
//	}
    for (int i = 0; i < domainSize; i++)
    {
        g_floor_h[i] = 0;
    }

    UpdateDeviceImage();
    
    cudaMemcpy(g_fA_d, g_fA_h, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(g_fB_d, g_fB_h, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(g_floor_d, g_floor_h, memsize_float, cudaMemcpyHostToDevice);
    cudaMemcpy(g_floorFiltered_d, g_floor_h, memsize_float, cudaMemcpyHostToDevice);
//	cudaMemcpy(g_im_d, g_im_h, memsize_int, cudaMemcpyHostToDevice);
    cudaMemcpy(g_obst_d, g_obstructions, memsize_inputs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rayCastIntersect_d, &d_rayCastIntersect, sizeof(float4), cudaMemcpyHostToDevice);

    //writeInputs();
    float u = Window.GetSlider("Slider_InletV")->m_sliderBar1->GetValue();

    float4 *dptr;
    cudaGraphicsMapResources(1, &g_cudaSolutionField, 0);
    size_t num_bytes,num_bytes2;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, g_cudaSolutionField);

    InitializeDomain(dptr, g_fA_d, g_im_d, u);
    InitializeDomain(dptr, g_fB_d, g_im_d, u);

    InitializeFloor(dptr, g_floor_d);
    InitializeFloor(dptr, g_floorFiltered_d);

    cudaGraphicsUnmapResources(1, &g_cudaSolutionField, 0);

}

void RunCuda(struct cudaGraphicsResource **vbo_resource, float3 cameraPosition)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    cudaGraphicsMapResources(1, &g_cudaSolutionField, 0);
    size_t num_bytes,num_bytes2;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, g_cudaSolutionField);

    float u = Window.GetSlider("Slider_InletV")->m_sliderBar1->GetValue();
    float omega = Window.GetSlider("Slider_Visc")->m_sliderBar1->GetValue();
    g_contMin = GetCurrentContourSlider()->m_sliderBar1->GetValue();
    g_contMax = GetCurrentContourSlider()->m_sliderBar2->GetValue();

    MarchSolution(dptr, g_fA_d, g_fB_d, g_im_d, g_obst_d, g_contourVar, g_contMin, g_contMax, g_viewMode, u, omega, g_tStep);
    SetObstructionVelocitiesToZero(g_obstructions, g_obst_d);

    if (g_viewMode == ViewMode::THREE_DIMENSIONAL || g_contourVar == ContourVariable::WATER_RENDERING)
    {
        LightSurface(dptr, g_obst_d, cameraPosition);
    }
    LightFloor(dptr, g_floor_d, g_floorFiltered_d, g_obst_d,cameraPosition);
    CleanUpDeviceVBO(dptr);

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
    int dx, dy;
    if (x >= 0 && x <= winw && y>=0 && y<=winh)
    {
        theMouse.Move(x, theMouse.m_winH-y-g_glutMouseYOffset);
    }
}


void MouseWheel(int button, int dir, int x, int y)
{
    if (dir > 0){
        translate_z -= 0.3f;
    }
    else
    {
        translate_z += 0.3f;
    }
    UpdateDeviceImage();
    
}

void Keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
    case (' ') :
        g_paused = (g_paused + 1) % 2;
        break;
    }
}

void UpdateWindowDimensionsBasedOnAspectRatio(int& heightOut, int& widthOut, int area, int leftPanelHeight, int leftPanelWidth, int xDim, int yDim, float scaleUp)
{
    float aspectRatio = static_cast<float>(xDim) / yDim;
    float leftPanelW = static_cast<float>(leftPanelWidth);
    heightOut = scaleUp*(-scaleUp*leftPanelW+sqrt(scaleUp*scaleUp*leftPanelW*leftPanelW+scaleUp*scaleUp*4*aspectRatio*area))/(scaleUp*scaleUp*2.f*aspectRatio);
    heightOut = max(heightOut, leftPanelHeight);
    widthOut = heightOut*aspectRatio+leftPanelW;
}

void UpdateDomainDimensionsBasedOnWindowSize(int leftPanelHeight, int leftPanelWidth, int windowWidth, int windowHeight, float scaleUp)
{

    int xDim = min(max(BLOCKSIZEX, int(ceil(((static_cast<float>(windowWidth - leftPanelWidth)/scaleUp)/BLOCKSIZEX))*BLOCKSIZEX)),MAX_XDIM);
    int yDim = min(max(1, int(ceil(static_cast<float>(windowHeight) / scaleUp))),MAX_YDIM);
    int xDimVisible = min(max(BLOCKSIZEX, int((static_cast<float>(windowWidth - leftPanelWidth)/scaleUp))),MAX_XDIM);
    int yDimVisible = yDim;

    g_simParams.SetXDimVisible(&g_simParams, xDimVisible);
    g_simParams.SetYDimVisible(&g_simParams, yDimVisible);


}

void Resize(int w, int h)
{
    int area = w*h;
    UpdateDomainDimensionsBasedOnWindowSize(max(g_leftPanelHeight, g_drawingPanelHeight), g_leftPanelWidth + g_drawingPanelWidth, w, h, g_initialScaleUp);

    winw = w;
    winh = h;

    theMouse.m_winW = winw;
    theMouse.m_winH = winh;

    Window.m_rectInt_abs = RectInt(200, 100, winw, winh);
    Window.m_rectFloat_abs = Window.RectIntAbsToRectFloatAbs();

    Window.GetPanel("CDV")->m_rectInt_abs = RectInt(0, winh - g_leftPanelHeight, g_leftPanelWidth, g_leftPanelHeight);
    Window.GetPanel("Drawing")->m_rectInt_abs = RectInt(g_leftPanelWidth, winh - g_drawingPanelHeight, g_drawingPanelWidth, g_drawingPanelHeight);
    Window.GetPanel("Graphics")->m_rectInt_abs = RectInt(g_leftPanelWidth+g_drawingPanelWidth, 0, winw-g_leftPanelWidth-g_drawingPanelWidth, winh);
    Window.UpdateAll();

    glViewport(0, 0, winw, winh);

    UpdateDeviceImage();

}


void ComputeFPS(int &fpsCount, int fpsLimit, clock_t &before){
    fpsCount++;
    if (fpsCount % fpsLimit == 0)
    {
        clock_t difference = clock() - before;
        float timeStepsPerSecond_prev = g_timeStepsPerSecond;
        g_fps = static_cast<float>(fpsLimit) / (static_cast<float>(difference) / CLOCKS_PER_SEC);
        g_timeStepsPerSecond = g_tStep * 2 * g_fps;
        before = clock();
        //fpsLimit = (int)min(max(avgFPS,1.f),30.f);
        //Time step optimizer
        //if (g_timeStepsPerSecond > timeStepsPerSecond_prev*1.2f){
        //	g_tStep = max(1, g_tStep - 1);
        //}
    }
    char fpsReport[256];
    int xDim = g_simParams.GetXDim(&g_simParams);
    int yDim = g_simParams.GetYDim(&g_simParams);
    sprintf(fpsReport, "Interactive CFD running at: %i timesteps/frame at %3.1f fps = %3.1f timesteps/second on %ix%i mesh", g_tStep * 2, g_fps, g_timeStepsPerSecond, xDim, yDim);
    glutSetWindowTitle(fpsReport);
}

void Draw()
{
    if (g_fpsCount == 0)
    {
        g_timeBefore = clock();
    }


    glutReshapeWindow(winw, winh);
    g_currentSize = Window.GetSlider("Slider_Size")->m_sliderBar1->GetValue();
    g_initialScaleUp = Window.GetSlider("Slider_Resolution")->m_sliderBar1->GetValue();
    Resize(winw, winh);

    int graphicsViewWidth = winw - g_leftPanelWidth - g_drawingPanelWidth;
    int graphicsViewHeight = winh;
    int xDimVisible = g_simParams.GetXDimVisible(&g_simParams);
    int yDimVisible = g_simParams.GetYDimVisible(&g_simParams);
    float xTranslation = -((static_cast<float>(winw)-xDimVisible*g_initialScaleUp)*0.5 - static_cast<float>(g_leftPanelWidth + g_drawingPanelWidth)) / winw*2.f;
    float yTranslation = -((static_cast<float>(winh)-yDimVisible*g_initialScaleUp)*0.5)/ winh*2.f;
    float3 cameraPosition = { -xTranslation - translate_x, -yTranslation - translate_y, +2 - translate_z };

    RunCuda(&g_cudaSolutionField, cameraPosition);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glTranslatef(xTranslation,yTranslation,0.f);
    glScalef((static_cast<float>(xDimVisible*g_initialScaleUp) / winw), (static_cast<float>(yDimVisible*g_initialScaleUp) / winh), 1.f);

    if (g_viewMode == ViewMode::TWO_DIMENSIONAL)
    {
        glOrtho(-1,1,-1,static_cast<float>(yDimVisible)/xDimVisible*2.f-1.f,-100,20);
    }
    else
    {
        gluPerspective(45.0, static_cast<float>(xDimVisible) / yDimVisible, 0.1, 10.0);
        glTranslatef(translate_x, translate_y, -2+translate_z);
        glRotatef(-rotate_x,1,0,0);
        glRotatef(rotate_z,0,0,1);
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //Draw solution field
    glBindBuffer(GL_ARRAY_BUFFER, g_vboSolutionField);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_elementArrayIndexBuffer);
    glVertexPointer(3, GL_FLOAT, 16, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glEnableClientState(GL_COLOR_ARRAY);
    glColorPointer(4, GL_UNSIGNED_BYTE, 16, (char *)NULL + 12);
    if (g_viewMode == ViewMode::THREE_DIMENSIONAL || g_contourVar == ContourVariable::WATER_RENDERING)
    {
        //Draw floor
        glDrawElements(GL_QUADS, (MAX_XDIM - 1)*(yDimVisible - 1)*4, GL_UNSIGNED_INT, BUFFER_OFFSET(sizeof(GLuint)*4*(MAX_XDIM - 1)*(MAX_YDIM - 1)));
    }
    //Draw water surface
    glDrawElements(GL_QUADS, (MAX_XDIM - 1)*(yDimVisible - 1)*4 , GL_UNSIGNED_INT, (GLvoid*)0);
    glDisableClientState(GL_VERTEX_ARRAY);

    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cout << "OpenGL error: " << err << std::endl;
    }


    // Update transformation matrices in graphics manager for mouse ray casting
    Window.GetPanel("Graphics")->m_graphicsManager->UpdateViewTransformations();

    /*
     *	Disable depth test and lighting for 2D elements
     */
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1,1,-1,1,-100,20);

    /*
     *	Draw the 2D overlay
     */
    Draw2D();

    /*
     *	Bring the back buffer to the front and vice-versa.
     */
    glutSwapBuffers();

    ComputeFPS(g_fpsCount, g_fpsLimit, g_timeBefore);

}

int main(int argc,char **argv)
{
    SetUpWindow();

    glutInit(&argc,argv);

    glutInitDisplayMode(GLUT_RGB|GLUT_DEPTH|GLUT_DOUBLE);
    //glutInitWindowSize(1400,g_leftPanelHeight+200);
    glutInitWindowSize(winw,winh);
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
    SetUpGLInterop();
    SetUpCUDA();

    glutMainLoop();

    return 0;
}