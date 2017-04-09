#include "Layout.h"
#include "Panel/ButtonGroup.h"
#include "Panel/Button.h"
#include "Panel/Slider.h"
#include "Panel/Panel.h"
#include "Graphics/GraphicsManager.h"
#include "Graphics/ShaderManager.h"
#include "Graphics/CudaLbm.h"
#include "kernel.h"
#include <algorithm>

extern const int g_leftPanelWidth(350);
extern const int g_leftPanelHeight(500);

void Layout::SetUpWindow(Panel &rootPanel)
{
    const int windowWidth = 1200;
    const int windowHeight = g_leftPanelHeight+100;

    rootPanel.SetSize_Absolute(RectInt(200, 100, windowWidth, windowHeight));
    rootPanel.m_draw = false;
    rootPanel.SetName("Main Window");

    Panel* const CDV = rootPanel.CreateSubPanel(RectInt(0, 0, g_leftPanelWidth, g_leftPanelHeight), Panel::DEF_ABS,
        "CDV", Color(Color::DARK_GRAY));
    Panel* const outputsPanel = CDV->CreateSubPanel(RectFloat(-1.f,  -0.9f, 2.f, 0.5f), Panel::DEF_REL,
        "Outputs", Color(Color::DARK_GRAY));
    Panel* const inputsPanel  = CDV->CreateSubPanel(RectFloat(-1.f, -0.4f, 2.f, 0.6f), Panel::DEF_REL,
        "Inputs", Color(Color::DARK_GRAY));
    Panel* const drawingPanel = CDV->CreateSubPanel(RectFloat(-1.f,  0.2f, 2.f, 0.8f), Panel::DEF_REL,
        "Drawing", Color(Color::DARK_GRAY));
    Panel* const viewModePanel = CDV->CreateSubPanel(RectFloat(-1.f,  -1.f, 2.f, 0.1f), Panel::DEF_REL,
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

    const float sliderH = 1.4f/3.f/2.f;
    const float sliderBarW = 0.1f;
    const float sliderBarH = 2.f;

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
    const float contourSliderBarWidth = 0.1f;
    const float contourSliderBarHeight = 2.f;
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
    Panel* const drawingPreview = rootPanel.GetPanel("Drawing")->CreateSubPanel(RectFloat(-0.5f, -1.f, 1.5f, 1.5f),
        Panel::DEF_REL, "DrawingPreview", Color(Color::DARK_GRAY));
    Panel* const drawingButtons = rootPanel.GetPanel("Drawing")->CreateSubPanel(RectFloat(-0.9f, -1.f, 0.4f, 1.5f),
        Panel::DEF_REL, "DrawingButtons", Color(Color::DARK_GRAY));

    drawingPanel->CreateSlider(RectFloat(-0.9f, 0.9f-sliderH*0.75f*2,1.8f, sliderH*0.75f), Panel::DEF_REL,
        "Slider_Size", Color(Color::LIGHT_GRAY));
    drawingPanel->CreateSubPanel(RectFloat(-0.9f,0.9f-sliderH*0.75f, 0.5f, sliderH*0.75f), Panel::DEF_REL,
        "Label_Size", Color(Color::DARK_GRAY));
    rootPanel.GetPanel("Label_Size")->SetDisplayText("Size");

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
    const float currentObstSize = rootPanel.GetSlider("Slider_Size")->m_sliderBar1->GetValue();
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetCurrentObstSize(currentObstSize);

    Layout::SetUpButtons(rootPanel);
    WaterRenderingButtonCallBack(rootPanel); //default is water rendering
    SquareButtonCallBack(rootPanel); //default is square shape
    ThreeDButtonCallBack(rootPanel);
}

/*----------------------------------------------------------------------------------------
 *	Button setup
 */

Slider* Layout::GetCurrentContourSlider(Panel &rootPanel)
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

float Layout::GetCurrentSliderValue(Panel &rootPanel, const std::string name, const int sliderNumber)
{
    if (sliderNumber == 1)
    {
        return rootPanel.GetSlider(name)->m_sliderBar1->GetValue();
    }
    else
    {
        return rootPanel.GetSlider(name)->m_sliderBar2->GetValue();
    }
}

float Layout::GetCurrentContourSliderValue(Panel &rootPanel, const int sliderNumber)
{
    if (sliderNumber == 1)
    {
        return GetCurrentContourSlider(rootPanel)->m_sliderBar1->GetValue();
    }
    else
    {
        return GetCurrentContourSlider(rootPanel)->m_sliderBar2->GetValue();
    }
}

void Layout::GetCurrentContourSliderBoundValues(Panel &rootPanel, float &minValue, float &maxValue)
{
    minValue = GetCurrentContourSlider(rootPanel)->GetMinValue();
    maxValue = GetCurrentContourSlider(rootPanel)->GetMaxValue();
}

void InitializeButtonCallBack(Panel &rootPanel)
{
    GraphicsManager* const graphicsManager = rootPanel.GetPanel("Graphics")->GetGraphicsManager();
    ShaderManager* graphics = graphicsManager->GetGraphics();
    cudaGraphicsResource* cudaSolutionField = graphics->GetCudaSolutionGraphicsResource();
    float4* dptr;
    cudaGraphicsMapResources(1, &cudaSolutionField, 0);
    size_t num_bytes,num_bytes2;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cudaSolutionField);

    CudaLbm* const cudaLbm = graphicsManager->GetCudaLbm();

    float* fA_d = cudaLbm->GetFA();
    float* fB_d = cudaLbm->GetFB();
    int* im_d = cudaLbm->GetImage();

    float u = rootPanel.GetSlider("Slider_InletV")->m_sliderBar1->GetValue();
    Domain* const domain = cudaLbm->GetDomain();
    InitializeDomain(dptr, fA_d, im_d, u, *domain);
    graphics->InitializeComputeShaderData();
}

void VelMagButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* const contourButtons = rootPanel.GetButtonGroup("ContourButtons");
    contourButtons->ExclusiveEnable(rootPanel.GetButton("Velocity Magnitude"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetContourVar(VEL_MAG);
}

void VelXButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* const contourButtons = rootPanel.GetButtonGroup("ContourButtons");
    contourButtons->ExclusiveEnable(rootPanel.GetButton("X Velocity"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetContourVar(VEL_U);
}

void VelYButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* const contourButtons = rootPanel.GetButtonGroup("ContourButtons");
    contourButtons->ExclusiveEnable(rootPanel.GetButton("Y Velocity"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetContourVar(VEL_V);
}

void StrainRateButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* const contourButtons = rootPanel.GetButtonGroup("ContourButtons");
    contourButtons->ExclusiveEnable(rootPanel.GetButton("StrainRate"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetContourVar(STRAIN_RATE);
}

void PressureButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* const contourButtons = rootPanel.GetButtonGroup("ContourButtons");
    contourButtons->ExclusiveEnable(rootPanel.GetButton("Pressure"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetContourVar(PRESSURE);
}

void WaterRenderingButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* const contourButtons = rootPanel.GetButtonGroup("ContourButtons");
    contourButtons->ExclusiveEnable(rootPanel.GetButton("Water Rendering"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetContourVar(WATER_RENDERING);
}

void SquareButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* const shapeButtons = rootPanel.GetButtonGroup("ShapeButtons");
    shapeButtons->ExclusiveEnable(rootPanel.GetButton("Square"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetCurrentObstShape(Shape::SQUARE);
}

void CircleButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* const shapeButtons = rootPanel.GetButtonGroup("ShapeButtons");
    shapeButtons->ExclusiveEnable(rootPanel.GetButton("Circle"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetCurrentObstShape(Shape::CIRCLE);
}

void HorLineButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* const shapeButtons = rootPanel.GetButtonGroup("ShapeButtons");
    shapeButtons->ExclusiveEnable(rootPanel.GetButton("Hor. Line"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetCurrentObstShape(Shape::HORIZONTAL_LINE);
}

void VertLineButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* const shapeButtons = rootPanel.GetButtonGroup("ShapeButtons");
    shapeButtons->ExclusiveEnable(rootPanel.GetButton("Vert. Line"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetCurrentObstShape(Shape::VERTICAL_LINE);
}

void ThreeDButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* const viewModeButtons = rootPanel.GetButtonGroup("ViewModeButtons");
    viewModeButtons->ExclusiveEnable(rootPanel.GetButton("3D"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetViewMode(THREE_DIMENSIONAL);
}

void TwoDButtonCallBack(Panel &rootPanel)
{
    ButtonGroup* const viewModeButtons = rootPanel.GetButtonGroup("ViewModeButtons");
    viewModeButtons->ExclusiveEnable(rootPanel.GetButton("2D"));
    rootPanel.GetPanel("Graphics")->GetGraphicsManager()->SetViewMode(TWO_DIMENSIONAL);
}

void Layout::SetUpButtons(Panel &rootPanel)
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
    ButtonGroup* const contourButtonGroup = rootPanel.CreateButtonGroup("ContourButtons", buttons);

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
    ButtonGroup* const shapeButtonGroup = rootPanel.CreateButtonGroup("ShapeButtons", buttons2);

    rootPanel.GetButton("3D")->SetCallback(ThreeDButtonCallBack);
    rootPanel.GetButton("2D")->SetCallback(TwoDButtonCallBack);
    
    std::vector<Button*> buttons3 = {
        rootPanel.GetButton("2D"),
        rootPanel.GetButton("3D")
    };
    ButtonGroup* const viewModeButtonGroup = rootPanel.CreateButtonGroup("ViewModeButtons", buttons3);

}

void Layout::Draw2D(Panel &rootPanel)
{
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1,1,-1,1,-100,20);
    rootPanel.DrawAll();
    Layout::DrawShapePreview(rootPanel);
}

void Layout::DrawShapePreview(Panel &rootPanel)
{
    Panel* const previewPanel = rootPanel.GetPanel("DrawingPreview");
    const float centerX = previewPanel->GetRectFloatAbs().GetCentroidX();
    const float centerY = previewPanel->GetRectFloatAbs().GetCentroidY();
    const int windowWidth = rootPanel.GetWidth();
    const int windowHeight = rootPanel.GetHeight();
    const float graphicsToWindowScaleFactor = static_cast<float>(windowWidth)/
        rootPanel.GetPanel("Graphics")->GetRectIntAbs().m_w;

    GraphicsManager* const graphicsManager = rootPanel.GetPanel("Graphics")->GetGraphicsManager();
    CudaLbm* const cudaLbm = graphicsManager->GetCudaLbm();
    const int xDimVisible = cudaLbm->GetDomain()->GetXDimVisible();
    const int yDimVisible = cudaLbm->GetDomain()->GetYDimVisible();
    const float currentSize = rootPanel.GetSlider("Slider_Size")->m_sliderBar1->GetValue();
    const int graphicsWindowWidth = rootPanel.GetPanel("Graphics")->GetRectIntAbs().m_w;
    const int graphicsWindowHeight = rootPanel.GetPanel("Graphics")->GetRectIntAbs().m_h;
    const int r1ix = currentSize*static_cast<float>(graphicsWindowWidth) / (xDimVisible); //r1x in pixels
    const int r1iy = currentSize*static_cast<float>(graphicsWindowHeight) / (yDimVisible); //r1x in pixels
    float r1fx = static_cast<float>(r1ix) / windowWidth*2.f;
    float r1fy = static_cast<float>(r1iy) / windowHeight*2.f;

    Shape currentShape = graphicsManager->GetCurrentObstShape();
    glColor3f(0.8f,0.8f,0.8f);
    switch (currentShape)
    {
    case Shape::CIRCLE:
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
    case Shape::SQUARE:
    {
        glBegin(GL_QUADS);
            glVertex2f(centerX - r1fx, centerY + r1fy);
            glVertex2f(centerX - r1fx, centerY - r1fy);
            glVertex2f(centerX + r1fx, centerY - r1fy);
            glVertex2f(centerX + r1fx, centerY + r1fy);
        glEnd();
        break;
    }
    case Shape::HORIZONTAL_LINE:
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
    case Shape::VERTICAL_LINE:
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

void Layout::UpdateWindowDimensionsBasedOnAspectRatio(int& heightOut, int& widthOut, const int area,
    const int leftPanelHeight, const int leftPanelWidth, const int xDim, const int yDim,
    const float scaleUp)
{
    const float aspectRatio = static_cast<float>(xDim) / yDim;
    const float leftPanelW = static_cast<float>(leftPanelWidth);
    heightOut = scaleUp*(-scaleUp*leftPanelW+sqrt(scaleUp*scaleUp*leftPanelW*leftPanelW
        +scaleUp*scaleUp*4*aspectRatio*area))/(scaleUp*scaleUp*2.f*aspectRatio);
    heightOut = std::max(heightOut, leftPanelHeight);
    widthOut = heightOut*aspectRatio+leftPanelW;
}

void Layout::UpdateDomainDimensionsBasedOnWindowSize(Panel &rootPanel, const int leftPanelHeight,
    const int leftPanelWidth)
{
    GraphicsManager* const graphicsManager = rootPanel.GetPanel("Graphics")->GetGraphicsManager();
    const float scaleUp = graphicsManager->GetScaleFactor();
    const int windowWidth = rootPanel.GetWidth();
    const int windowHeight = rootPanel.GetHeight();
    const int xDimVisible = static_cast<float>(windowWidth - leftPanelWidth) / scaleUp;
    const int yDimVisible = ceil(static_cast<float>(windowHeight) / scaleUp);
    CudaLbm* const cudaLbm = graphicsManager->GetCudaLbm();
    cudaLbm->GetDomain()->SetXDimVisible(xDimVisible);
    cudaLbm->GetDomain()->SetYDimVisible(yDimVisible);
}


