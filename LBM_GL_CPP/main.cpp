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

#include "kernel.h"
#include "Mouse.h"
#include "Panel.h"
#include "common.h"


int winw, winh;
const int g_leftPanelWidth(250);
const int g_drawingPanelWidth(200);
const int g_leftPanelHeight(600);
const int g_drawingPanelHeight(600);
float g_initialScaleUp(2.f); 

int g_fpsCount = 0;
int g_fpsLimit = 20;
float g_fps = 0;
float g_timeStepsPerSecond = 0;
clock_t g_timeBefore;


//simulation inputs
int g_xDim = 512; // 256;// 512;
int g_yDim = 384; //;// 384;
int g_xDimVisible, g_yDimVisible;
float g_uMax = 0.1f;
float g_contMin = 0.f;
float g_contMax = 0.1f;
int g_tStep = 15; //initial tstep value before adjustments

ContourVariable g_contourVar;

//view transformations
float rotate_x = 45.f;
float translate_z = 1.f;

Obstruction g_obstructions[MAXOBSTS];

Panel Window;
Mouse theMouse;

ButtonGroup contourButtons;
ButtonGroup shapeButtons;

Obstruction::Shape g_currentShape=Obstruction::CIRCLE;
float g_currentSize = 5.f;

GLuint g_vboSolutionField;
GLuint g_elementArrayIndexBuffer;
cudaGraphicsResource *g_cudaSolutionField;

int* g_elementArrayIndices;

float* g_fA_h;
float* g_fA_d;
float* g_fB_h;
float* g_fB_d;
int* g_im_h;
int* g_im_d;
Obstruction* g_obst_d;

const int g_glutMouseYOffset = 10; //hack to get better mouse precision


void SetUpButtons();
void VelMagButtonCallBack();
void SquareButtonCallBack();
void Resize(int w, int h);

void UpdateWindowDimensionsBasedOnAspectRatio(int& heightOut, int& widthOut, int area, int leftPanelHeight, int leftPanelWidth, int xDim, int yDim, float scaleUp);

void Init()
{
	glEnable(GL_LIGHT0);
	glewInit();
	glViewport(0,0,winw, winh);

}

void SetUpWindow()
{
	winw = g_xDim*g_initialScaleUp + g_leftPanelWidth;
	winh = max(g_yDim*g_initialScaleUp,g_leftPanelHeight+100);
	UpdateWindowDimensionsBasedOnAspectRatio(winh, winw, winw*winh, g_leftPanelHeight, g_leftPanelWidth, g_xDim, g_yDim, g_initialScaleUp);

	Window.m_rectInt_abs = RectInt(200, 100, winw, winh);
	Window.m_rectFloat_abs = Window.RectIntAbsToRectFloatAbs();
	Window.m_draw = false;
	Window.m_name = "Main Window";
	Window.m_sizeDefinition = Panel::DEF_ABS;
	theMouse.SetBasePanel(&Window);
	theMouse.m_simScaleUp = g_initialScaleUp;

	Panel* CDV = Window.CreateSubPanel(RectInt(0, 0, g_leftPanelWidth, g_leftPanelHeight), Panel::DEF_ABS, "CDV", Color(Color::BLACK));
	Panel* outputsPanel = CDV->CreateSubPanel(RectFloat(-1.f,  -1.f, 2.f, 0.6f), Panel::DEF_REL, "Outputs", Color(Color::RED));
	Panel* inputsPanel  = CDV->CreateSubPanel(RectFloat(-1.f, -0.4f, 2.f, 0.6f), Panel::DEF_REL, "Inputs", Color(Color::GREEN));
	Panel* drawingPanel = CDV->CreateSubPanel(RectFloat(-1.f,  0.2f, 2.f, 0.8f), Panel::DEF_REL, "Drawing", Color(Color::RED));


	outputsPanel->CreateButton(RectFloat(-0.9f,  0.f+0.02f , 1.8f, 0.2f ), Panel::DEF_REL, "Initialize", Color(Color::GRAY));
	outputsPanel->CreateButton(RectFloat(-0.9f, -0.2f+0.01f, 1.1f, 0.19f), Panel::DEF_REL, "Velocity Magnitude", Color(Color::GRAY));
	outputsPanel->CreateButton(RectFloat(-0.9f, -0.4f+0.01f, 1.1f, 0.19f), Panel::DEF_REL, "X Velocity", Color(Color::GRAY));
	outputsPanel->CreateButton(RectFloat(-0.9f, -0.6f+0.01f, 1.1f, 0.19f), Panel::DEF_REL, "Y Velocity", Color(Color::GRAY));
	outputsPanel->CreateButton(RectFloat(-0.9f, -0.8f+0.01f, 1.1f, 0.19f), Panel::DEF_REL, "StrainRate", Color(Color::GRAY));
	outputsPanel->CreateButton(RectFloat(-0.9f, -1.f+0.01f , 1.1f, 0.19f) ,Panel::DEF_REL, "Pressure"  , Color(Color::GRAY));

	Window.CreateSubPanel(RectInt(g_leftPanelWidth, 0, winw-g_leftPanelWidth, winh), Panel::DEF_ABS, "Graphics", Color(Color::RED));
	Window.GetPanel("Graphics")->m_draw = false;
	Window.GetPanel("Graphics")->CreateGraphicsManager();
	Window.GetPanel("Graphics")->m_graphicsManager->m_obstructions = &g_obstructions[0];


	float sliderW = 0.4f;
	float sliderBarW = 2.f;
	float sliderBarH = 0.2f;
	inputsPanel->CreateSubPanel(RectFloat(-0.9f,0.8f, 0.8f, 0.1f), Panel::DEF_REL, "Label_InletV", Color(Color::BLACK));
	Window.GetPanel("Label_InletV")->m_displayText = "Inlet Velocity";
	inputsPanel->CreateSlider(RectFloat(-0.5f-sliderW*0.5f,0.25f, sliderW, 0.5f), Panel::DEF_REL, "Slider_InletV", Color(Color::LIGHT_GRAY));
	Window.GetSlider("Slider_InletV")->CreateSliderBar(RectFloat(-sliderBarW*0.5f, 0.5f, sliderBarW, sliderBarH), Panel::DEF_REL, "SliderBar_InletV", Color(Color::GRAY));
	Window.GetSlider("Slider_InletV")->m_maxValue = 0.1f;
	Window.GetSlider("Slider_InletV")->m_minValue = 0.f;
	Window.GetSlider("Slider_InletV")->m_sliderBar1->UpdateValue();

	inputsPanel->CreateSubPanel(RectFloat(0.1f,0.8f, 0.8f, 0.1f), Panel::DEF_REL, "Label_Visc", Color(Color::BLACK));
	Window.GetPanel("Label_Visc")->m_displayText = "Viscosity";
	inputsPanel->CreateSlider(RectFloat(0.5f-sliderW*0.5f,0.25f, sliderW, 0.5f), Panel::DEF_REL, "Slider_Visc", Color(Color::LIGHT_GRAY));
	Window.GetSlider("Slider_Visc")->CreateSliderBar(RectFloat(-sliderBarW*0.5f, -0.85f, sliderBarW, sliderBarH), Panel::DEF_REL, "SliderBar_Visc", Color(Color::GRAY));
	Window.GetSlider("Slider_Visc")->m_maxValue = 1.8f;
	Window.GetSlider("Slider_Visc")->m_minValue = 1.99f;
	Window.GetSlider("Slider_Visc")->m_sliderBar1->UpdateValue();

	sliderBarH = 0.15f;
	std::string VarName = "Velocity Magnitude";
	std::string labelName = "Label_"+VarName;
	std::string sliderName = VarName;
	std::string sliderBarName1 = VarName+"Max";
	std::string sliderBarName2 = VarName+"Min";
	outputsPanel->CreateSubPanel(RectFloat(0.2f,-0.19f, 0.8f, 0.2f), Panel::DEF_REL, labelName, Color(Color::BLACK));
	Window.GetPanel(labelName)->m_displayText = "Contour";
	outputsPanel->CreateSlider(RectFloat(0.6f-sliderW*0.5f,-0.95f, sliderW, 0.75f), Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
	Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-sliderBarW*0.5f, -0.95f, sliderBarW, sliderBarH), Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
	Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-sliderBarW*0.5f,  0.65f, sliderBarW, sliderBarH), Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
	Window.GetSlider(sliderName)->m_maxValue = g_uMax*2.f;
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
	outputsPanel->CreateSubPanel(RectFloat(0.2f,-0.19f, 0.8f, 0.2f), Panel::DEF_REL, labelName, Color(Color::BLACK));
	Window.GetPanel(labelName)->m_displayText = "Contour";
	outputsPanel->CreateSlider(RectFloat(0.6f-sliderW*0.5f,-0.95f, sliderW, 0.75f), Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
	Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-sliderBarW*0.5f, -0.85f, sliderBarW, sliderBarH), Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
	Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-sliderBarW*0.5f,  0.65f, sliderBarW, sliderBarH), Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
	Window.GetSlider(sliderName)->m_maxValue = g_uMax*1.8f;
	Window.GetSlider(sliderName)->m_minValue = -g_uMax*1.f;
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
	outputsPanel->CreateSubPanel(RectFloat(0.2f,-0.19f, 0.8f, 0.2f), Panel::DEF_REL, labelName, Color(Color::BLACK));
	Window.GetPanel(labelName)->m_displayText = "Contour";
	outputsPanel->CreateSlider(RectFloat(0.6f-sliderW*0.5f,-0.95f, sliderW, 0.75f), Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
	Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-sliderBarW*0.5f, -0.65f, sliderBarW, sliderBarH), Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
	Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-sliderBarW*0.5f,  0.65f, sliderBarW, sliderBarH), Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
	Window.GetSlider(sliderName)->m_maxValue = g_uMax*1.f;
	Window.GetSlider(sliderName)->m_minValue = -g_uMax*1.f;
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
	outputsPanel->CreateSubPanel(RectFloat(0.2f,-0.19f, 0.8f, 0.2f), Panel::DEF_REL, labelName, Color(Color::BLACK));
	Window.GetPanel(labelName)->m_displayText = "Contour";
	outputsPanel->CreateSlider(RectFloat(0.6f-sliderW*0.5f,-0.95f, sliderW, 0.75f), Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
	Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-sliderBarW*0.5f, -0.9f, sliderBarW, sliderBarH), Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
	Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-sliderBarW*0.5f,  0.35f, sliderBarW, sliderBarH), Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
	Window.GetSlider(sliderName)->m_maxValue = g_uMax*0.1f;
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
	outputsPanel->CreateSubPanel(RectFloat(0.2f,-0.19f, 0.8f, 0.2f), Panel::DEF_REL, labelName, Color(Color::BLACK));
	Window.GetPanel(labelName)->m_displayText = "Contour";
	outputsPanel->CreateSlider(RectFloat(0.6f-sliderW*0.5f,-0.95f, sliderW, 0.75f), Panel::DEF_REL, sliderName, Color(Color::LIGHT_GRAY));
	Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-sliderBarW*0.5f, -0.45f, sliderBarW, sliderBarH), Panel::DEF_REL, sliderBarName1, Color(Color::GRAY));
	Window.GetSlider(sliderName)->CreateSliderBar(RectFloat(-sliderBarW*0.5f,  0.45f, sliderBarW, sliderBarH), Panel::DEF_REL, sliderBarName2, Color(Color::GRAY));
	Window.GetSlider(sliderName)->m_maxValue = 1.05f;
	Window.GetSlider(sliderName)->m_minValue = 0.95f;
	Window.GetSlider(sliderName)->m_sliderBar1->m_foregroundColor = Color::BLUE;
	Window.GetSlider(sliderName)->m_sliderBar2->m_foregroundColor = Color::WHITE;
	Window.GetSlider(sliderName)->m_sliderBar1->UpdateValue();
	Window.GetSlider(sliderName)->m_sliderBar2->UpdateValue();
	Window.GetSlider(sliderName)->Hide();

	//Drawing panel
	//Window.CreateSubPanel(RectInt(g_leftPanelWidth, 0, g_drawingPanelWidth, g_drawingPanelHeight), Panel::DEF_ABS, "Drawing", Color(Color::DARK_GRAY));
	Window.GetPanel("Drawing")->CreateSubPanel(RectFloat(-0.9f,0.85f, 0.8f, 0.1f), Panel::DEF_REL, "Label_Size", Color(Color::DARK_GRAY));
	Window.GetPanel("Label_Size")->m_displayText = "Size";
	Window.GetPanel("Drawing")->CreateButton(RectFloat(-0.9f, 0.35f-0.02f , 1.8f, 0.14f ), Panel::DEF_REL, "Square"    , Color(Color::GRAY));
	Window.GetPanel("Drawing")->CreateButton(RectFloat(-0.9f, 0.2f-0.02f , 1.8f, 0.14f ), Panel::DEF_REL, "Circle"    , Color(Color::GRAY));
	Window.GetPanel("Drawing")->CreateButton(RectFloat(-0.9f, 0.05f-0.02f , 1.8f, 0.14f ), Panel::DEF_REL, "Hor. Line" , Color(Color::GRAY));
	Window.GetPanel("Drawing")->CreateButton(RectFloat(-0.9f,-0.10f-0.02f , 1.8f, 0.14f ), Panel::DEF_REL, "Vert. Line", Color(Color::GRAY));
	Window.GetPanel("Drawing")->CreateSlider(RectFloat(-0.5f-sliderW*0.5f,0.5f, sliderW, 0.35f), Panel::DEF_REL, "Slider_Size", Color(Color::LIGHT_GRAY));
	Window.GetSlider("Slider_Size")->CreateSliderBar(RectFloat(-sliderBarW*0.5f, 0.f, sliderBarW, sliderBarH*1.5f), Panel::DEF_REL, "SliderBar_Size", Color(Color::GRAY));
	Window.GetSlider("Slider_Size")->m_maxValue = 15.f;
	Window.GetSlider("Slider_Size")->m_minValue = 1.f;
	Window.GetSlider("Slider_Size")->m_sliderBar1->UpdateValue();
	Window.GetPanel("Drawing")->CreateSlider(RectFloat(0.5f-sliderW*0.5f,0.5f, sliderW, 0.35f), Panel::DEF_REL, "Slider_Resolution", Color(Color::LIGHT_GRAY));
	Window.GetSlider("Slider_Resolution")->CreateSliderBar(RectFloat(-sliderBarW*0.5f, 0.f, sliderBarW, sliderBarH*1.5f), Panel::DEF_REL, "SliderBar_Resolution", Color(Color::GRAY));
	Window.GetSlider("Slider_Resolution")->m_maxValue = 1.f;
	Window.GetSlider("Slider_Resolution")->m_minValue = 4.f;
	Window.GetSlider("Slider_Resolution")->m_sliderBar1->UpdateValue();

	Window.GetPanel("Drawing")->CreateSubPanel(RectFloat(-1.f, -1.f, 2.f, 0.75f), Panel::DEF_REL, "DrawingPreview", Color(Color::DARK_GRAY));

	SetUpButtons();
	VelMagButtonCallBack(); //default is vel mag contour
	SquareButtonCallBack(); //default is square shape
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
}


void InitializeButtonCallBack()
{
	float4 *dptr;
	cudaGraphicsMapResources(1, &g_cudaSolutionField, 0);
	size_t num_bytes,num_bytes2;
	cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, g_cudaSolutionField);

	float u = Window.GetSlider("Slider_InletV")->m_sliderBar1->GetValue();
	InitializeDomain(dptr, g_fA_d, g_im_d, g_xDim, g_yDim, u, g_xDimVisible, g_yDimVisible);
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

void SetUpButtons()
{
	Window.GetButton("Initialize")->m_callBack = InitializeButtonCallBack;
	Window.GetButton("Velocity Magnitude")->m_callBack = VelMagButtonCallBack;
	Window.GetButton("X Velocity")->m_callBack = VelXButtonCallBack;
	Window.GetButton("Y Velocity")->m_callBack = VelYButtonCallBack;
	Window.GetButton("StrainRate")->m_callBack = StrainRateButtonCallBack;
	Window.GetButton("Pressure"  )->m_callBack = PressureButtonCallBack;
//	Window.GetButton("Initialize"        )->m_displayText = "Initialize"        ;
//	Window.GetButton("Velocity Magnitude")->m_displayText = "Velocity Magnitude";
//	Window.GetButton("X Velocity"        )->m_displayText = "X Velocity"        ;
//	Window.GetButton("Y Velocity"        )->m_displayText = "Y Velocity"        ;
//	Window.GetButton("StrainRate"        )->m_displayText = "Strain Rate"       ;
//	Window.GetButton("Pressure"          )->m_displayText = "Pressure"          ;

	std::vector<Button*> buttons = {
		Window.GetButton("Velocity Magnitude"),
		Window.GetButton("X Velocity"),
		Window.GetButton("Y Velocity"),
		Window.GetButton("StrainRate"),
		Window.GetButton("Pressure") };
	contourButtons = ButtonGroup(buttons);


	//Shape buttons
	Window.GetButton("Square")->m_callBack = SquareButtonCallBack;
	Window.GetButton("Circle")->m_callBack = CircleButtonCallBack;
	Window.GetButton("Hor. Line")->m_callBack = HorLineButtonCallBack;
	Window.GetButton("Vert. Line")->m_callBack = VertLineButtonCallBack;

//	Window.GetButton("Square"        )->m_displayText = "Square"        ;
//	Window.GetButton("Circle"        )->m_displayText = "Circle"        ;
//	Window.GetButton("Hor. Line"     )->m_displayText = "Hor. Line"     ;
//	Window.GetButton("Vert. Line"    )->m_displayText = "Vert. Line"    ;

	std::vector<Button*> buttons2 = {
		Window.GetButton("Square"),
		Window.GetButton("Circle"),
		Window.GetButton("Hor. Line"),
		Window.GetButton("Vert. Line") };
	shapeButtons = ButtonGroup(buttons2);
}

void DrawShapePreview()
{
	Panel* previewPanel = Window.GetPanel("DrawingPreview");
	float centerX = previewPanel->m_rectFloat_abs.GetCentroidX();
	float centerY = previewPanel->m_rectFloat_abs.GetCentroidY();
	float graphicsToWindowScaleFactor = static_cast<float>(winw)/Window.GetPanel("Graphics")->m_rectInt_abs.m_w;
	//float r1 = static_cast<float>(g_currentSize)/g_xDim*2.f*graphicsToWindowScaleFactor;

	int r1ix = g_currentSize*static_cast<float>(Window.GetPanel("Graphics")->m_rectInt_abs.m_w) / (g_xDimVisible); //r1x in pixels
	int r1iy = g_currentSize*static_cast<float>(Window.GetPanel("Graphics")->m_rectInt_abs.m_h) / (g_yDimVisible); //r1x in pixels
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
		r1fy = static_cast<float>(LINE_OBST_WIDTH) / winw*2.f;
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


void GenerateIndexList(){

	g_elementArrayIndices = new int[(MAX_XDIM-1)*(MAX_YDIM-1) * 4];
	for (int j = 0; j < MAX_YDIM-1; j++){
		for (int i = 0; i < MAX_XDIM-1; i++){
			//going clockwise, since y orientation will be flipped when rendered
			g_elementArrayIndices[j*(MAX_XDIM-1)*4+i * 4 + 0] = (i)+(j)*MAX_XDIM;
			g_elementArrayIndices[j*(MAX_XDIM-1)*4+i * 4 + 1] = (i + 1) + (j)*MAX_XDIM;
			g_elementArrayIndices[j*(MAX_XDIM-1)*4+i * 4 + 2] = (i+1)+(j + 1)*MAX_XDIM;
			g_elementArrayIndices[j*(MAX_XDIM-1)*4+i * 4 + 3] = (i)+(j + 1)*MAX_XDIM;
		}
	}

	glGenBuffers(1, &g_elementArrayIndexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_elementArrayIndexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*(MAX_XDIM-1)*(MAX_YDIM-1)*4, g_elementArrayIndices, GL_DYNAMIC_DRAW);
}

void CleanUpIndexList(){
	free(g_elementArrayIndices);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &g_elementArrayIndexBuffer);
}


void SetUpGLInterop()
{
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	GenerateIndexList();
	unsigned int solutionMemorySize = MAX_XDIM*MAX_YDIM * 4 * sizeof(float);
	//unsigned int solutionMemorySize = g_xDim*g_yDim * 4 * sizeof(float);
	CreateVBO(&g_vboSolutionField, &g_cudaSolutionField, solutionMemorySize, cudaGraphicsMapFlagsWriteDiscard);
}

void CleanUpGLInterop()
{
	CleanUpIndexList();
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
	//if(y == 0 || x == XDIM-1 || y == YDIM-1)
	if (x < 0.1f)
		return 3;//west
	else if ((g_xDim - x) < 1.1f)
		return 2;//east
	else if ((g_yDim - y) < 1.1f)
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
	size_t memsize, memsize_int, memsize_inputs;
	g_uMax = 0.06f;

	int domainSize = ((MAX_XDIM + BLOCKSIZEX - 1) / BLOCKSIZEX)*(MAX_YDIM / BLOCKSIZEY)
						*BLOCKSIZEX*BLOCKSIZEY;
	memsize = domainSize*sizeof(float)*9;
	memsize_int = domainSize*sizeof(int);
	memsize_inputs = sizeof(g_obstructions);

	g_fA_h = (float *)malloc(memsize);
	g_fB_h = (float *)malloc(memsize);
	g_im_h = (int *)malloc(memsize_int);
	//obstructions = (input_values *)malloc(memsize_inputs);

	cudaMalloc((void **)&g_fA_d, memsize);
	cudaMalloc((void **)&g_fB_d, memsize);
	cudaMalloc((void **)&g_im_d, memsize_int);
	cudaMalloc((void **)&g_obst_d, memsize_inputs);

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
	}	
	g_obstructions[0].r1 = 10;
	g_obstructions[0].x = g_xDim*0.33f;
	g_obstructions[0].y = g_yDim*0.5f;
	g_obstructions[0].shape = Obstruction::CIRCLE;

//	for (int i = 0; i < domainSize; i++)
//	{
//		int x = i%MAX_XDIM;
//		int y = i/MAX_XDIM;
//		g_im_h[i] = ImageFcn_h(x, y, g_obstructions);
//	}
	UpdateDeviceImage();
	
	cudaMemcpy(g_fA_d, g_fA_h, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(g_fB_d, g_fB_h, memsize, cudaMemcpyHostToDevice);
//	cudaMemcpy(g_im_d, g_im_h, memsize_int, cudaMemcpyHostToDevice);
	cudaMemcpy(g_obst_d, g_obstructions, memsize_inputs, cudaMemcpyHostToDevice);

	//writeInputs();
	float u = Window.GetSlider("Slider_InletV")->m_sliderBar1->GetValue();

	float4 *dptr;
	cudaGraphicsMapResources(1, &g_cudaSolutionField, 0);
	size_t num_bytes,num_bytes2;
	cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, g_cudaSolutionField);

	InitializeDomain(dptr, g_fA_d, g_im_d, MAX_XDIM, MAX_YDIM, u, g_xDimVisible, g_yDimVisible);
	InitializeDomain(dptr, g_fB_d, g_im_d, MAX_XDIM, MAX_YDIM, u, g_xDimVisible, g_yDimVisible);
}

void RunCuda(struct cudaGraphicsResource **vbo_resource)
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

	MarchSolution(dptr, g_fA_d, g_fB_d, g_im_d, g_obst_d, g_contourVar, g_contMin, g_contMax, g_xDim, g_yDim, u, omega, g_tStep, g_xDimVisible, g_yDimVisible);

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
		translate_z += 1.f;
		g_xDim += 64;
	}
	else
	{
		translate_z -= 1.f;
		g_xDim -= 64;
	}
	UpdateDeviceImage();
	
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

	g_xDim = min(max(BLOCKSIZEX, ceil(((static_cast<float>(windowWidth - leftPanelWidth)/scaleUp)/BLOCKSIZEX))*BLOCKSIZEX),MAX_XDIM);
	g_yDim = min(max(1, ceil(static_cast<float>(windowHeight) / scaleUp)),MAX_YDIM);
	g_xDimVisible = min(max(BLOCKSIZEX, ((static_cast<float>(windowWidth - leftPanelWidth)/scaleUp))),MAX_XDIM);
	g_yDimVisible = g_yDim;
}

void Resize(int w, int h)
{
	int area = w*h;
	//UpdateWindowDimensionsBasedOnAspectRatio(winh, winw, area, max(g_leftPanelHeight,g_drawingPanelHeight),g_leftPanelWidth+g_drawingPanelWidth, g_xDim, g_yDim, g_initialScaleUp);
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
		if (g_timeStepsPerSecond > timeStepsPerSecond_prev*1.2f){
			g_tStep = max(1, g_tStep - 1);
		}
	}
	char fpsReport[256];
	sprintf(fpsReport, "Interactive CFD running at: %i timesteps/frame at %3.1f fps = %3.1f timesteps/second on %ix%i mesh", g_tStep * 2, g_fps, g_timeStepsPerSecond, g_xDim, g_yDim);
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

	RunCuda(&g_cudaSolutionField);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
//	glEnable(GL_LIGHTING);

	/*
	 *	Set perspective viewing transformation
	 */
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1,1,-1,1,-100,20);

	int graphicsViewWidth = winw - g_leftPanelWidth - g_drawingPanelWidth;
	int graphicsViewHeight = winh;
	float xTranslation = -((static_cast<float>(winw)-g_xDimVisible*g_initialScaleUp)*0.5 - static_cast<float>(g_leftPanelWidth + g_drawingPanelWidth)) / winw*2.f;
	float yTranslation = -((static_cast<float>(winh)-g_yDimVisible*g_initialScaleUp)*0.5)/ winh*2.f;
	glTranslatef(xTranslation,yTranslation,0.f);
	glScalef((static_cast<float>(g_xDimVisible*g_initialScaleUp) / winw), (static_cast<float>(g_yDimVisible*g_initialScaleUp) / winh), 1.f);
	//glScalef((static_cast<float>(winw-g_leftPanelWidth-g_drawingPanelWidth) / winw), 1.f, 1.f);
	//glScalef((static_cast<float>(g_xDim) / winw), 1.f, 1.f);
	//glScalef((static_cast<float>(g_xDim) / (g_xDim+g_leftPanelWidth)), 1.f, 1.f);
	//glTranslatef(-(1.f - static_cast<float>(winw) / (winw-g_leftPanelWidth-g_drawingPanelWidth)),0.f,0.f);
	//glTranslatef(-(1.f - (g_xDim+g_leftPanelWidth) / (static_cast<float>(g_xDim) )),0.f,0.f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, g_vboSolutionField);
	glVertexPointer(3, GL_FLOAT, 16, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(4, GL_UNSIGNED_BYTE, 16, (char *)NULL + 12);

	glDrawElements(GL_QUADS, (MAX_XDIM - 1)*(MAX_YDIM - 1) * 4, GL_UNSIGNED_INT, (GLvoid*)0);

	glDisableClientState(GL_VERTEX_ARRAY);

	/*
	 *	Draw the 3D elements in the scene
	 */
	//Draw3D();

	/*
	 *	Disable depth test and lighting for 2D elements
	 */
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);


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

	//g_xDim = 256;
	//g_yDim = 192;

}

int main(int argc,char **argv)
{
	SetUpWindow();

	glutInit(&argc,argv);


	glutInitDisplayMode(GLUT_RGB|GLUT_DEPTH|GLUT_DOUBLE);
	glutInitWindowSize(g_initialScaleUp*g_xDim+g_leftPanelWidth,g_initialScaleUp*g_yDim);
	glutInitWindowPosition(200,100);
	glutCreateWindow("Interactive CFD");


	glutDisplayFunc(Draw);
	glutReshapeFunc(Resize);
	glutMouseFunc(MouseButton);
	glutMotionFunc(MouseMotion);
//	glutPassiveMotionFunc(MousePassiveMotion);
	glutMouseWheelFunc(MouseWheel);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	Init();
	SetUpGLInterop();
	SetUpCUDA();

	glutMainLoop();

	return 0;
}