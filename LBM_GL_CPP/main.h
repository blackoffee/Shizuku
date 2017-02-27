#pragma once 
#include <GLEW/glew.h>
#include <GLUT/freeglut.h>
#include <string>
#include <iostream>
#include <vector>
#include "Panel.h"
#include "GraphicsManager.h"
#include "RectFloat.h"
#include "RectInt.h"

#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

FW_API void UpdateWindowDimensionsBasedOnAspectRatio(int& heightOut, int& widthOut, int area,
    int leftPanelHeight, int leftPanelWidth, int xDim, int yDim, float scaleUp);
FW_API void UpdateDomainDimensionsBasedOnWindowSize(int leftPanelHeight, int leftPanelWidth,
    int windowWidth, int windowHeight, float scaleUp);

FW_API void SetUpWindow(Panel &rootPanel);
FW_API Slider* GetCurrentContourSlider(Panel &rootPanel);
FW_API void InitializeButtonCallBack(Panel &rootPanel);

FW_API void VelMagButtonCallBack(Panel &rootPanel);
FW_API void VelXButtonCallBack(Panel &rootPanel);
FW_API void VelYButtonCallBack(Panel &rootPanel);
FW_API void StrainRateButtonCallBack(Panel &rootPanel);
FW_API void PressureButtonCallBack(Panel &rootPanel);
FW_API void WaterRenderingButtonCallBack(Panel &rootPanel);
FW_API void SquareButtonCallBack(Panel &rootPanel);
FW_API void CircleButtonCallBack(Panel &rootPanel);
FW_API void HorLineButtonCallBack(Panel &rootPanel);
FW_API void VertLineButtonCallBack(Panel &rootPanel);
FW_API void ThreeDButtonCallBack(Panel &rootPanel);
FW_API void TwoDButtonCallBack(Panel &rootPanel);
FW_API void SetUpButtons(Panel &rootPanel);
FW_API void DrawShapePreview(Panel &rootPanel);

