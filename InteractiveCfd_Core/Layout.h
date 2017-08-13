#pragma once
#include <string>

#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class Panel;
class Slider;

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

namespace Layout
{
    FW_API void SetUpWindow(Panel &rootPanel);
    FW_API Slider* GetCurrentContourSlider(Panel &rootPanel);
    FW_API float GetCurrentSliderValue(Panel &rootPanel, const std::string name, const int sliderNumber = 1);
    FW_API float GetCurrentContourSliderValue(Panel &rootPanel, const int sliderNumber = 1);
    FW_API void GetCurrentContourSliderBoundValues(Panel &rootPanel, float &minValue, float &maxValue);
    FW_API void SetUpButtons(Panel &rootPanel);
    FW_API void Draw2D(Panel &rootPanel);
    FW_API void DrawShapePreview(Panel &rootPanel);
};

