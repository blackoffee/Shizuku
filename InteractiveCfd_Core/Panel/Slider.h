#pragma once 
#include "Panel.h"

#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class SliderBar;

class FW_API Slider : public Panel
{
public:
    SliderBar* m_sliderBar1 = NULL;
    SliderBar* m_sliderBar2 = NULL;

    Slider(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Panel* parent = NULL);
    Slider(const RectInt rectInt    , const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Panel* parent = NULL);

    void CreateSliderBar(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color);

    void UpdateAll();
    
    void Draw();
    void DrawAll();
    void Hide();
    void Show();
};