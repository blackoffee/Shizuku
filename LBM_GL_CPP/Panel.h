#pragma once 
#include <GLEW/glew.h>
#include <GLUT/freeglut.h>
#include <string>
#include <iostream>
#include <vector>
#include "GraphicsManager.h"
#include "RectFloat.h"
#include "RectInt.h"


#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

FW_API void test();


class Color
{
public:
    FW_API enum ColorName{WHITE,BLACK,RED,GREEN,BLUE,DARK_GRAY,GRAY,LIGHT_GRAY};
    float r = 1.f;
    float g = 1.f;
    float b = 1.f;
    FW_API Color();
    FW_API Color(ColorName color);
};

class Button;
class Slider;
class GraphicsManager;
class Mouse;
class ButtonGroup;

class Panel
{
public:
    enum SizeDefinitionMethod {DEF_ABS, DEF_REL};
private:
    std::string m_name;
    RectInt m_rectInt_abs; //absolute coordinates in Window
    RectInt m_rectInt_rel; //relative coordinates wrt to parent
    RectFloat m_rectFloat_abs; //absolute coordinates in window. this is the one used for drawing, so always want to keep this up-to-date.
    RectFloat m_rectFloat_rel;
    Color m_backgroundColor;
    Color m_foregroundColor;
    float m_minValue = 1.f;
    float m_maxValue = 0.f;
    std::string m_displayText = "";
    SizeDefinitionMethod m_sizeDefinition;
protected:
    void(*m_callback)(Panel &rootPanel) = NULL;
public:
    std::vector<Panel*> m_subPanels;
    std::vector<Button*> m_buttons;
    std::vector<Slider*> m_sliders;
    std::vector<ButtonGroup*> m_buttonGroups;
    Panel* m_parent = NULL; //pointer to parent frame
    GraphicsManager* m_graphicsManager = NULL;
    bool m_draw = true;
    //these two members below should ideally be in Slider class

    FW_API Panel();
    FW_API Panel(const RectInt rectInt  , const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Panel* parent = NULL);
    FW_API Panel(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Panel* parent = NULL);

    FW_API std::string GetName();
    FW_API void SetName(const std::string name);

    FW_API void SetSize_Relative(const RectFloat);
    FW_API void SetSize_Absolute(const RectFloat);
    FW_API void SetSize_Absolute(const RectInt);

    FW_API RectFloat GetRectFloatAbs();
    FW_API RectInt GetRectIntAbs();

    FW_API int GetWidth();
    FW_API int GetHeight();
    FW_API Panel* GetRootPanel();
    FW_API Panel* GetPanel(const std::string name);
    FW_API Button* GetButton(const std::string name);
    FW_API Slider* GetSlider(const std::string name);
    FW_API ButtonGroup* GetButtonGroup(const std::string name);

    FW_API void SetCallback(void(*callback)(Panel &rootPanel));
    
    FW_API void SetBackgroundColor(Color color);
    FW_API void SetForegroundColor(Color color);
    FW_API Color GetBackgroundColor();
    FW_API Color GetForegroundColor();

    FW_API void SetMinValue(float value);
    FW_API void SetMaxValue(float value);
    FW_API float GetMinValue();
    FW_API float GetMaxValue();

    FW_API std::string GetDisplayText();
    FW_API void SetDisplayText(std::string displayText);


    FW_API void CreateGraphicsManager();

    FW_API Panel* CreateSubPanel(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color);
    FW_API Panel* CreateSubPanel(const RectInt rectInt    , const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color);
    FW_API Button* CreateButton(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color);
    FW_API Button* CreateButton(const RectInt rectInt    , const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color);
    FW_API Slider* CreateSlider(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color);
    FW_API ButtonGroup* CreateButtonGroup(const std::string name, std::vector<Button*> &buttons);

    FW_API RectFloat RectIntAbsToRectFloatAbs();
    FW_API RectFloat RectFloatRelToRectFloatAbs();
    FW_API RectFloat RectFloatAbsToRectFloatRel();

    FW_API virtual void Update();
    FW_API virtual void UpdateAll();
    
    FW_API virtual void Draw(); //draw current panel only
    FW_API virtual void DrawAll(); //draw current panel, then invoke DrawAll on immediate children. Effectively draws all subpanels

    FW_API virtual void Drag(const int x, const int y, const float dx, const float dy, const int button);
    FW_API virtual void Wheel(const int button, const int dir, const int x, const int y);
    FW_API virtual void ClickDown(Mouse mouse);
    FW_API virtual void ClickDown();

    FW_API ~Panel();
};

class Button : public Panel
{
public:
    bool m_highlighted = false;
    
    using Panel::Panel;
    FW_API Button(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Panel* parent = NULL);
    FW_API Button(const RectInt rectInt    , const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Panel* parent = NULL);

    virtual void ClickDown(Mouse mouse);
    virtual void ClickDown();
};

class Slider;

class SliderBar : public Panel
{
public:
    enum Orientation {VERTICAL, HORIZONTAL};
    Orientation m_orientation = HORIZONTAL;
    float m_value = 0.5f;

    FW_API SliderBar();
    FW_API SliderBar(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Slider* parent = NULL);
    FW_API SliderBar(const RectInt rectInt    , const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Slider* parent = NULL);

    FW_API void Draw();
    FW_API void UpdateValue();
    FW_API float GetValue();
    virtual void Drag(int x, int y, float dx, float dy, int button);
};

class Slider : public Panel
{
public:
    float m_currentValue;
    SliderBar* m_sliderBar1 = NULL;
    SliderBar* m_sliderBar2 = NULL;

    FW_API Slider(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Panel* parent = NULL);
    FW_API Slider(const RectInt rectInt    , const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Panel* parent = NULL);

    FW_API void CreateSliderBar(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color);

    FW_API void UpdateAll();
    
    FW_API void Draw();
    FW_API void DrawAll();
    FW_API void Hide();
    FW_API void Show();
};

class ButtonGroup
{
    std::string m_name;
    std::vector<Button*> m_buttons;
public:
    FW_API ButtonGroup();
    FW_API ButtonGroup(const std::string name, std::vector<Button*> &buttons);

    FW_API std::string GetName();
    FW_API void AddButton(Button* button);
    FW_API std::vector<Button*> GetButtons(Button* button);
    FW_API void ExclusiveEnable(Button* button);
    FW_API Button* GetCurrentEnabledButton();
};

FW_API Panel* GetPanelThatPointIsIn(Panel* parentPanel, float x, float y);