#pragma once 
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <string>
#include <iostream>
#include <vector>
#include "GraphicsManager.h"
#include "RectFloat.h"
#include "RectInt.h"

class Color
{
public:
    enum ColorName{WHITE,BLACK,RED,GREEN,BLUE,DARK_GRAY,GRAY,LIGHT_GRAY};
    float r = 1.f;
    float g = 1.f;
    float b = 1.f;
    Color();
    Color(ColorName color);
};

class Button;
class Slider;
class GraphicsManager;
class Mouse;

class Panel
{
private:
    std::string m_name;
    RectInt m_rectInt_abs; //absolute coordinates in Window
    RectInt m_rectInt_rel; //relative coordinates wrt to parent
    RectFloat m_rectFloat_abs; //absolute coordinates in window. this is the one used for drawing, so always want to keep this up-to-date.
    RectFloat m_rectFloat_rel;
public:
    enum SizeDefinitionMethod {DEF_ABS, DEF_REL};
    std::vector<Panel*> m_subPanels;
    std::vector<Button*> m_buttons;
    std::vector<Slider*> m_sliders;
    Panel* m_parent = NULL; //pointer to parent frame

    Color m_backgroundColor;
    Color m_foregroundColor;
    SizeDefinitionMethod m_sizeDefinition;
    GraphicsManager* m_graphicsManager = NULL;
    bool m_draw = true;
    void(*m_callBack)() = NULL;
    std::string m_displayText = "";
    //these two members below should ideally be in Slider class
    float m_minValue = 1.f;
    float m_maxValue = 0.f;



    Panel();
    Panel(const RectInt rectInt  , const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Panel* parent = NULL);
    Panel(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Panel* parent = NULL);

    std::string GetName();
    void SetName(const std::string name);

    void SetSize_Relative(const RectFloat);
    void SetSize_Absolute(const RectFloat);
    void SetSize_Absolute(const RectInt);

    RectFloat GetRectFloatAbs();
    RectInt GetRectIntAbs();

    int GetWidth();
    int GetHeight();
    Panel* GetRootPanel();
    Panel* GetPanel(const std::string name);
    Button* GetButton(const std::string name);
    Slider* GetSlider(const std::string name);

    void CreateGraphicsManager();

    Panel* CreateSubPanel(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color);
    Panel* CreateSubPanel(const RectInt rectInt    , const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color);
    Button* CreateButton(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color);
    Button* CreateButton(const RectInt rectInt    , const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color);
    Slider* CreateSlider(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color);

    RectFloat RectIntAbsToRectFloatAbs();
    RectFloat RectFloatRelToRectFloatAbs();
    RectFloat RectFloatAbsToRectFloatRel();

    virtual void Update();
    virtual void UpdateAll();
    
    virtual void Draw(); //draw current panel only
    virtual void DrawAll(); //draw current panel, then invoke DrawAll on immediate children. Effectively draws all subpanels

    virtual void Drag(const int x, const int y, const float dx, const float dy, const int button);
    virtual void Wheel(const int button, const int dir, const int x, const int y);
    virtual void ClickDown(Mouse mouse);
};

class Button : public Panel
{
public:
    bool m_highlighted = false;
    
    using Panel::Panel;
    Button(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Panel* parent = NULL);
    Button(const RectInt rectInt    , const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Panel* parent = NULL);

    virtual void ClickDown(Mouse mouse);
};

class Slider;

class SliderBar : public Panel
{
public:
    enum Orientation {VERTICAL, HORIZONTAL};
    Orientation m_orientation = HORIZONTAL;
    float m_value = 0.5f;

    SliderBar();
    SliderBar(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Slider* parent = NULL);
    SliderBar(const RectInt rectInt    , const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Slider* parent = NULL);

    void Draw();
    void UpdateValue();
    float GetValue();
    virtual void Drag(int x, int y, float dx, float dy, int button);
};

class Slider : public Panel
{
public:
    float m_currentValue;
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

class ButtonGroup
{
public:
    std::vector<Button*> m_buttons;
    ButtonGroup();
    ButtonGroup(std::vector<Button*> &buttons);

    void ExclusiveEnable(Button* button);
    Button* GetCurrentEnabledButton();
};

Panel* GetPanelThatPointIsIn(Panel* parentPanel, float x, float y);