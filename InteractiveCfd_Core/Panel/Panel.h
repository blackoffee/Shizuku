#pragma once 
#include "RectFloat.h"
#include "RectInt.h"
#include <string>
#include <vector>

#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class FW_API Color
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
class ButtonGroup;
class GraphicsManager;

class FW_API Panel
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
    SizeDefinitionMethod m_sizeDefinition = DEF_ABS;
    GraphicsManager* m_graphicsManager = NULL;
protected:
    void(*m_callback)(Panel &rootPanel) = NULL;
    std::vector<Panel*> m_subPanels;
    std::vector<Button*> m_buttons;
    std::vector<Slider*> m_sliders;
    std::vector<ButtonGroup*> m_buttonGroups;
    Panel* m_parent = NULL; //pointer to parent frame
public:
    bool m_draw = true;
    //these two members below should ideally be in Slider class

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
    ButtonGroup* GetButtonGroup(const std::string name);
    std::vector<Panel*>& GetSubPanels();
    std::vector<Button*>& GetButtons();
    std::vector<Slider*>& GetSliders();
    std::vector<ButtonGroup*>& GetButtonGroups();

    GraphicsManager* GetGraphicsManager();

    void SetBackgroundColor(Color color);
    void SetForegroundColor(Color color);
    Color GetBackgroundColor();
    Color GetForegroundColor();

    void SetMinValue(float value);
    void SetMaxValue(float value);
    float GetMinValue();
    float GetMaxValue();

    std::string GetDisplayText();
    void SetDisplayText(std::string displayText);

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
    ButtonGroup* CreateButtonGroup(const std::string name, std::vector<Button*> &buttons);

    RectFloat RectIntAbsToRectFloatAbs();
    RectFloat RectFloatRelToRectFloatAbs();
    RectFloat RectFloatAbsToRectFloatRel();

    virtual void Update();
    virtual void UpdateAll();
    
    virtual void Draw(); //draw current panel only
    virtual void DrawAll(); //draw current panel, then invoke DrawAll on immediate children. Effectively draws all subpanels

    virtual void Drag(const int x, const int y, const float dx, const float dy, const int button);
    virtual void ClickDown();

    ~Panel();
};

FW_API float intCoordToFloatCoord(const int x, const int xDim);
FW_API int floatCoordToIntCoord(const float x, const int xDim);

FW_API Panel* GetPanelThatPointIsIn(Panel* parentPanel, float x, float y);