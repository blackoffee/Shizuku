#pragma once
#include "Panel.h"

#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class FW_API Button : public Panel
{
public:
    bool m_highlighted = false;
    void SetCallback(void(*callback)(Panel &rootPanel));
    void Callback();
    using Panel::Panel;
    Button(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Panel* parent = NULL);
    Button(const RectInt rectInt    , const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Panel* parent = NULL);
    void SetHighlight(const bool state);

    virtual void ClickDown();
};