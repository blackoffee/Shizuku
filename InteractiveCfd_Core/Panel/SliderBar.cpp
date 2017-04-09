#include "SliderBar.h"
#include "Slider.h"
#include <GLEW/glew.h>
#include <algorithm>

SliderBar::SliderBar()
{
}

SliderBar::SliderBar(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
    const std::string name, const Color color, Slider* parent) 
    : Panel(rectFloat, sizeDefinition, name, color, parent)
{
    SetBackgroundColor(Color::GRAY);
}

void SliderBar::Draw()
{
    RectFloat rect = this->GetRectFloatAbs();
    Color backgroundColor = GetBackgroundColor();
    glColor3f(backgroundColor.r, backgroundColor.g, backgroundColor.b);
    glBegin(GL_QUADS);
    glVertex2f(rect.m_x, rect.m_y + rect.m_h);
    glVertex2f(rect.m_x, rect.m_y);
    glVertex2f(rect.m_x + rect.m_w, rect.m_y);
    glVertex2f(rect.m_x + rect.m_w, rect.m_y + rect.m_h);
    glEnd();

    Color foregroundColor = GetForegroundColor();
    glColor3f(foregroundColor.r, foregroundColor.g, foregroundColor.b);
    float outlineWidth = 0.003f;
    glBegin(GL_QUADS);
    glVertex2f(rect.m_x + outlineWidth, rect.m_y + rect.m_h - outlineWidth*2.f);
    glVertex2f(rect.m_x + outlineWidth, rect.m_y + outlineWidth*2.f);
    glVertex2f(rect.m_x + rect.m_w - outlineWidth, rect.m_y + outlineWidth*2.f);
    glVertex2f(rect.m_x + rect.m_w - outlineWidth, rect.m_y + rect.m_h - outlineWidth*2.f);
    glEnd();
}

void SliderBar::UpdateValue()
{
    RectFloat rect = this->GetRectFloatAbs();
    RectFloat parentRect = m_parent->GetRectFloatAbs();
    if (m_orientation == VERTICAL)
    {
        m_value = m_parent->GetMinValue() + (m_parent->GetMaxValue() - m_parent->GetMinValue())*
            (rect.GetCentroidY() - (parentRect.m_y+rect.m_h*0.5f)) /
            (parentRect.m_h-rect.m_h);
    }
    else
    {
        m_value = m_parent->GetMinValue() + (m_parent->GetMaxValue() - m_parent->GetMinValue())*
            (rect.GetCentroidX() - (parentRect.m_x+rect.m_w*0.5f)) /
            (parentRect.m_w-rect.m_w);
    }
}

float SliderBar::GetValue()
{
    UpdateValue();
    return m_value;
}

SliderBar::Orientation SliderBar::GetOrientation()
{
    return m_orientation;
}

void SliderBar::Drag(int x, int y, float dx, float dy, int button)
{
    RectFloat rect = this->GetRectFloatAbs();
    RectFloat parentRect = m_parent->GetRectFloatAbs();
    //dx and dy are coming in as float abs coordinates
    if (m_orientation == VERTICAL)
    {
        rect.m_y = std::max(parentRect.m_y, 
            std::min(parentRect.m_y + parentRect.m_h - rect.m_h, rect.m_y + dy));
    }
    else
    { 
        rect.m_x = std::max(parentRect.m_x,
            std::min(parentRect.m_x + parentRect.m_w - rect.m_w, rect.m_x + dx));
    }
    SetSize_Absolute(rect);
    UpdateValue();
}
