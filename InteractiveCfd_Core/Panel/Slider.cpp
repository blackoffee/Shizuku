#include "Slider.h"
#include "SliderBar.h"
#include <GLEW/glew.h>

Slider::Slider(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
    const std::string name, const Color color, Panel* parent) 
    : Panel(rectFloat, sizeDefinition, name, color, parent)
{
}

Slider::Slider(const RectInt rectInt, const SizeDefinitionMethod sizeDefinition,
    const std::string name, const Color color, Panel* parent) 
    : Panel(rectInt, sizeDefinition, name, color, parent)
{
}

void Slider::CreateSliderBar(const RectFloat rectFloat,
    const SizeDefinitionMethod sizeDefinition, const std::string name, const Color color)
{
    SliderBar* slider = new SliderBar(rectFloat, sizeDefinition, name, color, this);
    if (m_sliderBar1 == NULL)
    {
        m_sliderBar1 = slider;
    }
    else if (m_sliderBar2 == NULL)
    {
        m_sliderBar2 = slider;
    }
}

void Slider::UpdateAll()
{
    Update();
    if (m_sliderBar1 != NULL)
    {
        m_sliderBar1->Update();
    }
    if (m_sliderBar2 != NULL)
    {
        m_sliderBar2->Update();
    }
}

void Slider::Draw()
{
    Color minColor, maxColor;
    RectFloat rect = this->GetRectFloatAbs();
    if (m_sliderBar2 == NULL)
    {
        if (m_sliderBar1->GetOrientation() == SliderBar::VERTICAL)
        {
            minColor = Color::BLUE;
            maxColor = Color::WHITE;
            glBegin(GL_QUADS);
            glColor3f(maxColor.r, maxColor.g, maxColor.b);
            glVertex2f(rect.m_x, rect.m_y + rect.m_h);
            glColor3f(minColor.r, minColor.g, minColor.b);
            glVertex2f(rect.m_x, rect.m_y);
            glVertex2f(rect.m_x + rect.m_w, rect.m_y);
            glColor3f(maxColor.r, maxColor.g, maxColor.b);
            glVertex2f(rect.m_x + rect.m_w, rect.m_y + rect.m_h);
            glEnd();
        }
        else
        {
            minColor = Color::BLUE;
            maxColor = Color::WHITE;
            glBegin(GL_QUADS);
            glColor3f(minColor.r, minColor.g, minColor.b);
            glVertex2f(rect.m_x, rect.m_y + rect.m_h);
            glVertex2f(rect.m_x, rect.m_y);
            glColor3f(maxColor.r, maxColor.g, maxColor.b);
            glVertex2f(rect.m_x + rect.m_w, rect.m_y);
            glVertex2f(rect.m_x + rect.m_w, rect.m_y + rect.m_h);
            glEnd();

        }
    }
    else{ //this doesn't work for horizontal sliders yet
        minColor = Color::BLUE;
        maxColor = Color::WHITE;
        Color lowerColor, higherColor;
        SliderBar* lowerSliderBar;
        SliderBar* higherSliderBar;
        if (m_sliderBar1->GetValue() < m_sliderBar2->GetValue())
        {
            lowerColor = m_sliderBar1->GetForegroundColor();
            higherColor = m_sliderBar2->GetForegroundColor();
            lowerSliderBar = m_sliderBar1;
            higherSliderBar = m_sliderBar2;
        }
        else
        {
            lowerColor = m_sliderBar2->GetForegroundColor();
            higherColor = m_sliderBar1->GetForegroundColor();
            lowerSliderBar = m_sliderBar2;
            higherSliderBar = m_sliderBar1;
        }
        if (m_sliderBar1->GetOrientation() == SliderBar::VERTICAL)
        {
            glBegin(GL_QUADS);
            glColor3f(lowerColor.r, lowerColor.g, lowerColor.b);
            glVertex2f(rect.m_x, lowerSliderBar->GetRectFloatAbs().GetCentroidY());
            glColor3f(lowerColor.r, lowerColor.g, lowerColor.b);
            glVertex2f(rect.m_x, rect.m_y);
            glVertex2f(rect.m_x + rect.m_w, rect.m_y);
            glColor3f(lowerColor.r, lowerColor.g, lowerColor.b);
            glVertex2f(rect.m_x + rect.m_w, lowerSliderBar->GetRectFloatAbs().GetCentroidY());
            glEnd();

            glBegin(GL_QUADS);
            glColor3f(higherColor.r, higherColor.g, higherColor.b);
            glVertex2f(rect.m_x, higherSliderBar->GetRectFloatAbs().GetCentroidY());
            glColor3f(lowerColor.r, lowerColor.g, lowerColor.b);
            glVertex2f(rect.m_x, lowerSliderBar->GetRectFloatAbs().GetCentroidY());
            glVertex2f(rect.m_x + rect.m_w, lowerSliderBar->GetRectFloatAbs().GetCentroidY());
            glColor3f(higherColor.r, higherColor.g, higherColor.b);
            glVertex2f(rect.m_x + rect.m_w, higherSliderBar->GetRectFloatAbs().GetCentroidY());
            glEnd();


            glBegin(GL_QUADS);
            glColor3f(higherColor.r, higherColor.g, higherColor.b);
            glVertex2f(rect.m_x, rect.m_y+rect.m_h);
            glColor3f(higherColor.r, higherColor.g, higherColor.b);
            glVertex2f(rect.m_x, higherSliderBar->GetRectFloatAbs().GetCentroidY());
            glVertex2f(rect.m_x + rect.m_w, higherSliderBar->GetRectFloatAbs().GetCentroidY());
            glColor3f(higherColor.r, higherColor.g, higherColor.b);
            glVertex2f(rect.m_x + rect.m_w, rect.m_y+rect.m_h);
            glEnd();
        }
        else
        {
            glBegin(GL_QUADS);
            glColor3f(lowerColor.r, lowerColor.g, lowerColor.b);
            glVertex2f(rect.m_x, rect.m_y+rect.m_h);
            glVertex2f(rect.m_x, rect.m_y);
            glVertex2f(lowerSliderBar->GetRectFloatAbs().GetCentroidX(), rect.m_y);
            glVertex2f(lowerSliderBar->GetRectFloatAbs().GetCentroidX(), rect.m_y+rect.m_h);
            glEnd();

            glBegin(GL_QUADS);
            glColor3f(lowerColor.r, lowerColor.g, lowerColor.b);
            glVertex2f(lowerSliderBar->GetRectFloatAbs().GetCentroidX(), rect.m_y+rect.m_h);
            glVertex2f(lowerSliderBar->GetRectFloatAbs().GetCentroidX(), rect.m_y);
            glColor3f(higherColor.r, higherColor.g, higherColor.b);
            glVertex2f(higherSliderBar->GetRectFloatAbs().GetCentroidX(), rect.m_y);
            glVertex2f(higherSliderBar->GetRectFloatAbs().GetCentroidX(), rect.m_y+rect.m_h);
            glEnd();

            glBegin(GL_QUADS);
            glColor3f(higherColor.r, higherColor.g, higherColor.b);
            glVertex2f(higherSliderBar->GetRectFloatAbs().GetCentroidX(), rect.m_y+rect.m_h);
            glVertex2f(higherSliderBar->GetRectFloatAbs().GetCentroidX(), rect.m_y);
            glVertex2f(rect.m_x+rect.m_w, rect.m_y);
            glVertex2f(rect.m_x+rect.m_w, rect.m_y+rect.m_h);
            glEnd();
        }
    }
}

void Slider::DrawAll()
{
    if (m_draw == true)
    {
        Draw();
        if (m_sliderBar1 != NULL)
        {
            m_sliderBar1->Draw();
        }
        if (m_sliderBar2 != NULL)
        {
            m_sliderBar2->Draw();
        }
    }
}

void Slider::Hide()
{
    m_draw = false;
    if (m_sliderBar1 != NULL)
    {
        m_sliderBar1->m_draw = false;
    }
    if (m_sliderBar2 != NULL)
    {
        m_sliderBar2->m_draw = false;
    }
}

void Slider::Show()
{
    m_draw = true;
    if (m_sliderBar1 != NULL)
    {
        m_sliderBar1->m_draw = true;
    }
    if (m_sliderBar2 != NULL)
    {
        m_sliderBar2->m_draw = true;
    }
}

