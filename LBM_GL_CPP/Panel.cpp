
#include <string>
#include <iostream>
#include <vector>
#include "RectFloat.h"
#include "RectInt.h"
#include "Panel.h"


Color::Color()
{
	r = 1.f; g = 1.f; b = 1.f; 
}

//ColorName{WHITE,BLACK,RED,GREEN,BLUE,DARK_GRAY,GRAY,LIGHT_GRAY};
Color::Color(ColorName color)
{
	r = 1.f; g = 1.f; b = 1.f; 
	switch (color){
	case WHITE:
		r = 1.f; g = 1.f; b = 1.f; break;
	case BLACK:
		r = 0.f; g = 0.f; b = 0.f; break;
	case RED:
		r = 1.f; g = 0.f; b = 0.f; break;
	case DARK_GRAY:
		r = 0.1f; g = 0.1f; b = 0.1f; break;
	case GRAY:
		r = 0.5f; g = 0.5f; b = 0.5f; break;
	case LIGHT_GRAY:
		r = 0.9f; g = 0.9f; b = 0.9f; break;
	}
}

Panel::Panel()
{
	m_name = "None";
}

Panel::Panel(RectInt rectInt, SizeDefinitionMethod sizeDefinition, std::string name, Panel* parent)
{
	m_name = name;
	if (parent != NULL)
	{
		m_parent = parent;
	}
	if (sizeDefinition == DEF_ABS)
	{
		m_rectInt_abs = rectInt;
		m_rectFloat_abs = RectIntAbsToRectFloatAbs();
	}
	else
	{
		//not fully supported yet. need converting function
		m_rectInt_rel = rectInt;
	}
}

Panel::Panel(RectFloat rectFloat, SizeDefinitionMethod sizeDefinition, std::string name, Panel* parent)
{
	m_name = name;
	if (parent != NULL)
	{
		m_parent = parent;
	}
	if (sizeDefinition == DEF_ABS)
	{
		m_rectFloat_abs = rectFloat;
	}
	else
	{
		m_rectFloat_rel = rectFloat;
		m_rectFloat_abs = RectFloatRelToRectFloatAbs();
	}
}

void Panel::CreateSubPanel(RectFloat rectFloat, SizeDefinitionMethod sizeDefinition, std::string name)
{
	Panel* subPanel = new Panel(rectFloat, sizeDefinition, name, this);
	m_subPanels.push_back(subPanel);
}

void Panel::CreateSubPanel(RectInt rectInt, SizeDefinitionMethod sizeDefinition, std::string name)
{
	Panel* subPanel = new Panel(rectInt, sizeDefinition, name, this);
	m_subPanels.push_back(subPanel);
}

RectFloat Panel::RectIntAbsToRectFloatAbs()
{
	Panel* rootPanel(this);
	bool isBasePanel(this->m_parent == NULL);
	while (rootPanel->m_parent != NULL)
	{
		rootPanel = rootPanel->m_parent;
	}
	int windowWidth = rootPanel->m_rectInt_abs.m_w;
	int windowHeight = rootPanel->m_rectInt_abs.m_h;
	RectFloat rectFloat;
	if (isBasePanel)
	{
		rectFloat.m_x = -1.f;
		rectFloat.m_y = -1.f;
	}
	else
	{
		rectFloat.m_x = static_cast<float> (m_rectInt_abs.m_x) / rootPanel->m_rectInt_abs.m_w*2.f - 1.f;
		rectFloat.m_y = static_cast<float> (m_rectInt_abs.m_y) / rootPanel->m_rectInt_abs.m_h*2.f - 1.f;
	}
	rectFloat.m_w = static_cast<float> (m_rectInt_abs.m_w) / rootPanel->m_rectInt_abs.m_w*2.f;
	rectFloat.m_h = static_cast<float> (m_rectInt_abs.m_h) / rootPanel->m_rectInt_abs.m_h*2.f;
	return rectFloat;
}

RectFloat Panel::RectFloatRelToRectFloatAbs()
{
	RectFloat rectFloat;
	if (m_parent != NULL)
	{
		rectFloat = m_parent->m_rectFloat_abs*m_rectFloat_rel;
	}
	else
	{
		rectFloat = m_rectFloat_rel;
	}
	return rectFloat;
}

void Panel::Draw()
{
	glColor3f(m_backgroundColor.r, m_backgroundColor.g, m_backgroundColor.b);
	glBegin(GL_QUADS);
		glVertex2f(m_rectFloat_abs.m_x, m_rectFloat_abs.m_y+m_rectFloat_abs.m_h);
		glVertex2f(m_rectFloat_abs.m_x, m_rectFloat_abs.m_y);
		glVertex2f(m_rectFloat_abs.m_x+m_rectFloat_abs.m_w, m_rectFloat_abs.m_y);
		glVertex2f(m_rectFloat_abs.m_x+m_rectFloat_abs.m_w, m_rectFloat_abs.m_y+m_rectFloat_abs.m_h);
	glEnd();
}

void Panel::CreateButton(RectFloat rectFloat, SizeDefinitionMethod sizeDefinition, std::string name)
{
	Button* button = new Button(rectFloat, sizeDefinition, name, this);
	m_buttons.push_back(button);
}

//void Panel::CreateButton(RectInt rectInt, SizeDefinitionMethod sizeDefinition, std::string name)
//{
//	Button* button = new Button(rectInt, sizeDefinition, name, this);
//	m_buttons.push_back(button);
//}

