#pragma once 

#include <string>
#include <iostream>
#include <vector>
#include <GL/glew.h>
#include <GL/freeglut.h>
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


class Panel
{
public:
	enum SizeDefinitionMethod {DEF_ABS, DEF_REL};
	std::string m_name;
	std::vector<Panel*> m_subPanels;
	std::vector<Panel*> m_buttons;
	Panel* m_parent = NULL; //pointer to parent frame
	RectInt m_rectInt_abs; //absolute coordinates in Window
	RectInt m_rectInt_rel; //relative coordinates wrt to parent
	RectFloat m_rectFloat_abs; //absolute coordinates in window. this is the one used for drawing, so always want to keep this up-to-date.
	RectFloat m_rectFloat_rel;
	Color m_backgroundColor;

	Panel();
	Panel(RectInt rectInt  , SizeDefinitionMethod sizeDefinition, std::string name, Color color, Panel* parent = NULL);
	Panel(RectFloat rectFloat, SizeDefinitionMethod sizeDefinition, std::string name, Color color, Panel* parent = NULL);

	void CreateSubPanel(RectFloat rectFloat, SizeDefinitionMethod sizeDefinition, std::string name, Color color);
	void CreateSubPanel(RectInt rectInt    , SizeDefinitionMethod sizeDefinition, std::string name, Color color);

	void CreateButton(RectFloat rectFloat, SizeDefinitionMethod sizeDefinition, std::string name, Color color);
	void CreateButton(RectInt rectInt    , SizeDefinitionMethod sizeDefinition, std::string name, Color color);

	RectFloat RectIntAbsToRectFloatAbs();
	RectFloat RectFloatRelToRectFloatAbs();

	void Draw();
};

class Button : public Panel
{
public:
	std::string m_secondName;
	

	Button(RectFloat rectFloat, SizeDefinitionMethod sizeDefinition, std::string name, Color color, Panel* parent = NULL);
	Button(RectInt rectInt    , SizeDefinitionMethod sizeDefinition, std::string name, Color color, Panel* parent = NULL);


	//	Button(std::string name1, std::string name2)
//	{
//		m_name = name1;
//		m_secondName = name2;
//	}
//	Button(std::string name1)
//	{
//		m_name = name1;
//	}
};