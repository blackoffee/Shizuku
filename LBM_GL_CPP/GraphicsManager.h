#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <string>
#include <iostream>
#include <vector>
#include "Common.h"

class Panel;
class Mouse;

class GraphicsManager
{
public:
	Obstruction* m_currentObst;
	Obstruction* m_obstructions;
	Panel* m_parent;

	GraphicsManager();
	GraphicsManager(Panel* panel);

	void Click(Mouse mouse);
	void AddObstruction(Mouse mouse);
	void RemoveObstruction(Mouse mouse);
	void MoveObstruction(Mouse mouse);
	Obstruction* FindUnusedObstruction();

};