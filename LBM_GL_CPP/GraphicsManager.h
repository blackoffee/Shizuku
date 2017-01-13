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
	int m_currentObstId = -1;
	Obstruction* m_obstructions;
	Panel* m_parent;

	GraphicsManager();
	GraphicsManager(Panel* panel);

	void GetSimCoordFromMouseCoord(int &xOut, int &yOut, Mouse mouse);
	void GetSimCoordFromFloatCoord(int &xOut, int &yOut, float xf, float yf);
	void Click(Mouse mouse);
	void Drag(float dx, float dy);
	void AddObstruction(Mouse mouse);
	void RemoveObstruction(Mouse mouse);
	void MoveObstruction(float dx, float dy);
	int FindUnusedObstructionId();
	int FindClosestObstructionId(Mouse mouse);
	bool IsInClosestObstruction(Mouse mouse);

};

float GetDistanceBetweenTwoPoints(float x1, float y1, float x2, float y2);