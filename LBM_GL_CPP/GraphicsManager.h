#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <string>
#include <iostream>
#include <vector>
#include "Common.h"
#include "kernel.h"

class Panel;
class Mouse;

class GraphicsManager
{
public:
	int m_currentObstId = -1;
	float m_currentZ = -1000.f;
	Obstruction* m_obstructions;
	Panel* m_parent;

	GraphicsManager();
	GraphicsManager(Panel* panel);

	void GetSimCoordFromMouseCoord(int &xOut, int &yOut, Mouse mouse);
	void GetSimCoordFromFloatCoord(int &xOut, int &yOut, float xf, float yf);
	void GetMouseRay(float3 &rayOrigin, float3 &rayDir, int mouseX, int mouseY);
	int GetSimCoordFrom3DMouseClickOnObstruction(int &xOut, int &yOut, Mouse mouse);
	void GetSimCoordFrom2DMouseRay(int &xOut, int &yOut, Mouse mouse);
	void GetSimCoordFrom2DMouseRay(int &xOut, int &yOut, int mouseX, int mouseY);
	void Click(Mouse mouse);
	void Drag(int x, int y, float dx, float dy);
	void AddObstruction(Mouse mouse);
	void AddObstruction(int simX, int simY);
	void RemoveObstruction(Mouse mouse);
	void MoveObstruction(int x, int y, float dx, float dy);
	int FindUnusedObstructionId();
	int FindClosestObstructionId(Mouse mouse);
	int FindClosestObstructionId(int simX, int simY);
	int FindObstructionPointIsInside(int x, int y, float tolerance=0.f);
	bool IsInClosestObstruction(Mouse mouse);

};

float GetDistanceBetweenTwoPoints(float x1, float y1, float x2, float y2);