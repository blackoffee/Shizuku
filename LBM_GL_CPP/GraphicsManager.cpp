#include "GraphicsManager.h"
#include "Mouse.h"
#include "kernel.h"


extern Obstruction* g_obst_d;
extern int g_xDim;

GraphicsManager::GraphicsManager()
{

}

GraphicsManager::GraphicsManager(Panel* panel)
{
	m_parent = panel;
}

void GraphicsManager::GetSimCoordFromMouseCoord(int &xOut, int &yOut, Mouse mouse)
{
	float xf = intCoordToFloatCoord(mouse.m_x, mouse.m_winW);
	float yf = intCoordToFloatCoord(mouse.m_y, mouse.m_winH);
	RectFloat coordsInRelFloat = RectFloat(xf, yf, 1.f, 1.f) / m_parent->m_rectFloat_abs;
	float graphicsToSimDomainScalingFactor = static_cast<float>(g_xDim) / m_parent->m_rectInt_abs.m_w;
	xOut = floatCoordToIntCoord(coordsInRelFloat.m_x, m_parent->m_rectInt_abs.m_w)*graphicsToSimDomainScalingFactor;
	yOut = floatCoordToIntCoord(coordsInRelFloat.m_y, m_parent->m_rectInt_abs.m_h)*graphicsToSimDomainScalingFactor;
}

void GraphicsManager::Click(Mouse mouse)
{
	if (mouse.m_rmb == 1)
	{
		int xi, yi;
		GetSimCoordFromMouseCoord(xi, yi, mouse);
		Obstruction obst = { Obstruction::SQUARE, xi, yi, 10, 0 };
		int obstId = FindUnusedObstructionId();
		m_obstructions[obstId] = obst;
		UpdateDeviceObstructions(g_obst_d, obstId, obst);
	}
	else if (mouse.m_mmb == 1)
	{
		int obstId = FindClosestObstructionId(mouse);
		if (obstId < 0) return;
		Obstruction obst = { Obstruction::SQUARE, -100, -100, 0, 0 };
		m_obstructions[obstId] = obst;
		UpdateDeviceObstructions(g_obst_d, obstId, obst);
	}
}

void GraphicsManager::AddObstruction(Mouse mouse)
{

}


void GraphicsManager::RemoveObstruction(Mouse mouse)
{

}


void GraphicsManager::MoveObstruction(Mouse mouse)
{

}

int GraphicsManager::FindUnusedObstructionId()
{
	for (int i = 0; i < MAXOBSTS; i++)
	{
		if (m_obstructions[i].y < 0)
		{
			return i;
		}
	}
	MessageBox(0, "Object could not be added. You are currently using the maximum number of objects.", "Error", MB_OK);
	return 0;
}

int GraphicsManager::FindClosestObstructionId(Mouse mouse)
{
	int xi, yi;
	GetSimCoordFromMouseCoord(xi, yi, mouse);
	float dist = 999999999999.f;
	int closestObstId = -1;
	for (int i = 0; i < MAXOBSTS; i++)
	{
		if (m_obstructions[i].y >= 0)
		{
			float newDist = GetDistanceBetweenTwoPoints(xi, yi, m_obstructions[i].x, m_obstructions[i].y);
			if (newDist < dist)
			{
				dist = newDist;
				closestObstId = i;
			}
		}
	}
	return closestObstId;
}

float GetDistanceBetweenTwoPoints(float x1, float y1, float x2, float y2)
{
	float dx = x2 - x1;
	float dy = y2 - y1;
	return sqrt(dx*dx + dy*dy);
}