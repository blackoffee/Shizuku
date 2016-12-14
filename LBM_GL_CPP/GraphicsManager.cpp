#include "GraphicsManager.h"
#include "Mouse.h"
#include "kernel.h"


extern Obstruction* g_obst_d;
extern int g_xDim, g_yDim, g_xDimVisible, g_yDimVisible;
extern Obstruction::Shape g_currentShape;
extern float g_currentSize;
extern float g_initialScaleUp;
extern int g_TwoDView;

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
	//float graphicsToSimDomainScalingFactor = static_cast<float>(g_xDim) / m_parent->m_rectInt_abs.m_w;
	//float graphicsToSimDomainScalingFactor = 1.f/g_initialScaleUp;// static_cast<float>(g_xDim) / m_parent->m_rectInt_abs.m_w;
	float graphicsToSimDomainScalingFactorX = static_cast<float>(g_xDimVisible) / min(m_parent->m_rectInt_abs.m_w, MAX_XDIM*g_initialScaleUp);
	float graphicsToSimDomainScalingFactorY = static_cast<float>(g_yDimVisible) / min(m_parent->m_rectInt_abs.m_h, MAX_YDIM*g_initialScaleUp);
	xOut = floatCoordToIntCoord(coordsInRelFloat.m_x, m_parent->m_rectInt_abs.m_w)*graphicsToSimDomainScalingFactorX;
	yOut = floatCoordToIntCoord(coordsInRelFloat.m_y, m_parent->m_rectInt_abs.m_h)*graphicsToSimDomainScalingFactorY;
}

void GraphicsManager::GetSimCoordFromFloatCoord(int &xOut, int &yOut, float xf, float yf)
{
	RectFloat coordsInRelFloat = RectFloat(xf, yf, 1.f, 1.f) / m_parent->m_rectFloat_abs;
	//float graphicsToSimDomainScalingFactor = static_cast<float>(g_xDim) / m_parent->m_rectInt_abs.m_w;
	//float graphicsToSimDomainScalingFactor = 1.f/g_initialScaleUp;// static_cast<float>(g_xDim) / m_parent->m_rectInt_abs.m_w;
	float graphicsToSimDomainScalingFactorX = static_cast<float>(g_xDimVisible) / min(m_parent->m_rectInt_abs.m_w, MAX_XDIM*g_initialScaleUp);
	float graphicsToSimDomainScalingFactorY = static_cast<float>(g_yDimVisible) / min(m_parent->m_rectInt_abs.m_h, MAX_YDIM*g_initialScaleUp);
	xOut = floatCoordToIntCoord(coordsInRelFloat.m_x, m_parent->m_rectInt_abs.m_w)*graphicsToSimDomainScalingFactorX;
	yOut = floatCoordToIntCoord(coordsInRelFloat.m_y, m_parent->m_rectInt_abs.m_h)*graphicsToSimDomainScalingFactorY;
}

void GraphicsManager::Click(Mouse mouse)
{
	if (g_TwoDView == 1)
	{
		if (mouse.m_rmb == 1)
		{
			AddObstruction(mouse);
		}
		else if (mouse.m_mmb == 1)
		{
			if (IsInClosestObstruction(mouse))
			{
				RemoveObstruction(mouse);
			}
		}
		else if (mouse.m_lmb == 1)
		{
			if (IsInClosestObstruction(mouse))
			{
				m_currentObstId = FindClosestObstructionId(mouse);
			}
			else
			{
				m_currentObstId = -1;
			}
		}
	}
}

void GraphicsManager::Drag(float dx, float dy)
{
	MoveObstruction(dx, dy);
}

void GraphicsManager::AddObstruction(Mouse mouse)
{
	int xi, yi;
	GetSimCoordFromMouseCoord(xi, yi, mouse);
	Obstruction obst = { g_currentShape, xi, yi, g_currentSize, 0 };
	int obstId = FindUnusedObstructionId();
	m_obstructions[obstId] = obst;
	UpdateDeviceObstructions(g_obst_d, obstId, obst);
}


void GraphicsManager::RemoveObstruction(Mouse mouse)
{
	int obstId = FindClosestObstructionId(mouse);
	if (obstId < 0) return;
	Obstruction obst = { g_currentShape, -100, -100, 0, 0 };
	m_obstructions[obstId] = obst;
	UpdateDeviceObstructions(g_obst_d, obstId, obst);
}


void GraphicsManager::MoveObstruction(float dx, float dy)
{
	if (m_currentObstId > -1)
	{
		Obstruction obst = m_obstructions[m_currentObstId];
		float dxf, dyf;
		int windowWidth = m_parent->GetRootPanel()->m_rectInt_abs.m_w;
		int windowHeight = m_parent->GetRootPanel()->m_rectInt_abs.m_h;
		dxf = dx*static_cast<float>(g_xDimVisible) / min(m_parent->m_rectFloat_abs.m_w, g_xDimVisible*g_initialScaleUp/windowWidth*2.f);
		dyf = dy*static_cast<float>(g_yDimVisible) / min(m_parent->m_rectFloat_abs.m_h, g_yDimVisible*g_initialScaleUp/windowHeight*2.f);
		obst.x += dxf;
		obst.y += dyf;
		m_obstructions[m_currentObstId] = obst;
		UpdateDeviceObstructions(g_obst_d, m_currentObstId, obst);
	}
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

bool GraphicsManager::IsInClosestObstruction(Mouse mouse)
{
	int closestObstId = FindClosestObstructionId(mouse);
	int xi, yi;
	GetSimCoordFromMouseCoord(xi, yi, mouse);
	return (GetDistanceBetweenTwoPoints(xi,yi,m_obstructions[closestObstId].x, m_obstructions[closestObstId].y) < m_obstructions[closestObstId].r1);
}

float GetDistanceBetweenTwoPoints(float x1, float y1, float x2, float y2)
{
	float dx = x2 - x1;
	float dy = y2 - y1;
	return sqrt(dx*dx + dy*dy);
}