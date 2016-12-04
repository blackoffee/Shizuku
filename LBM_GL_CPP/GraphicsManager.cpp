#include "GraphicsManager.h"
#include "Mouse.h"
#include "kernel.h"

extern Obstruction* g_obst_d;

GraphicsManager::GraphicsManager()
{

}

GraphicsManager::GraphicsManager(Panel* panel)
{
	m_parent = panel;
}

void GraphicsManager::Click(Mouse mouse)
{
	if (mouse.m_lmb == 1)
	{
		Obstruction obst = { Obstruction::SQUARE, mouse.m_x, mouse.m_y, 10, 0 };
		UpdateDeviceObstructions(g_obst_d, 1, obst);
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

Obstruction* GraphicsManager::FindUnusedObstruction()
{
	return NULL;
}

