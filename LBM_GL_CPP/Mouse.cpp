#include "Mouse.h"

void Mouse::SetBasePanel(Panel* basePanel)
{
	m_basePanel = basePanel;
	m_winW = m_basePanel->m_rectInt_abs.m_w;
	m_winH = m_basePanel->m_rectInt_abs.m_h;
	m_x = 0;
	m_y = 0;
}

void Mouse::Update(int x, int y, int button, int state)
{
	m_x = x;
	m_y = y;
	m_lmbState = (button == GLUT_LEFT_BUTTON) ? state : m_lmbState;
	m_rmbState = (button == GLUT_RIGHT_BUTTON) ? state : m_rmbState;
	m_mmbState = (button == GLUT_MIDDLE_BUTTON) ? state : m_mmbState;
}
void Mouse::Update(int x, int y)
{
	m_x = x;
	m_y = y;
}
void Mouse::GetChange(int x, int y)
{
	m_xprev = x;
	m_yprev = y;
}
int Mouse::GetX()
{
	return m_x;
}

int Mouse::GetY()
{
	return m_y;
}

void Mouse::Move(int x, int y)
{
	float dx = intCoordToFloatCoord(x, m_winW) - intCoordToFloatCoord(m_xprev, m_winW);
	float dy = intCoordToFloatCoord(y, m_winH) - intCoordToFloatCoord(m_yprev, m_winH);
	m_xprev = x;
	m_yprev = y;

	if (m_lmb == GLUT_DOWN)
	{
		m_currentlySelectedPanel->Drag(dx, dy);
	}

}


void Mouse::Click(int x, int y, int button, int state)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		m_xprev = x;
		m_yprev = y;
		m_currentlySelectedPanel = GetPanelThatPointIsIn(m_basePanel, intCoordToFloatCoord(x, m_winW), intCoordToFloatCoord(y, m_winH));
		m_currentlySelectedPanel->Click();
	}
}

void Mouse::LeftClickDown(int x, int y)
{

}




float intCoordToFloatCoord(int x, int xDim)
{
	return (static_cast<float> (x) / xDim)*2.f - 1.f;
}
