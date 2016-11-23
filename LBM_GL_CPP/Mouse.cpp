#include <GL/freeglut.h>
#include "Mouse.h"

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