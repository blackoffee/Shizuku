#include "Mouse.h"

extern float rotate_x;
extern float rotate_z;
extern float translate_x;
extern float translate_y;
extern int g_TwoDView;

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
    int mouseState = (state == GLUT_DOWN) ? 1 : 0;
    m_lmb = (button == GLUT_LEFT_BUTTON) ? mouseState : m_lmb;
    m_rmb = (button == GLUT_RIGHT_BUTTON) ? mouseState : m_rmb;
    m_mmb = (button == GLUT_MIDDLE_BUTTON) ? mouseState : m_mmb;
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
    float dx = intCoordToFloatCoord(x, m_winW) - intCoordToFloatCoord(m_x, m_winW);
    float dy = intCoordToFloatCoord(y, m_winH) - intCoordToFloatCoord(m_y, m_winH);
    Update(x, y);

    if (m_currentlySelectedPanel != NULL)
    {
        int mod = glutGetModifiers();
        if (m_lmb == 1)
        {
            m_currentlySelectedPanel->Drag(x,y,dx, dy);
        }
        else if (m_mmb == 1 && mod == GLUT_ACTIVE_CTRL)
        {
            translate_x += dx;
            translate_y += dy;
        }
        else if (m_mmb == 1)
        {
            rotate_x += dy*45.f;
            rotate_z += dx*45.f;
        }
    }
    else
    {
        return;
    }
}

void Mouse::Click(int x, int y, int button, int state)
{
    Update(x, y, button, state);
    m_currentlySelectedPanel = GetPanelThatPointIsIn(m_basePanel, intCoordToFloatCoord(x, m_winW), intCoordToFloatCoord(y, m_winH));
    if (m_currentlySelectedPanel != NULL && state == GLUT_DOWN)
    {
        m_currentlySelectedPanel->ClickDown(*this);
    }
}

void Mouse::LeftClickDown(int x, int y)
{

}




float intCoordToFloatCoord(int x, int xDim)
{
    return (static_cast<float> (x) / xDim)*2.f - 1.f;
}

int floatCoordToIntCoord(float x, int xDim)
{
    return static_cast<int> ((x+1.f)/2.f*xDim);
}
