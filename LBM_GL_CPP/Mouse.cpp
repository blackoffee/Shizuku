#include "Mouse.h"

void Mouse::SetBasePanel(Panel* basePanel)
{
    m_basePanel = basePanel;
    m_winW = m_basePanel->GetRectIntAbs().m_w;
    m_winH = m_basePanel->GetRectIntAbs().m_h;
    m_x = 0;
    m_y = 0;
}

void Mouse::Update(int x, int y, int button, int state)
{
    m_x = x;
    m_y = y;
    m_button = button;
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
        m_currentlySelectedPanel->Drag(x,y,dx, dy, m_button);
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

void Mouse::Wheel(int button, int dir, int x, int y)
{
    if (m_currentlySelectedPanel != NULL)
    {
        m_currentlySelectedPanel->Wheel(m_button,dir,x,y);
    }
    else
    {
        return;
    }
}
