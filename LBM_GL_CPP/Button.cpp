#include <vector>
#include "RectInt.h"
#include "RectFloat.h"
#include "Window.h"
#include "Frame.h"
#include "Button.h"
#include <GL/glew.h>
#include <GL/freeglut.h>

Button::Button()
{

}

Button::Button(Frame* parent, int x, int y, int w, int h, std::string title)
{
	m_rect_i = { x, y, w, h };
	m_parent = parent;
	m_rect_f = RectInt2RectFloat(m_rect_i,m_parent->GetRectInt());
	m_title = title;
}

Button::Button(Frame* parent, RectInt rect, std::string title, ButtonCallback)
{
	m_rect_i = rect;
	m_parent = parent;
	m_rect_f = RectInt2RectFloat(m_rect_i,m_parent->GetRectInt());
	m_title = title;
}

Button::Button(Frame* parent, RectFloat rect, std::string title, ButtonCallback)
{
	m_parent = parent;
	m_rect_f = rect;
	m_title = title;
}

Button::Button(RectFloat rect, std::string title)
{
	m_rect_f = rect;
	m_title = title;
}

RectInt Button::GetButtonRectangle()
{
	return m_rect_i;
}

std::string Button::GetTitle()
{
	return m_title;
}

void Button::SetSize(RectInt rect){
	m_rect_i = rect;
	m_rect_f = RectInt2RectFloat(m_rect_i,m_parent->GetRectInt());
}

void Button::SetTitle(std::string title)
{
	m_title = title;
}

void Button::Draw2D()
{
	RectFloat absCoordRectFloat(m_parent->GetRectFloat()*m_rect_f);
	//RectFloat absCoordRectFloat(0.05f,0.05f,0.1f,0.2f);
	glColor3f(0.8f, 0.8f, 0.8f);
	glBegin(GL_QUADS);
		glVertex2f(absCoordRectFloat.m_x         ,absCoordRectFloat.m_y+absCoordRectFloat.m_h);
		glVertex2f(absCoordRectFloat.m_x         ,absCoordRectFloat.m_y         );
		glVertex2f(absCoordRectFloat.m_x+absCoordRectFloat.m_w,absCoordRectFloat.m_y         );
		glVertex2f(absCoordRectFloat.m_x+absCoordRectFloat.m_w,absCoordRectFloat.m_y+absCoordRectFloat.m_h);
	glEnd();
}




