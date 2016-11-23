#include "Frame.h"
#include <iostream>

extern Window theWindow;

Frame::Frame(int x, int y, int w, int h, std::string title)
{
	m_rect_i = { x, y, w, h };
	m_title = title;
}


Frame::Frame(RectInt rect, std::string title)
{
	m_rect_i = rect;
	m_title = title;
	m_rect_f = RectInt2RectFloat(rect,theWindow.GetWindowRectangle());
}

RectInt Frame::GetRectInt()
{
	return m_rect_i;
}

RectFloat Frame::GetRectFloat()
{
	return m_rect_f;
}

std::string Frame::GetTitle()
{
	return m_title;
}

void Frame::SetSize(RectInt rect)
{
	m_rect_i = rect;
}

void Frame::SetTitle(std::string title)
{
	m_title = title;
}

void Frame::Draw2D()
{
	glBegin(GL_QUADS);
		glVertex2f(m_rect_f.m_x         ,m_rect_f.m_y+m_rect_f.m_h);
		glVertex2f(m_rect_f.m_x         ,m_rect_f.m_y         );
		glVertex2f(m_rect_f.m_x+m_rect_f.m_w,m_rect_f.m_y         );
		glVertex2f(m_rect_f.m_x+m_rect_f.m_w,m_rect_f.m_y+m_rect_f.m_h);
	glEnd();

	std::vector<Button*>::iterator it;
	for (it = m_buttons.begin(); it < m_buttons.end(); it++)
	{
		(*it)->Draw2D();
	}


}

void Frame::CreateButton(RectFloat rect, std::string title)
{
	//Button button(this, rect,title,NULL);
	Button* button = new Button(rect, title);
	m_buttons.push_back(button);
	std::cout << m_buttons.size();
}

Button Frame::GetButtonByID(int id)
{
	if (m_buttons.size() > 0)
	{
		return *m_buttons[id];
	}
}

