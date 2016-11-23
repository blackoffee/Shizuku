#include "Window.h"


RectFloat RectInt2RectFloat(RectInt rectChild, RectInt rectParent)
{
	float xf = (static_cast< float >(rectChild.m_x) / rectParent.m_w)*2.f - 1.f;
	float yf = (static_cast< float >(rectChild.m_y) / rectParent.m_h)*2.f - 1.f;
	float wf = (static_cast< float >(rectChild.m_w) / rectParent.m_w)*2.f;
	float hf = (static_cast< float >(rectChild.m_h) / rectParent.m_h)*2.f;
	return RectFloat(xf, yf, wf, hf);
}


Window::Window(int x, int y)
{
	m_rect_i = { 100, 100, x, y };
}

void Window::SetTitle(std::string title)
{
	m_title = title;
}
int Window::GetWidth()
{
	return m_width;
}
int Window::GetHeight()
{
	return m_height;
}
RectInt Window::GetWindowRectangle()
{
	return m_rect_i;
}
std::string Window::GetTitle()
{
	return m_title;
}
void Window::CreateFrame(RectInt rect, std::string title)
{
	Frame* frame = new Frame(rect, title);
	m_frames.push_back(frame);
}

//Frame Window::GetFrameByID(int id)
//{
//	if (m_frames.size() > 0)
//	{
//		return *m_frames[id];
//	}
//}