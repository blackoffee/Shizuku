#pragma once

class RectInt
{
public:
	int m_x, m_y, m_w, m_h;
	RectInt();
	RectInt(int x, int y, int w, int h) : m_x(x), m_y(y), m_w(w), m_h(h)
	{
	}
	friend bool operator==(const RectInt rec1, const RectInt rec2);
};