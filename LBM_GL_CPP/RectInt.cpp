#include "RectInt.h"

RectInt::RectInt() : m_x(0), m_y(0), m_w(0), m_h(0)
{
}

bool operator==(const RectInt rec1, const RectInt rec2)
{
	bool result(true);
	result *= rec1.m_x == rec2.m_x;
	result *= rec1.m_y == rec2.m_y;
	result *= rec1.m_w == rec2.m_w;
	result *= rec1.m_h == rec2.m_h;
	return result;
}
