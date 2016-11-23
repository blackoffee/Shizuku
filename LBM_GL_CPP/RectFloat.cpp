#include "RectFloat.h"
#include "math.h"

RectFloat::RectFloat() : m_x(0), m_y(0), m_w(0), m_h(0)
{
}

//// chains together nested float coordinates. Use to get absolute float coordinates from relative float coordinates
RectFloat operator*(RectFloat recParent, RectFloat recChild)
{
	RectFloat result;
	result.m_x = recParent.m_x+recParent.m_w*(recChild.m_x+1.f)/2.f;
	result.m_y = recParent.m_y+recParent.m_h*(recChild.m_y+1.f)/2.f;
	result.m_w = (recParent.m_w / 2.f)*(recChild.m_w / 2.f)*2.f;
	result.m_h = (recParent.m_h / 2.f)*(recChild.m_h / 2.f)*2.f;
	return result;
}

bool operator==(RectFloat rec1, RectFloat rec2)
{
	bool result(true);
	result *= fabs(rec1.m_x - rec2.m_x) < 0.001f*(fabs(rec1.m_x) + fabs(rec2.m_x));
	result *= fabs(rec1.m_y - rec2.m_y) < 0.001f*(fabs(rec1.m_y) + fabs(rec2.m_y));
	result *= fabs(rec1.m_w - rec2.m_w) < 0.001f*(fabs(rec1.m_w) + fabs(rec2.m_w));
	result *= fabs(rec1.m_h - rec2.m_h) < 0.001f*(fabs(rec1.m_h) + fabs(rec2.m_h));
	return result;
}

RectFloat RectInt2RectFloat(RectInt rectChild, RectInt rectParent)
{
	float xf = (static_cast< float >(rectChild.m_x) / rectParent.m_w)*2.f - 1.f;
	float yf = (static_cast< float >(rectChild.m_y) / rectParent.m_h)*2.f - 1.f;
	float wf = (static_cast< float >(rectChild.m_w) / rectParent.m_w)*2.f;
	float hf = (static_cast< float >(rectChild.m_h) / rectParent.m_h)*2.f;
	return RectFloat(xf, yf, wf, hf);
}

