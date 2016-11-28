#pragma once
#include "RectInt.h"

class RectFloat
{
public:
	float m_x, m_y, m_w, m_h;
	RectFloat();
	RectFloat(float x, float y, float w, float h) : m_x(x), m_y(y), m_w(w), m_h(h)
	{
	}

	float GetCentroidX();
	float GetCentroidY();

	friend RectFloat operator*(const RectFloat rec1, const RectFloat rec2);
	friend bool operator==(const RectFloat rec1, const RectFloat rec2);
};

RectFloat RectInt2RectFloat(RectInt rectChild, RectInt rectParent);
