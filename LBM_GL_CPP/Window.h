#pragma once

#include "RectInt.h"
#include "RectFloat.h"
#include "Frame.h"
#include <vector>

class Frame;

RectFloat RectInt2RectFloat(RectInt rectChild, RectInt rectParent);

class Window
{
	int m_width, m_height;
	RectInt m_rect_i; //Postion and size with respect to screen
	RectFloat m_rect_f; //Position and size with respect to screen, in NDC
	std::vector<Frame> m_frames;
	std::string m_title;
public:
	Window(int x, int y);
	void SetTitle(std::string title);
	int GetWidth();
	int GetHeight();
	RectInt GetWindowRectangle();
	std::string GetTitle();
	void CreateFrame(RectInt rect,std::string);
	Frame GetFrameByID(int id);
		
};
