#pragma once

#include <vector>
#include "RectInt.h"
#include "RectFloat.h"

class Frame;

//typedef void(*ButtonCallback)();
typedef int ButtonCallback;

class Button
{
	RectInt m_rect_i; //Postion and size with respect to Frame
	RectFloat m_rect_f; //Position and size with respect to Frame, in NDC
	std::string m_title = "no title";
	int state;
	int highlighted;
	ButtonCallback m_callback;
	
public:
	Frame* m_parent;
	Button();
	Button(Frame* parent, int x, int y, int w, int h, std::string title = "no title");
	Button(Frame* parent, RectInt rect, std::string title="no title", ButtonCallback = NULL);
	Button(Frame* parent, RectFloat rect, std::string title="no title", ButtonCallback = NULL);
	Button(RectFloat rect, std::string title="no title");
	RectInt GetButtonRectangle();
	std::string GetTitle();
	void SetSize(RectInt rect);
	void SetTitle(std::string title);
	void Draw2D();

};

