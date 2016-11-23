#ifndef FRAME
#define FRAME

#include "RectInt.h"
#include "RectFloat.h"
#include "Window.h"
#include "Button.h"
#include <vector>
#include <GL/glew.h>
#include <GL/freeglut.h>

class Window;
class Button;

class Frame
{
	RectInt m_rect_i; //Postion and size with respect to Window
	RectFloat m_rect_f; //Position and size with respect to Window, in NDC
	std::string m_title = "no title";
public:
	std::vector<Button*> m_buttons;
	Frame(int x, int y, int w, int h, std::string title = "no title");
	Frame(RectInt rect, std::string title="no title");
	RectInt GetRectInt();
	RectFloat GetRectFloat();
	std::string GetTitle();
	void SetSize(RectInt rect);
	void SetTitle(std::string title);
	void Draw2D();
	void CreateButton(RectFloat rect, std::string title);
	Button GetButtonByID(int id);

};


#endif