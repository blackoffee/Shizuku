#ifndef MOUSE
#define MOUSE

#include "Panel.h"

class Mouse
{
public:
	int m_x, m_y;
	int m_lmb, m_mmb, m_rmb;
	int m_xprev, m_yprev;
	int m_winW, m_winH;
	Panel* m_basePanel = NULL;
	Panel* m_currentlySelectedPanel = NULL;
	Mouse() :m_x(0), m_y(0), m_lmb(0), m_mmb(0), m_rmb(0)
	{
	}
	void Update(int x, int y, int button, int state);
	void Update(int x, int y);
	void GetChange(int x, int y);
	int GetX();
	int GetY();


	void SetBasePanel(Panel* basePanel);

	void Move(int x, int y); //store dx and dy
	void Click(int x, int y, int button, int state);

	void LeftClickUp(int x, int y);
	void LeftClickDown(int x, int y);


};

float intCoordToFloatCoord(int x, int xDim);

#endif
