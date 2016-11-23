#ifndef MOUSE
#define MOUSE

class Mouse
{
	int m_x, m_y;
	int m_lmb, m_mmb, m_rmb;
	int m_lmbState, m_mmbState, m_rmbState;
	int m_xprev, m_yprev;
public:
	Mouse() :m_x(0), m_y(0), m_lmb(0), m_mmb(0), m_rmb(0)
	{
	}
	void Update(int x, int y, int button, int state);
	void Update(int x, int y);
	void GetChange(int x, int y);
	int GetX();
	int GetY();


};

#endif
