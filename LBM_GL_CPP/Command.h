#pragma once

#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class Panel;
class Button;
class SliderBar;
class GraphicsManager;

class FW_API Command
{
protected:
    Panel* m_rootPanel;
    enum State {ACTIVE, INACTIVE};
    State m_state;
    Command();
    Command(Panel &rootPanel);
    void Start();
    void Track();
    void End();
    Panel* GetRootPanel();
    GraphicsManager* GetGraphicsManager();
};


class FW_API Zoom : public Command
{
public:
    Zoom(Panel &rootPanel);
    void Start(const int dir, const float mag);
};


class FW_API Pan : public Command
{
    float m_initialX;
    float m_initialY;
public:
    Pan(Panel &rootPanel);
    void Start(const float initialX, const float initialY);
    void Track(const float currentX, const float currentY);
    void End();
};


class FW_API Rotate : public Command
{
    float m_initialX;
    float m_initialY;
public:
    Rotate(Panel &rootPanel);
    void Start(const float initialX, const float initialY);
    void Track(const float currentX, const float currentY);
    void End();
};


class FW_API ButtonPress : public Command
{
    Button* m_button;
public:
    ButtonPress(Panel &rootPanel);
    void Start(Button* button);
    void End(Button* button);
};

class FW_API SliderDrag : public Command
{
    float m_initialX;
    float m_initialY;
    SliderBar* m_sliderBar;
public:
    SliderDrag(Panel &rootPanel);
    void Start(SliderBar* sliderBar, const float currentX, const float currentY);
    void Track(const float currentX, const float currentY);
    void End();
};

class FW_API AddObstruction : public Command
{
public:
    AddObstruction(Panel &rootPanel);
    void Start(const float currentX, const float currentY);
};

class FW_API RemoveObstruction : public Command
{
    int m_currentObst;
public:
    RemoveObstruction(Panel &rootPanel);
    void Start(const float currentX, const float currentY);
    void End(const float currentX, const float currentY);
};


class FW_API MoveObstruction : public Command
{
    int m_currentObst;
    float m_initialX;
    float m_initialY;
public:
    MoveObstruction(Panel &rootPanel);
    void Start(const float currentX, const float currentY);
    void Track(const float currentX, const float currentY);
    void End();
};