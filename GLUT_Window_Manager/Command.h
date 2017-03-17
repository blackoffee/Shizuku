#pragma once
#include "Panel.h"

class Command
{
protected:
    Panel* m_rootPanel;
public:
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


class Zoom : public Command
{
public:
    Zoom(Panel &rootPanel);
    void Start(const int dir, const float mag);
};


class Pan : public Command
{
    float m_initialX;
    float m_initialY;
public:
    Pan(Panel &rootPanel);
    void Start(const float initialX, const float initialY);
    void Track(const float currentX, const float currentY);
    void End();
};


class Rotate : public Command
{
    float m_initialX;
    float m_initialY;
public:
    Rotate(Panel &rootPanel);
    void Start(const float initialX, const float initialY);
    void Track(const float currentX, const float currentY);
    void End();
};


class ButtonPress : public Command
{
    Button* m_button;
public:
    ButtonPress(Panel &rootPanel);
    void Start(Button* button);
    void End(Button* button);
};

class SliderDrag : public Command
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

class AddObstruction : public Command
{
public:
    AddObstruction(Panel &rootPanel);
    void Start(const float currentX, const float currentY);
};

class RemoveObstruction : public Command
{
    int m_currentObst;
public:
    RemoveObstruction(Panel &rootPanel);
    void Start(const float currentX, const float currentY);
    void End(const float currentX, const float currentY);
};


class MoveObstruction : public Command
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