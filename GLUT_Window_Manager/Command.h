#pragma once
#include "main.h"

class Command
{
    Panel* m_rootPanel;
public:
    enum State {ACTIVE, UNACTIVE};
    State m_state;
    Command();
    void Start();
    void Track();
    void End();
    void Initialize(Panel &rootPanel);
    Panel* GetRootPanel();
    GraphicsManager* GetGraphicsManager();
};


class Zoom : public Command
{
public:
    Zoom();
    void Start(const int dir, const float mag);
};


class Pan : public Command
{
    float m_initialX;
    float m_initialY;
public:
    Pan();
    void Start(const float initialX, const float initialY);
    void Track(const float currentX, const float currentY);
    void End();
};


class Rotate : public Command
{
    float m_initialX;
    float m_initialY;
public:
    Rotate();
    void Start(const float initialX, const float initialY);
    void Track(const float currentX, const float currentY);
    void End();
};


class ButtonPress : public Command
{
    Button* m_button;
public:
    ButtonPress();
    void Start(Button* button);
    void End(Button* button);
};

