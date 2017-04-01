#pragma once
#include "Command.h"

class FW_API ButtonPress : public Command
{
    Button* m_button;
public:
    ButtonPress(Panel &rootPanel);
    void Start(Button* button);
    void End(Button* button);
};
