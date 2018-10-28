#include "Command.h"

Command::Command(GraphicsManager &graphicsManager)
{
    m_graphics = &graphicsManager;
}

GraphicsManager* Command::GetGraphicsManager()
{
    return m_graphics;
}

