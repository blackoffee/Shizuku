#include "Command.h"

using namespace Shizuku::Flow::Command;

Command::Command(GraphicsManager &graphicsManager)
{
    m_graphics = &graphicsManager;
}

GraphicsManager* Command::GetGraphicsManager()
{
    return m_graphics;
}

