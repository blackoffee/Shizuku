#include "FpsTracker.h"

using namespace Shizuku::Core;

FpsTracker::FpsTracker()
{
    m_frameCount = 0;
    m_frameLimit = 20;
    m_fps = 0;
}

void FpsTracker::Tick()
{
    if (m_frameCount == 0)
    {
        m_before = clock();
    }
    m_frameCount++;
}

void FpsTracker::Tock()
{
    if (m_frameCount == m_frameLimit)
    {
        m_diff = clock() - m_before;
        m_fps = static_cast<float>(m_frameLimit) / (static_cast<float>(m_diff) / CLOCKS_PER_SEC);
        m_frameCount = 0;
    }
}

float FpsTracker::GetFps()
{
    return m_fps;
}
