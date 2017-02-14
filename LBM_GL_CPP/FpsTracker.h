#pragma once
#include <time.h>

class FpsTracker
{
private:
    clock_t m_before, m_diff;
    int m_frameCount;
    int m_frameLimit;
    float m_fps;
public:
    FpsTracker();
    void Tick();
    void Tock();
    float GetFps();
};

