#include "Stopwatch.h"
#include "StopwatchImpl.h"

using namespace Shizuku::Core;

Stopwatch::Stopwatch()
{
    m_impl = new StopwatchImpl();
}

Stopwatch::Stopwatch(const int p_recCount)
{
    m_impl = new StopwatchImpl(p_recCount);
}

void Stopwatch::Tick()
{
    m_impl->Tick();
}

double Stopwatch::Tock()
{
    return m_impl->Tock();
}

void Stopwatch::Reset()
{
    m_impl->Reset();
}

double Stopwatch::GetAverage()
{
    return m_impl->GetAverage();
}