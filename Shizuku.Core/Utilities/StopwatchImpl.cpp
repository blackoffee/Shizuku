#include "StopwatchImpl.h"

using namespace Shizuku::Core;
using namespace std::chrono;

StopwatchImpl::StopwatchImpl()
{
    m_total = 0;
    m_recCount = 1;
    m_recs = std::queue<double>();
}

StopwatchImpl::StopwatchImpl(const int p_recCount)
{
    m_total = 0;
    m_recCount = p_recCount;
    m_recs = std::queue<double>();
}

void StopwatchImpl::Tick()
{
    m_before = high_resolution_clock::now();
}

double StopwatchImpl::Tock()
{
    const double time = duration_cast<duration<double>>(high_resolution_clock::now() - m_before).count();
    m_total += time;

    m_recs.push(time);
    while (m_recs.size() > m_recCount)
    {
        m_total -= m_recs.front();
        m_recs.pop();
    }

    return time;
}

void StopwatchImpl::Reset()
{
    m_total = 0;
    while (m_recs.size() > 0)
        m_recs.pop();
}

double StopwatchImpl::GetAverage()
{
    return m_total / m_recs.size();
}