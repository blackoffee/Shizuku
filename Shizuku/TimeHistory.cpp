#include "TimeHistory.h"

using namespace Shizuku::Core;
using namespace Shizuku::Presentation;

TimeHistory::TimeHistory() : m_size(20)
{
    m_queue = std::deque<double>();
}

TimeHistory::TimeHistory(const int p_size) : m_size(p_size)
{
    m_queue = std::deque<double>();
}

void TimeHistory::Append(const double p_value)
{
    m_queue.push_back(p_value);
    m_minMax.Min = std::min(m_minMax.Min, p_value);
    m_minMax.Max = std::max(m_minMax.Max, p_value);
    if (m_queue.size() > m_size)
        m_queue.pop_front();
}

float TimeHistory::DataProvider(void* p_data, int p_index)
{
    return static_cast<float>(m_queue[p_index]);
}

int TimeHistory::Size()
{
    return m_queue.size();
}

Types::MinMax<double> TimeHistory::MinMax()
{
    return m_minMax;
}
