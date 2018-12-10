#include "PillarDefinition.h"

using namespace Shizuku::Flow;

PillarDefinition::PillarDefinition()
{
}

PillarDefinition::PillarDefinition(const Types::Point<float>& p_pos, const Types::Box<float>& p_size)
    :m_position(p_pos), m_size(p_size)
{
}

Types::Point<float>& PillarDefinition::Pos()
{
    return m_position;
}

Types::Box<float>& PillarDefinition::Size()
{
    return m_size;
}

void PillarDefinition::SetPosition(const Types::Point<float>& p_pos)
{
    m_position = p_pos;
}

void PillarDefinition::SetSize(const Types::Box<float>& p_size)
{
    m_size = p_size;
}