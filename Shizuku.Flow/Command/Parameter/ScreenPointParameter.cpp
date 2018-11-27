#include "Command/Parameter/ScreenPointParameter.h"

using namespace Shizuku::Core;
using namespace Shizuku::Flow::Command;

ScreenPointParameter::ScreenPointParameter()
{
}

ScreenPointParameter::ScreenPointParameter(const Types::Point<int>& p_pos) : position(p_pos)
{
}