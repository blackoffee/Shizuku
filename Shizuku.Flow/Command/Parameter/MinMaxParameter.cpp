#include "Command/Parameter/MinMaxParameter.h"

using namespace Shizuku::Flow::Command;

MinMaxParameter::MinMaxParameter()
{
}

MinMaxParameter::MinMaxParameter(const Shizuku::Core::Types::MinMax<float>& p_minMax) : MinMax(p_minMax)
{
}