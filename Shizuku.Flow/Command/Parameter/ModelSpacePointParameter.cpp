#include "Command/Parameter/ModelSpacePointParameter.h"

using namespace Shizuku::Core;
using namespace Shizuku::Flow::Command;

ModelSpacePointParameter::ModelSpacePointParameter()
{
}

ModelSpacePointParameter::ModelSpacePointParameter(const Types::Point<float>& p_pos) : Position(p_pos)
{
}