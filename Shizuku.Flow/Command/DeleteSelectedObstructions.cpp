#include "DeleteSelectedObstructions.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"
#include "Parameter/ScreenPointParameter.h"

using namespace Shizuku::Core::Types;
using namespace Shizuku::Flow::Command;

DeleteSelectedObstructions::DeleteSelectedObstructions(Flow& p_flow) : Command(p_flow)
{
}

void DeleteSelectedObstructions::Start(boost::any const p_param)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    graphicsManager->DeleteSelectedObstructions();
}
