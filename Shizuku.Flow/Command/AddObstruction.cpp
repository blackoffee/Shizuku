#include "AddObstruction.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"
#include "Parameter/ScreenPointParameter.h"

using namespace Shizuku::Flow::Command;

AddObstruction::AddObstruction(Flow& p_flow) : Command(p_flow)
{
    m_state = INACTIVE;
}

void AddObstruction::Start(boost::any const p_param)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    try
    {
        const ScreenPointParameter& pos = boost::any_cast<ScreenPointParameter>(p_param);
        int simX, simY;
        const float waterHeight = graphicsManager->GetWaterHeight();
        graphicsManager->GetSimCoordFromMouseRay(simX, simY, pos.Position, waterHeight);
        graphicsManager->AddObstruction(simX, simY);
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}