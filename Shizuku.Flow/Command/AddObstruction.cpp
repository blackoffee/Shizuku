#include "AddObstruction.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"
#include "Parameter/ModelSpacePointParameter.h"

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
        const ModelSpacePointParameter& pos = boost::any_cast<ModelSpacePointParameter>(p_param);
        graphicsManager->AddObstruction(pos.Position);
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}