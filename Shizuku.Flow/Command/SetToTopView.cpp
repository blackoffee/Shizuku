#include "SetToTopView.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

SetToTopView::SetToTopView(Flow& p_flow) : Command(p_flow)
{
}

void SetToTopView::Start(boost::any const p_param)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    try
    {
        const bool ortho = boost::any_cast<bool>(p_param);
        graphicsManager->SetToTopView(ortho);
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}

