#include "SetViscosity.h"
#include "Graphics/GraphicsManager.h"
#include "Parameter/ViscosityParameter.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

SetViscosity::SetViscosity(Flow& p_flow) : Command(p_flow)
{
}

void SetViscosity::Start(boost::any const p_param)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    try
    {
        const ViscosityParameter& visc = boost::any_cast<ViscosityParameter>(p_param);
        graphicsManager->SetViscosity(visc.Viscosity);
    }
    catch (boost::bad_any_cast &e)
    {
        throw (e.what());
    }
}
