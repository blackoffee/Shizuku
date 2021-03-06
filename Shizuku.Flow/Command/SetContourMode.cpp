#include "SetContourMode.h"
#include "Graphics/GraphicsManager.h"
#include "Flow.h"
#include "common.h"

using namespace Shizuku::Flow::Command;

SetContourMode::SetContourMode(Flow &p_flow) : Command(p_flow)
{
}

void SetContourMode::Start(const ContourMode p_contourMode)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    ContourVariable contour;
    switch (p_contourMode)
    {
    case ContourMode::VelocityMagnitude:
        contour = VEL_MAG;
        break;
    case ContourMode::VelocityU:
        contour = VEL_U;
        break;
    case ContourMode::VelocityV:
        contour = VEL_V;
        break;
    case ContourMode::Pressure:
        contour = PRESSURE;
        break;
    case ContourMode::StrainRate:
        contour = STRAIN_RATE;
        break;
    case ContourMode::Water:
        contour = WATER_RENDERING;
        break;
    case ContourMode::NUMB_CONTOUR_MODE:
        throw "Unexpected contour mode";
    }

    graphicsManager->SetContourVar(contour);
}

