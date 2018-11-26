#include "SetSurfaceShadingMode.h"
#include "Graphics/GraphicsManager.h"
#include "Graphics/ShadingMode.h"
#include "Flow.h"

using namespace Shizuku::Flow::Command;

SetSurfaceShadingMode::SetSurfaceShadingMode(Flow& p_flow) : Command(p_flow)
{
}

void SetSurfaceShadingMode::Start(const SurfaceShadingMode p_mode)
{
    GraphicsManager* graphicsManager= m_flow->Graphics();
    switch (p_mode)
    {
    case SurfaceShadingMode::RayTracing:
        graphicsManager->SetSurfaceShadingMode(ShadingMode::RayTracing);
        break;
    case SurfaceShadingMode::SimplifiedRayTracing:
        graphicsManager->SetSurfaceShadingMode(ShadingMode::SimplifiedRayTracing);
        break;
    case SurfaceShadingMode::Phong:
        graphicsManager->SetSurfaceShadingMode(ShadingMode::Phong);
        break;
    default:
        throw "Unexpected shading mode";
    }
}

