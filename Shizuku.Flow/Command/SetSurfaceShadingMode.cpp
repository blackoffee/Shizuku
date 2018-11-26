#include "SetSurfaceShadingMode.h"
#include "Graphics/GraphicsManager.h"
#include "Graphics/ShadingMode.h"

using namespace Shizuku::Flow::Command;

SetSurfaceShadingMode::SetSurfaceShadingMode(GraphicsManager &p_graphicsManager) : Command(p_graphicsManager)
{
}

void SetSurfaceShadingMode::Start(const SurfaceShadingMode p_mode)
{
    GraphicsManager* graphicsManager = GetGraphicsManager();
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

