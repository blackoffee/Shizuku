#include "Flow.h"
#include "Graphics/GraphicsManager.h"
#include "Graphics/CudaLbm.h"
#include <memory>

using namespace Shizuku::Core;
using namespace Shizuku::Flow;

class Shizuku::Flow::Impl
{
private:
    std::shared_ptr<GraphicsManager> m_graphics;

public:
    Impl()
    {
        m_graphics = std::make_shared<GraphicsManager>();
    }

    std::shared_ptr<GraphicsManager> Graphics()
    {
        return m_graphics;
    }
};

Flow::Flow()
{
    m_impl = new Impl();
}

Flow::~Flow()
{
    delete m_impl;
}


void Flow::Initialize()
{
    m_impl->Graphics()->SetUpGLInterop();
    m_impl->Graphics()->SetUpCuda();
    m_impl->Graphics()->SetUpShaders();
}

void Flow::Update()
{
    m_impl->Graphics()->UpdateGraphicsInputs();
    m_impl->Graphics()->GetCudaLbm()->UpdateDeviceImage();

    m_impl->Graphics()->RunSimulation();

    m_impl->Graphics()->RenderCausticsToTexture();

    m_impl->Graphics()->RunSurfaceRefraction();

    m_impl->Graphics()->UpdateViewMatrices();
}

void Flow::Draw3D()
{
    m_impl->Graphics()->Render();
}

void Flow::Resize(const Rect<int>& p_size)
{
    m_impl->Graphics()->SetViewport(p_size);
}

GraphicsManager* Flow::Graphics()
{
    return m_impl->Graphics().get();
}

