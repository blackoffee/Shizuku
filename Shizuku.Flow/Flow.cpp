#include "Flow.h"
#include "Graphics/GraphicsManager.h"
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
}

void Flow::Draw3D()
{
}

void Flow::Resize(const Rect<int>& p_size)
{
    m_impl->Graphics()->SetViewport(p_size);
}

GraphicsManager* Flow::Graphics()
{
    return m_impl->Graphics().get();
}

