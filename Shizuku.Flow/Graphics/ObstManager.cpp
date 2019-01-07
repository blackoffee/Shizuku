#include "ObstManager.h"
#include "common.h"
#include "Shizuku.Core/Ogl/Ogl.h"
#include <GLEW/glew.h>

using namespace Shizuku::Core;
using namespace Shizuku::Flow;

ObstManager::ObstManager(std::shared_ptr<Ogl> p_ogl)
{
    m_ogl = p_ogl;
    m_obsts = std::make_shared<std::list<std::shared_ptr<Obstruction>>>();
    m_obstData = new Obstruction[MAXOBSTS];
    m_pillars = std::map<const int, std::shared_ptr<Pillar>>();
}

void ObstManager::Initialize()
{
    for (int i = 0; i < MAXOBSTS; i++)
    {
        m_obstData[i] = Obstruction();
    }

    m_ogl->CreateBuffer(GL_SHADER_STORAGE_BUFFER, m_obstData, MAXOBSTS, "managed_obsts", GL_STATIC_DRAW);
}

void ObstManager::AddObst(const Obstruction& p_obst)
{
    m_obsts->push_front(std::make_shared<Obstruction>(p_obst));

    int i = 0;
    for (auto& obst : *m_obsts)
    {
        m_obstData[i] = *obst;
        ++i;
        if (i >= MAXOBSTS)
            break;
    }

    m_ogl->UpdateBufferData(GL_SHADER_STORAGE_BUFFER, m_obstData, MAXOBSTS, "managed_obsts", GL_STATIC_DRAW);
}

void ObstManager::RemoveObst(Obstruction& p_obst)
{
	m_obsts->remove_if(
		[&](std::shared_ptr<Obstruction> obst)->bool {
		return &p_obst == obst.get();
	}
	);

	auto obst = m_obsts->begin();
	for (int i = 0; i < MAXOBSTS; ++i)
	{
		if (obst != m_obsts->end())
		{
			m_obstData[i] = **obst;
			++obst;
		}
		else
		{
			m_obstData[i] = Obstruction();
		}
	}

    m_ogl->UpdateBufferData(GL_SHADER_STORAGE_BUFFER, m_obstData, MAXOBSTS, "managed_obsts", GL_STATIC_DRAW);
}

std::weak_ptr<std::list<std::shared_ptr<Obstruction>>> ObstManager::Obsts()
{
	return m_obsts;
}

void ObstManager::RenderPillars(const glm::mat4 &modelMatrix, const glm::mat4 &projectionMatrix, const glm::vec3& p_cameraPos)
{
    for (const auto pillar : m_pillars)
    {
        pillar.second->Draw(modelMatrix, projectionMatrix, p_cameraPos);
    }
}

void ObstManager::UpdatePillar(const int obstId, const PillarDefinition& p_def)
{
    const auto mapIt = m_pillars.lower_bound(obstId);
    if (mapIt != m_pillars.end() && !m_pillars.key_comp()(obstId, mapIt->first))
    {
        m_pillars[obstId]->SetDefinition(p_def);
    }
    else
    {
        std::shared_ptr<Pillar> pillar = std::make_shared<Pillar>(m_ogl);
        pillar->Initialize();
        pillar->SetDefinition(p_def);
        m_pillars.insert(mapIt, std::map<const int, std::shared_ptr<Pillar>>::value_type(obstId, pillar));
    }
}

void ObstManager::RemovePillar(const int obstId)
{
    m_pillars.erase(obstId);
}

