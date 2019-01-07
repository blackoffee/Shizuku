#include "ObstManager.h"
#include "common.h"
#include "Shizuku.Core/Ogl/Ogl.h"
#include <GLEW/glew.h>

using namespace Shizuku::Core;
using namespace Shizuku::Flow;

namespace {
    int GetObstImage(const Point<float> modelCoord,
        ObstDefinition* obstructions, const float tolerance = 0.f)
    {
        for (int i = 0; i < MAXOBSTS; i++){
            if (obstructions[i].state != State::INACTIVE)
            {
                float r1 = obstructions[i].r1 + tolerance;
                if (obstructions[i].shape == Shape::SQUARE){
                    if (abs(modelCoord.X - obstructions[i].x) < r1 && abs(modelCoord.Y - obstructions[i].y) < r1)
                        return 1;
                }
            }
        }
        return 0;
    }
}

ObstManager::ObstManager(std::shared_ptr<Ogl> p_ogl)
{
    m_ogl = p_ogl;
    m_obsts = std::make_shared<std::list<std::shared_ptr<ObstDefinition>>>();
    m_obstData = new ObstDefinition[MAXOBSTS];
    m_pillars = std::map<const int, std::shared_ptr<Pillar>>();
}

void ObstManager::Initialize()
{
    for (int i = 0; i < MAXOBSTS; i++)
    {
        m_obstData[i] = ObstDefinition();
    }

    m_ogl->CreateBuffer(GL_SHADER_STORAGE_BUFFER, m_obstData, MAXOBSTS, "managed_obsts", GL_STATIC_DRAW);
}

void ObstManager::AddObst(const ObstDefinition& p_obst)
{
    m_obsts->push_front(std::make_shared<ObstDefinition>(p_obst));

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

void ObstManager::RemoveObst(ObstDefinition& p_obst)
{
	m_obsts->remove_if(
		[&](std::shared_ptr<ObstDefinition> obst)->bool {
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
			m_obstData[i] = ObstDefinition();
		}
	}

    m_ogl->UpdateBufferData(GL_SHADER_STORAGE_BUFFER, m_obstData, MAXOBSTS, "managed_obsts", GL_STATIC_DRAW);
}

std::weak_ptr<std::list<std::shared_ptr<ObstDefinition>>> ObstManager::Obsts()
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

bool ObstManager::IsInsideObstruction(const Point<float>& p_modelCoord)
{
	const float tolerance = 0.f;
	for (const auto obst : *m_obsts)
	{
		if (obst->state != State::INACTIVE)
		{
            const float r1 = obst->r1 + tolerance;
			if (abs(p_modelCoord.X - obst->x) < r1 && abs(p_modelCoord.Y - obst->y) < r1)
				return true;
		}
	}

	return false;
}
