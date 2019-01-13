#include "ObstManager.h"
#include "Obst.h"
#include "common.h"
#include "Shizuku.Core/Ogl/Ogl.h"
#include <GLEW/glew.h>

using namespace Shizuku::Core;
using namespace Shizuku::Flow;

namespace {
    float PillarHeightFromDepth(const float p_depth)
    {
        return p_depth + 0.3f;
    }

	//TODO this is used in many places
	void GetMouseRay(glm::vec3 &p_rayOrigin, glm::vec3 &p_rayDir, const HitParams& p_params)
	{
		glm::mat4 mvp = p_params.Projection*p_params.Modelview;
		glm::mat4 mvpInv = glm::inverse(mvp);
		glm::vec4 v1 = { (float)p_params.ScreenPos.X / (p_params.ViewSize.Width)*2.f - 1.f, (float)p_params.ScreenPos.Y / (p_params.ViewSize.Height)*2.f - 1.f, 0.0f*2.f - 1.f, 1.0f };
		glm::vec4 v2 = { (float)p_params.ScreenPos.X / (p_params.ViewSize.Width)*2.f - 1.f, (float)p_params.ScreenPos.Y / (p_params.ViewSize.Height)*2.f - 1.f, 1.0f*2.f - 1.f, 1.0f };
		glm::vec4 r1 = mvpInv * v1;
		glm::vec4 r2 = mvpInv * v2;
		p_rayOrigin.x = r1.x / r1.w;
		p_rayOrigin.y = r1.y / r1.w;
		p_rayOrigin.z = r1.z / r1.w;
		p_rayDir.x = r2.x / r2.w - p_rayOrigin.x;
		p_rayDir.y = r2.y / r2.w - p_rayOrigin.y;
		p_rayDir.z = r2.z / r2.w - p_rayOrigin.z;
		float mag = sqrt(p_rayDir.x*p_rayDir.x + p_rayDir.y*p_rayDir.y + p_rayDir.z*p_rayDir.z);
		p_rayDir.x /= mag;
		p_rayDir.y /= mag;
		p_rayDir.z /= mag;
	}

	//! Hits against water surface and floor
	glm::vec3 GetModelSpaceCoordFromScreenPos(const HitParams& p_params, const boost::optional<float> p_modelSpaceZPos, const float p_waterDepth)
	{
		glm::vec3 rayOrigin, rayDir;
		GetMouseRay(rayOrigin, rayDir, p_params);

		float t;
		if (p_modelSpaceZPos.is_initialized())
		{
			const float z = p_modelSpaceZPos.value();
			t = (z - rayOrigin.z) / rayDir.z;
			return rayOrigin + t * rayDir;
		}
		else
		{
			const float t1 = (-1.f - rayOrigin.z) / rayDir.z;
			const float t2 = (-1.f + p_waterDepth - rayOrigin.z) / rayDir.z;
			t = std::min(t1, t2);
			glm::vec3 res = rayOrigin + t * rayDir;

			if (res.x <= 1.f && res.y <= 1.f && res.x >= -1.f && res.y >= -1.f)
			{
				res;
			}
			else
			{
				t = std::max(t1, t2);
				return rayOrigin.x + t * rayDir;
			}
		}
	}
}

ObstManager::ObstManager(std::shared_ptr<Ogl> p_ogl)
{
    m_ogl = p_ogl;
    m_obsts = std::make_shared<std::list<std::shared_ptr<Obst>>>();
    m_obstData = new ObstDefinition[MAXOBSTS];
	m_selection = std::list<std::shared_ptr<Obst>>();
}

void ObstManager::Initialize()
{
    for (int i = 0; i < MAXOBSTS; i++)
    {
        m_obstData[i] = ObstDefinition();
    }

    m_ogl->CreateBuffer(GL_SHADER_STORAGE_BUFFER, m_obstData, MAXOBSTS, "managed_obsts", GL_STATIC_DRAW);
}

int ObstManager::ObstCount()
{
	return m_obsts->size();
}

int ObstManager::SelectedObstCount()
{
	return m_selection.size();
}

int ObstManager::PreSelectedObstCount()
{
	return m_preSelection.size();
}

void ObstManager::SetWaterHeight(const float p_height)
{
	m_waterHeight = p_height;
	const float pillarHeight = PillarHeightFromDepth(p_height);
	for (auto& obst : *m_obsts)
	{
		obst->SetHeight(pillarHeight);
	}
}

void ObstManager::CreateObst(const ObstDefinition& p_obst)
{
    m_obsts->push_back(std::make_shared<Obst>(m_ogl, p_obst, PillarHeightFromDepth(m_waterHeight)));
	RefreshObstStates();
}

void ObstManager::AddObstructionToSelection(const HitParams& p_params)
{
	float dist = std::numeric_limits<float>::max();
	std::shared_ptr<Obst> closest;
	bool hit(false);
    for (const auto& obst : *m_obsts)
    {
		HitResult result = obst->Hit(p_params);
		if (result.Hit)
		{
			assert(result.Dist.is_initialized());
			hit = true;
			if (result.Dist < dist)
			{
				dist = result.Dist.value();
				closest = obst;
			}
		}
    }

	if (hit)
	{
		m_selection.push_back(closest);
		closest->SetHighlight(true);
	}

	RefreshObstStates();
}

void ObstManager::DoClearSelection()
{
	for (const auto& obst : m_selection)
	{
		obst->SetHighlight(false);
	}

	m_selection.clear();
}

void ObstManager::ClearSelection()
{
	DoClearSelection();

	RefreshObstStates();
}

void ObstManager::RemoveObstructionFromSelection(const HitParams& p_params)
{
	float dist = std::numeric_limits<float>::max();
	std::shared_ptr<Obst> closest;
	bool hit(false);
    for (const auto& obst : m_selection)
    {
		HitResult result = obst->Hit(p_params);
		if (result.Hit)
		{
			assert(result.Dist.is_initialized());
			hit = true;
			if (result.Dist < dist)
			{
				dist = result.Dist.value();
				closest = obst;
			}
		}
    }

	if (hit)
	{
		m_selection.remove(closest);
		closest->SetHighlight(false);
	}

	RefreshObstStates();
}

void ObstManager::AddObstructionToPreSelection(const HitParams& p_params)
{
	float dist = std::numeric_limits<float>::max();
	std::shared_ptr<Obst> closest;
	bool hit(false);
    for (const auto& obst : *m_obsts)
    {
		HitResult result = obst->Hit(p_params);
		if (result.Hit)
		{
			assert(result.Dist.is_initialized());
			hit = true;
			if (result.Dist < dist)
			{
				dist = result.Dist.value();
				closest = obst;
			}
		}
    }

	if (hit)
	{
		m_preSelection.push_back(closest);
		closest->SetHighlight(true);
	}

	RefreshObstStates();
}

void ObstManager::ClearPreSelection()
{
	DoClearPreSelection();

	RefreshObstStates();
}

void ObstManager::DoClearPreSelection()
{
	for (const auto& obst : m_selection)
	{
		obst->SetHighlight(false);
	}

	m_preSelection.clear();
}

void ObstManager::RemoveObstructionFromPreSelection(const HitParams& p_params)
{
	float dist = std::numeric_limits<float>::max();
	std::shared_ptr<Obst> closest;
	bool hit(false);
    for (const auto& obst : m_selection)
    {
		HitResult result = obst->Hit(p_params);
		if (result.Hit)
		{
			assert(result.Dist.is_initialized());
			hit = true;
			if (result.Dist < dist)
			{
				dist = result.Dist.value();
				closest = obst;
			}
		}
    }

	if (hit)
	{
		m_preSelection.remove(closest);
		closest->SetHighlight(false);
	}

	RefreshObstStates();
}

void ObstManager::AddPreSelectionToSelection()
{
	for (const auto& obst : m_preSelection)
	{
		//TODO: check if already exists
		m_selection.push_back(obst);
	}

	RefreshObstStates();
}

void ObstManager::RemovePreSelectionFromSelection()
{
	for (const auto& obst : m_preSelection)
	{
		//TODO: check if already exists
		m_selection.remove(obst);
	}

	RefreshObstStates();
}

void ObstManager::RefreshObstStates()
{
	for (const auto& obst : *m_obsts)
		obst->SetHighlight(false);
	for (const auto& obst : m_selection)
		obst->SetHighlight(true);
	for (const auto& obst : m_preSelection)
		obst->SetHighlight(true);

    int i = 0;
    for (auto& obst : *m_obsts)
    {
        m_obstData[i] = obst->Def();
        ++i;
        if (i >= MAXOBSTS)
            break;
    }

    m_ogl->UpdateBufferData(GL_SHADER_STORAGE_BUFFER, m_obstData, MAXOBSTS, "managed_obsts", GL_STATIC_DRAW);
}

void ObstManager::DeleteSelectedObsts()
{
	for (const auto& obst : m_selection)
	{
		m_obsts->remove(obst);
	}

	DoClearSelection();

	RefreshObstStates();
}

bool ObstManager::TryStartMoveSelectedObsts(const HitParams& p_params)
{
	float dist = std::numeric_limits<float>::max();
	bool hit(false);
    for (const auto& obst : m_selection)
    {
		HitResult result = obst->Hit(p_params);
		if (result.Hit)
		{
			assert(result.Dist.is_initialized());
			hit = true;
			if (result.Dist < dist)
			{
				dist = result.Dist.value();
			}
		}
    }

	if (hit)
	{
		glm::vec3 rayOrigin;
		glm::vec3 rayDir;
		GetMouseRay(rayOrigin, rayDir, p_params);
		m_moveOrigin = rayOrigin + dist * rayDir;
	}
	else
	{
		m_moveOrigin = boost::none;
	}

	return hit;
}

void ObstManager::MoveSelectedObsts(const HitParams& p_dest)
{
	assert(m_moveOrigin.has_value());
	const glm::vec3 destModelCoord = GetModelSpaceCoordFromScreenPos(p_dest, m_moveOrigin.value().z, m_waterHeight);
	const glm::vec3 trans = destModelCoord - m_moveOrigin.value();

	//need to get transform instead
	for (const auto& obst : m_selection)
	{
		ObstDefinition def = obst->Def();
		def.x += trans.x;
		def.y += trans.y;
		obst->SetDef(def);
	}

	m_moveOrigin = destModelCoord;
}

std::weak_ptr<std::list<std::shared_ptr<Obst>>> ObstManager::Obsts()
{
	return m_obsts;
}

void ObstManager::Render(const RenderParams& p_params)
{
    for (const auto& obst : *m_obsts)
    {
		obst->Render(p_params);
    }
}

bool ObstManager::IsInsideObstruction(const Point<float>& p_modelCoord)
{
	const float tolerance = 0.f;
	for (const auto obst : *m_obsts)
	{
		if (obst->Hit(p_modelCoord).Hit)
			return true;
	}

	return false;
}
