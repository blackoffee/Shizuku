#include "Obst.h"
#include "Shizuku.Core/Types/Box.h"
#include "Shizuku.Core/Types/Point.h"

using namespace Shizuku::Core;
using namespace Shizuku::Flow;

namespace
{
	PillarDefinition PillarDefFromObstDef(const ObstDefinition& p_def, const float p_height)
	{
		const Types::Point<float> pillarPos(p_def.x, p_def.y);
		const Types::Box<float> pillarSize(2.f*p_def.r1, 2.f*p_def.r1, p_height);
		return PillarDefinition(pillarPos, pillarSize);
	}
}

Obst::Obst(std::shared_ptr<Shizuku::Core::Ogl> p_ogl, const ObstDefinition& p_def, const float p_height)
	:m_def(p_def), m_vis(p_ogl), m_height(p_height)
{
	m_vis.Initialize();
	m_vis.SetDefinition(PillarDefFromObstDef(p_def, m_height));
}

const ObstDefinition& Obst::Def()
{
	return m_def;
}

void Obst::SetDef(const ObstDefinition& p_def)
{
	m_def = p_def;
	m_vis.SetDefinition(PillarDefFromObstDef(p_def, m_height));
}

void Obst::SetHeight(const float p_height)
{
	m_height = p_height;
	PillarDefinition def = m_vis.Def();
	def.SetHeight(p_height);
	m_vis.SetDefinition(def);
}

void Obst::Render(const RenderParams& p_params)
{
	m_vis.Render(p_params);
}

HitResult Obst::Hit(const HitParams& p_params)
{
	return m_vis.Hit(p_params);
}

HitResult Obst::Hit(const Types::Point<float>& p_modelSpace)
{
	const float r1 = m_def.r1;
	const bool result = (abs(p_modelSpace.X - m_def.x) < r1 && abs(p_modelSpace.Y - m_def.y) < r1);
	return HitResult{
		result,
		boost::none
	};
}
