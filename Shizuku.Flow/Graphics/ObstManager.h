#pragma once

#include "Pillar.h"
#include "PillarDefinition.h"
#include "Shizuku.Core/Types/Point.h"
#include "Obstruction.h"
#include <memory>
#include <map>
#include <list>

namespace Shizuku{
namespace Core{
    class Ogl;
    class ShaderProgram;
}
}

using namespace Shizuku::Core::Types;
using namespace Shizuku::Flow;

namespace Shizuku { namespace Flow{
    class ObstManager
    {
    private:
        std::shared_ptr<Core::Ogl> m_ogl;
        std::shared_ptr<std::list<std::shared_ptr<Obstruction>>> m_obsts;
        Obstruction* m_obstData;

        std::shared_ptr<Core::ShaderProgram> m_shaderProgram;

		std::map<const int, std::shared_ptr<Pillar>> m_pillars;
    	void RemovePillar(const int obstId);
    public:
        ObstManager(std::shared_ptr<Core::Ogl> p_ogl);

        void AddObst(const Obstruction& p_obst);
        Obstruction& PickObstruction(const Point<int>& p_pos);
        void UpdateObst(const Obstruction& p_obst);
        void RemoveObst(Obstruction& p_obst);

        void AddObstruction(const Point<int>& p_simPos);
        void AddObstruction(const Point<float>& p_modelSpacePos);
        void RemoveObstruction(const int simX, const int simY);
        void RemoveSpecifiedObstruction(const int obstId);
        //int PickObstruction(const Point<int>& p_pos);
        void MoveObstruction(int obstId, const Point<int>& p_pos, const Point<int>& p_diff);

        std::weak_ptr<std::list<std::shared_ptr<Obstruction>>> Obsts();

		void UpdatePillar(const int obstId, const PillarDefinition& p_def);
		void RenderPillars(const glm::mat4 &p_modelMatrix, const glm::mat4 &p_projectionMatrix, const glm::vec3& p_cameraPos);

        void Initialize();
    };
} }
