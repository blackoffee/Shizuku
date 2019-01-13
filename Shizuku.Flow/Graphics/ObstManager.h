#pragma once

#include "Pillar.h"
#include "PillarDefinition.h"
#include "HitParams.h"
#include "RenderParams.h"
#include "ObstDefinition.h"

#include "Shizuku.Core/Types/Point.h"

#include <memory>
#include <vector>
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
	class Obst;

    class ObstManager
    {
    private:
        std::shared_ptr<Core::Ogl> m_ogl;
		float m_waterHeight;

		// make this Obst. Should Obst hold Pillar? or just ptr to it?
        //std::shared_ptr<std::list<std::shared_ptr<ObstDefinition>>> m_obsts;
        std::shared_ptr<std::list<std::shared_ptr<Obst>>> m_obsts;
		std::list<std::shared_ptr<Obst>> m_selection;
		std::list<std::shared_ptr<Obst>> m_preSelection;
        ObstDefinition* m_obstData;


        std::shared_ptr<Core::ShaderProgram> m_shaderProgram;

		std::map<const int, std::shared_ptr<Pillar>> m_pillars;
    	void RemovePillar(const int obstId);
		void RefreshObstStates();
		void DoClearSelection();
		void DoClearPreSelection();

    public:
        ObstManager(std::shared_ptr<Core::Ogl> p_ogl);

		void SetWaterHeight(const float p_height);
		int ObstCount();

		//maybe remove this?
        void AddObstructionToSelection(const HitParams& p_params);
        void RemoveObstructionFromSelection(const HitParams& p_params);
		void ClearSelection();

        void AddObstructionToPreSelection(const HitParams& p_params);
        void RemoveObstructionFromPreSelection(const HitParams& p_params);
		void ClearPreSelection();

		void AddPreSelectionToSelection();
		void RemovePreSelectionFromSelection();

        void CreateObst(const ObstDefinition& p_obst);
		void DeleteSelectedObsts();
		void MoveSelectedObst(const Point<int>& p_pos, const Point<int>& p_diff);

        void UpdateObst(const ObstDefinition& p_obst);

        void AddObstruction(const Point<int>& p_simPos);
        void AddObstruction(const Point<float>& p_modelSpacePos);
        void RemoveObstruction(const int simX, const int simY);
        void RemoveSpecifiedObstruction(const int obstId);
        //int PickObstruction(const Point<int>& p_pos);
        void MoveObstruction(int obstId, const Point<int>& p_pos, const Point<int>& p_diff);

        std::weak_ptr<std::list<std::shared_ptr<Obst>>> Obsts();

		void UpdatePillar(const int obstId, const PillarDefinition& p_def);
		void Render(const RenderParams& p_params);

		bool IsInsideObstruction(const Point<float>& p_modelCoord);

        void Initialize();
    };
} }
