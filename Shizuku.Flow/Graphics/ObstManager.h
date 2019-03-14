#pragma once

#include "HitParams.h"
#include "RenderParams.h"
#include "ObstDefinition.h"
#include "Info/ObstInfo.h"

#include "Shizuku.Core/Types/Point.h"

#include "cuda_runtime.h"
#include <GLEW/glew.h>
#include "cuda_gl_interop.h"

#include <memory>
#include <set>

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
        boost::optional<glm::vec3> m_moveOrigin;

        std::shared_ptr<std::set<std::shared_ptr<Obst>>> m_obsts;
        std::set<std::shared_ptr<Obst>> m_selection;
        std::set<std::shared_ptr<Obst>> m_preSelection;
        ObstDefinition* m_obstData;

        std::shared_ptr<Core::ShaderProgram> m_shaderProgram;

        cudaGraphicsResource* m_cudaObstsResource;

        void RefreshObstStates();
        void DoClearSelection();
        void DoClearPreSelection();

    public:
        ObstManager(std::shared_ptr<Core::Ogl> p_ogl);

        void Initialize();

        void SetWaterHeight(const float p_height);

        // Queries
        int ObstCount();
        int SelectedObstCount();
        int PreSelectedObstCount();
        bool IsInsideObstruction(const Point<float>& p_modelCoord);
        boost::optional<const Info::ObstInfo> ObstInfo(const HitParams& p_params);
        cudaGraphicsResource* GetCudaObstsResource();

        void AddObstructionToPreSelection(const HitParams& p_params);
        void RemoveObstructionFromPreSelection(const HitParams& p_params);
        void ClearPreSelection();

        void AddPreSelectionToSelection();
        void RemovePreSelectionFromSelection();
        void TogglePreSelectionInSelection();
        void ClearSelection();

        void CreateObst(const ObstDefinition& p_obst);
        void DeleteSelectedObsts();
        bool TryStartMoveSelectedObsts(const HitParams& p_params);
        void MoveSelectedObsts(const HitParams& p_dest);

        glm::vec3 GetSurfaceOrFloorIntersect(const HitParams& p_params);

        std::weak_ptr<std::set<std::shared_ptr<Obst>>> Obsts();

        void Render(const RenderParams& p_params);
    };
} }
