#pragma once

#include "Shizuku.Core/Rect.h"

#ifdef SHIZUKU_FLOW_EXPORTS  
#define FLOW_API __declspec(dllexport)   
#else  
#define FLOW_API __declspec(dllimport)   
#endif  

using namespace Shizuku::Core;

namespace Shizuku { namespace Flow{
    class GraphicsManager;
    class Impl;
    class FLOW_API Flow{
    public:
        Flow();
        ~Flow();

        void Initialize();

        void Update();

		void SetUpFrame();

        void Draw3D();

        void Resize(const Rect<int>& p_size);

        //TODO: remove this
        GraphicsManager* Graphics();

    private:
        Impl* m_impl;
    };
} }