#pragma once

#ifdef SHIZUKU_CORE_EXPORTS  
#define CORE_API __declspec(dllexport)   
#else  
#define CORE_API __declspec(dllimport)   
#endif  

namespace Shizuku{
    namespace Core{
        template <typename T>
        class Rect
        {
        public:
            T Width;
            T Height;

            Rect()
            {
            }
            Rect(T w, T h) : Width(w), Height(h)
            {
            }
        };
    }
}
