#pragma once

#ifdef SHIZUKU_CORE_EXPORTS  
#define CORE_API __declspec(dllexport)   
#else  
#define CORE_API __declspec(dllimport)   
#endif  

namespace Shizuku{ namespace Core{ namespace Types{
    template <typename T>
    class Box
    {
    public:
        T Width;
        T Height;
        T Depth;

        Box()
        {
        }
        Box(T w, T h, T d) : Width(w), Height(h), Depth(d)
        {
        }
    };
} } }
