#pragma once

#include <algorithm>

#ifdef SHIZUKU_CORE_EXPORTS  
#define CORE_API __declspec(dllexport)   
#else  
#define CORE_API __declspec(dllimport)   
#endif  

namespace Shizuku{
    namespace Core{
        template <typename T>
        class MinMax
        {
        public:
            T Min;
            T Max;

            MinMax()
            {
            }
            MinMax(T p_min, T p_max) : Min(p_min), Max(p_max)
            {
            }

            void Clamp(const MinMax<T>& p_clamper)
            {
                Min = std::min(std::max(Min, p_clamper.Min), p_clamper.Max);
                Max = std::min(std::max(Max, p_clamper.Min), p_clamper.Max);
            }
        };

        template <typename T>
        bool operator==(const MinMax<T>& p_minMax1, const MinMax<T>& p_minMax2)
        {
            return p_minMax1.Min == p_minMax2.Min && p_minMax1.Max == p_minMax2.Max;
        }

        template <typename T>
        bool operator!=(const MinMax<T>& p_minMax1, const MinMax<T>& p_minMax2)
        {
            return !(p_minMax1 == p_minMax2);
        }

    }
}
