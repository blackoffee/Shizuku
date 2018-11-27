#pragma once

#ifdef SHIZUKU_CORE_EXPORTS  
#define CORE_API __declspec(dllexport)   
#else  
#define CORE_API __declspec(dllimport)   
#endif  

namespace Shizuku{ namespace Core{ namespace Types{
    template <typename T>
    class Point
    {
    public:
        T X;
        T Y;

        Point()
        {
        }
        Point(T p_x, T p_y) : X(p_x), Y(p_y)
        {
        }
    };

    template <typename T>
    Point<int> operator+(const Point<T>& p_a, const Point<T>& p_b)
    {
        return Point<T>(p_a.X + p_b.X, p_a.Y + p_b.Y);
    }

    template <typename T>
    Point<int> operator-(const Point<T>& p_a, const Point<T>& p_b)
    {
        return Point<T>(p_a.X - p_b.X, p_a.Y - p_b.Y);
    }

    template <typename T>
    bool operator==(const Point<T>& p_a, const Point<T>& p_b)
    {
        return p_a.X == p_b.X && p_a.Y == p_b.Y;
    }

    template <typename T>
    bool operator!=(const Point<T>& p_a, const Point<T>& p_b)
    {
        return !(p_a == p_b);
    }
} } }
