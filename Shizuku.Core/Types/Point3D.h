#pragma once

#ifdef SHIZUKU_CORE_EXPORTS  
#define CORE_API __declspec(dllexport)   
#else  
#define CORE_API __declspec(dllimport)   
#endif  

namespace Shizuku{ namespace Core{ namespace Types{
    template <typename T>
    class Point3D
    {
    public:
        T X;
        T Y;
        T Z;

        Point3D()
        {
        }
        Point3D(T p_x, T p_y, T p_z) : X(p_x), Y(p_y), Z(p_z)
        {
        }
    };

    template <typename T>
    Point3D<int> operator+(const Point3D<T>& p_a, const Point3D<T>& p_b)
    {
        return Point3D<T>(p_a.X + p_b.X, p_a.Y + p_b.Y, p_a.Z + p_b.Z);
    }

    template <typename T>
    Point3D<int> operator-(const Point3D<T>& p_a, const Point3D<T>& p_b)
    {
        return Point3D<T>(p_a.X - p_b.X, p_a.Y - p_b.Y, p_a.Z - p_b.Z);
    }

    template <typename T>
    bool operator==(const Point3D<T>& p_a, const Point3D<T>& p_b)
    {
        return p_a.X == p_b.X && p_a.Y == p_b.Y && p_a.Z == p_b.Z;
    }

    template <typename T>
    bool operator!=(const Point3D<T>& p_a, const Point3D<T>& p_b)
    {
        return !(p_a == p_b);
    }
} } }
