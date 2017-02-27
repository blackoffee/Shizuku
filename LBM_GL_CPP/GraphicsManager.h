#pragma once

#include <GLEW/glew.h>
#include <GLUT/freeglut.h>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include "Common.h"
#include "kernel.h"


#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class Panel;
class Mouse;

class GraphicsManager
{
private:
    float m_currentZ = -1000.f;
    //view transformations
    float m_rotate_x = 60.f;
    float m_rotate_y = 0.f;
    float m_rotate_z = 30.f;
    float m_translate_x = 0.f;
    float m_translate_y = 0.8f;
    float m_translate_z = -0.2f;
    int m_currentObstId = -1;
    float m_currentObstSize = 0.f;
    Obstruction::Shape m_currentObstShape = Obstruction::SQUARE;
    ViewMode m_viewMode;
    Obstruction* m_obstructions;
    Panel* m_parent;
    bool m_paused = 0;
    float m_scaleFactor = 1.f;
    GLint m_viewport[4];
    GLdouble m_modelMatrix[16];
    GLdouble m_projectionMatrix[16];
    ContourVariable m_contourVar;


public:
    float4* m_rayCastIntersect_d;

    FW_API GraphicsManager(Panel* panel);

    FW_API float3 GetRotationTransforms();
    FW_API float3 GetTranslationTransforms();

    FW_API void SetCurrentObstSize(const float size);

    FW_API Obstruction::Shape GetCurrentObstShape();
    FW_API void SetCurrentObstShape(const Obstruction::Shape shape);

    FW_API ViewMode GetViewMode();
    FW_API void SetViewMode(const ViewMode viewMode);

    FW_API ContourVariable GetContourVar();
    FW_API void SetContourVar(const ContourVariable contourVar);

    FW_API void SetObstructionsPointer(Obstruction* obst);

    FW_API bool IsPaused();
    FW_API void TogglePausedState();

    FW_API float GetScaleFactor();
    FW_API void SetScaleFactor(const float scaleFactor);

    FW_API void ClickDown(Mouse mouse);


    FW_API void Drag(const int xi, const int yi, const float dxf, const float dyf,
        const int button);
    FW_API void Wheel(const int button, const int dir, const int x, const int y);
   
    FW_API void UpdateViewTransformations();

    FW_API void GetSimCoordFromMouseCoord(int &xOut, int &yOut, Mouse mouse);
    FW_API void GetSimCoordFromFloatCoord(int &xOut, int &yOut, const float xf, const float yf);
    FW_API void GetMouseRay(float3 &rayOrigin, float3 &rayDir, const int mouseX, const int mouseY);
    FW_API int GetSimCoordFrom3DMouseClickOnObstruction(int &xOut, int &yOut, Mouse mouse);
    FW_API void GetSimCoordFrom2DMouseRay(int &xOut, int &yOut, Mouse mouse);
    FW_API void GetSimCoordFrom2DMouseRay(int &xOut, int &yOut, const int mouseX, const int mouseY);
    FW_API void AddObstruction(Mouse mouse);
    FW_API void AddObstruction(const int simX, const int simY);
    FW_API void RemoveObstruction(Mouse mouse);
    FW_API void RemoveObstruction(const int simX, const int simY);
    FW_API void MoveObstruction(const int xi, const int yi, const float dxf, const float dyf);
    FW_API int FindUnusedObstructionId();
    FW_API int FindClosestObstructionId(Mouse mouse);
    FW_API int FindClosestObstructionId(const int simX, const int simY);
    FW_API int FindObstructionPointIsInside(const int x, const int y, const float tolerance=0.f);
    FW_API bool IsInClosestObstruction(Mouse mouse);
 
};

float GetDistanceBetweenTwoPoints(const float x1, const float y1,
    const float x2, const float y2);