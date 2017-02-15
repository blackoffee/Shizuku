#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <string>
#include <iostream>
#include <vector>
#include "Common.h"
#include "kernel.h"

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

    void GetSimCoordFromMouseCoord(int &xOut, int &yOut, Mouse mouse);
    void GetSimCoordFromFloatCoord(int &xOut, int &yOut, const float xf, const float yf);
    void GetMouseRay(float3 &rayOrigin, float3 &rayDir, const int mouseX, const int mouseY);
    int GetSimCoordFrom3DMouseClickOnObstruction(int &xOut, int &yOut, Mouse mouse);
    void GetSimCoordFrom2DMouseRay(int &xOut, int &yOut, Mouse mouse);
    void GetSimCoordFrom2DMouseRay(int &xOut, int &yOut, const int mouseX, const int mouseY);
    void AddObstruction(Mouse mouse);
    void AddObstruction(const int simX, const int simY);
    void RemoveObstruction(Mouse mouse);
    void RemoveObstruction(const int simX, const int simY);
    void MoveObstruction(const int xi, const int yi, const float dxf, const float dyf);
    int FindUnusedObstructionId();
    int FindClosestObstructionId(Mouse mouse);
    int FindClosestObstructionId(const int simX, const int simY);
    int FindObstructionPointIsInside(const int x, const int y, const float tolerance=0.f);
    bool IsInClosestObstruction(Mouse mouse);
 
public:
    float4* m_rayCastIntersect_d;

    GraphicsManager(Panel* panel);

    float3 GetRotationTransforms();
    float3 GetTranslationTransforms();

    void SetCurrentObstSize(const float size);

    Obstruction::Shape GetCurrentObstShape();
    void SetCurrentObstShape(const Obstruction::Shape shape);

    ViewMode GetViewMode();
    void SetViewMode(const ViewMode viewMode);

    ContourVariable GetContourVar();
    void SetContourVar(const ContourVariable contourVar);

    void SetObstructionsPointer(Obstruction* obst);

    bool IsPaused();
    void TogglePausedState();

    float GetScaleFactor();
    void SetScaleFactor(const float scaleFactor);

    void ClickDown(Mouse mouse);
    void Drag(const int xi, const int yi, const float dxf, const float dyf,
        const int button);
    void Wheel(const int button, const int dir, const int x, const int y);
   
    void UpdateViewTransformations();

};

float GetDistanceBetweenTwoPoints(const float x1, const float y1,
    const float x2, const float y2);