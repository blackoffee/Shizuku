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
    void GetSimCoordFromFloatCoord(int &xOut, int &yOut, float xf, float yf);
    void GetMouseRay(float3 &rayOrigin, float3 &rayDir, int mouseX, int mouseY);
    int GetSimCoordFrom3DMouseClickOnObstruction(int &xOut, int &yOut, Mouse mouse);
    void GetSimCoordFrom2DMouseRay(int &xOut, int &yOut, Mouse mouse);
    void GetSimCoordFrom2DMouseRay(int &xOut, int &yOut, int mouseX, int mouseY);
    void AddObstruction(Mouse mouse);
    void AddObstruction(int simX, int simY);
    void RemoveObstruction(Mouse mouse);
    void RemoveObstruction(int simX, int simY);
    void MoveObstruction(int xi, int yi, float dxf, float dyf);
    int FindUnusedObstructionId();
    int FindClosestObstructionId(Mouse mouse);
    int FindClosestObstructionId(int simX, int simY);
    int FindObstructionPointIsInside(int x, int y, float tolerance=0.f);
    bool IsInClosestObstruction(Mouse mouse);
 
public:
    float4* m_rayCastIntersect_d;

    GraphicsManager(Panel* panel);

    float3 GetRotationTransforms();
    float3 GetTranslationTransforms();

    void SetCurrentObstSize(float size);

    Obstruction::Shape GetCurrentObstShape();
    void SetCurrentObstShape(Obstruction::Shape shape);

    ViewMode GetViewMode();
    void SetViewMode(ViewMode viewMode);

    ContourVariable GetContourVar();
    void SetContourVar(ContourVariable contourVar);

    void SetObstructionsPointer(Obstruction* obst);

    bool IsPaused();
    void TogglePausedState();

    float GetScaleFactor();
    void SetScaleFactor(float scaleFactor);


    void ClickDown(Mouse mouse);
    void Drag(int xi, int yi, float dxf, float dyf, int button);
    void Wheel(int button, int dir, int x, int y);
   
    void UpdateViewTransformations();

};

float GetDistanceBetweenTwoPoints(float x1, float y1, float x2, float y2);