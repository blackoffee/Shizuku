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
public:
    int m_currentObstId = -1;
    float m_currentZ = -1000.f;
    float m_currentObstSize = 0.f;
    float4* m_rayCastIntersect;
    Obstruction::Shape m_currentObstShape = Obstruction::SQUARE;
    Obstruction* m_obstructions;
    Panel* m_parent;

    GLint m_viewport[4];
    GLdouble m_modelMatrix[16];
    GLdouble m_projectionMatrix[16];

    //view transformations
    float m_rotate_x = 60.f;
    float m_rotate_z = 30.f;
    float m_translate_x = 0.f;
    float m_translate_y = 0.8f;
    float m_translate_z = -0.2f;
    int m_paused = 0;
    float m_scaleFactor = 1.f;

    GraphicsManager();
    GraphicsManager(Panel* panel);

    void GetSimCoordFromMouseCoord(int &xOut, int &yOut, Mouse mouse);
    void GetSimCoordFromFloatCoord(int &xOut, int &yOut, float xf, float yf);
    void GetMouseRay(float3 &rayOrigin, float3 &rayDir, int mouseX, int mouseY);
    int GetSimCoordFrom3DMouseClickOnObstruction(int &xOut, int &yOut, Mouse mouse);
    void GetSimCoordFrom2DMouseRay(int &xOut, int &yOut, Mouse mouse);
    void GetSimCoordFrom2DMouseRay(int &xOut, int &yOut, int mouseX, int mouseY);
    void ClickDown(Mouse mouse);
    void Drag(int xi, int yi, float dxf, float dyf, int button);
    void Wheel(int button, int dir, int x, int y);
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
    
    void UpdateViewTransformations();

};

float GetDistanceBetweenTwoPoints(float x1, float y1, float x2, float y2);