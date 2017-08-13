#pragma once
#include "common.h"
#include "cuda_runtime.h"
#include <GLEW/glew.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>


#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class Panel;
class ShaderManager;
class CudaLbm;

class FW_API GraphicsManager
{
private:
    float m_currentZ = -1000.f;
    //view transformations
    float3 m_rotate;
    float3 m_translate;
    int m_currentObstId = -1;
    float m_currentObstSize = 0.f;
    Shape m_currentObstShape = Shape::SQUARE;
    ViewMode m_viewMode;
    bool m_rayTracingPaused = false;
    glm::vec4 m_cameraPosition;
    Obstruction* m_obstructions;
    Panel* m_parent;
    float m_scaleFactor = 1.f;
    GLint m_viewport[4];
    GLdouble m_modelMatrix[16];
    GLdouble m_projectionMatrix[16];
    float m_contourMinValue;
    float m_contourMaxValue;
    ContourVariable m_contourVar;
    ShaderManager* m_graphics;
    bool m_useCuda = true;
    float4* m_rayCastIntersect_d;

public:
    GraphicsManager(Panel* panel);

    void UseCuda(bool useCuda);

    float3 GetRotationTransforms();
    float3 GetTranslationTransforms();

    void SetCurrentObstSize(const float size);

    Shape GetCurrentObstShape();
    void SetCurrentObstShape(const Shape shape);

    ViewMode GetViewMode();
    void SetViewMode(const ViewMode viewMode);

    float GetContourMinValue();
    float GetContourMaxValue();
    void SetContourMinValue(const float contourMinValue);
    void SetContourMaxValue(const float contourMaxValue);
    ContourVariable GetContourVar();
    void SetContourVar(const ContourVariable contourVar);

    void SetObstructionsPointer(Obstruction* obst);

    float GetScaleFactor();
    void SetScaleFactor(const float scaleFactor);

    CudaLbm* GetCudaLbm();
    ShaderManager* GetGraphics();

    bool IsCudaCapable();

    void CenterGraphicsViewToGraphicsPanel(const int leftPanelWidth);
    void SetUpGLInterop();
    void SetUpShaders();
    void SetUpCuda();
    void RunCuda();
    void RunSurfaceRefraction();
    void RunComputeShader();
    void RunSimulation();
    void RenderFloorToTexture();
    void RenderVbo();
    bool ShouldRenderFloor();
    bool ShouldRefractSurface();

    void Zoom(const int dir, const float mag);
    void Pan(const float dx, const float dy);
    void Rotate(const float dx, const float dy);
    int PickObstruction(const float mouseXf, const float mouseYf);
    int PickObstruction(const int mouseX, const int mouseY);
    void UnpickObstruction();
    void MoveObstruction(int obstId, const float mouseXf, const float mouseYf,
        const float dxf, const float dyf);
    void SetRayTracingPausedState(const bool state);
   
    void UpdateViewTransformations();
    void UpdateGraphicsInputs();
    void UpdateDomainDimensions();
    void UpdateObstructionScales();
    void UpdateLbmInputs();
    glm::vec4 GetViewportMatrix();
    glm::mat4 GetModelMatrix();
    glm::mat4 GetProjectionMatrix();
    void SetModelMatrix(glm::mat4 modelMatrix);
    void SetProjectionMatrix(glm::mat4 projMatrix);

    void GetSimCoordFromFloatCoord(int &xOut, int &yOut, const float xf, const float yf);
    void GetMouseRay(float3 &rayOrigin, float3 &rayDir, const int mouseX, const int mouseY);
    glm::vec4 GetCameraDirection();
    glm::vec4 GetCameraPosition();
    int GetSimCoordFrom3DMouseClickOnObstruction(int &xOut, int &yOut, 
        const int mouseX, const int mouseY);
    void GetSimCoordFromMouseRay(int &xOut, int &yOut, const int mouseX, const int mouseY);
    void GetSimCoordFromMouseRay(int &xOut, int &yOut, const float mouseXf, const float mouseYf,
        const float planeZ);
    void GetSimCoordFromMouseRay(int &xOut, int &yOut, const int mouseX, const int mouseY,
        const float planeZ);
    void AddObstruction(const int simX, const int simY);
    void RemoveObstruction(const int simX, const int simY);
    void RemoveSpecifiedObstruction(const int obstId);
    int FindUnusedObstructionId();
    int FindClosestObstructionId(const int simX, const int simY);
    int FindObstructionPointIsInside(const int x, const int y, const float tolerance=0.f);
 
};

float GetDistanceBetweenTwoPoints(const float x1, const float y1,
    const float x2, const float y2);
