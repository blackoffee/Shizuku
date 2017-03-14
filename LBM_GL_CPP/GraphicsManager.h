#pragma once
#include <GLEW/glew.h>
#include <GLUT/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "helper_cuda.h"
#include "helper_cuda_gl.h"
#include "device_launch_parameters.h"

#include "Common.h"
#include "kernel.h"
#include "Shader.h"
#define BUFFER_OFFSET(i) ((char *)NULL + (i))


#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class Panel;
class Mouse;

class CudaLbm
{
private:
    int m_maxX;
    int m_maxY;
    Domain* m_domain;
    float* m_fA_d;
    float* m_fB_d;
    int* m_Im_d;
    float* m_FloorTemp_d;
    Obstruction* m_obst_d;
    Obstruction m_obst_h[MAXOBSTS];
    float m_inletVelocity;
    float m_omega;
    bool m_isPaused;
    int m_timeStepsPerFrame;
public:
    FW_API CudaLbm();
    FW_API CudaLbm(int maxX, int maxY);
    FW_API Domain* GetDomain();
    FW_API float* GetFA();
    FW_API float* GetFB();
    FW_API int* GetImage();
    FW_API float* GetFloorTemp();
    FW_API Obstruction* GetDeviceObst();
    FW_API Obstruction* GetHostObst();
    FW_API float GetInletVelocity();
    FW_API float GetOmega();
    FW_API void SetInletVelocity(float velocity);
    FW_API void SetOmega(float omega);
    FW_API void SetPausedState(bool isPaused);
    FW_API bool IsPaused();
    FW_API int GetTimeStepsPerFrame();
    FW_API void SetTimeStepsPerFrame(const int timeSteps);

    FW_API void AllocateDeviceMemory();
    FW_API void InitializeDeviceMemory();
    FW_API void DeallocateDeviceMemory();
    FW_API void UpdateDeviceImage();
    FW_API int ImageFcn(const int x, const int y);

   
};

class Graphics
{
    CudaLbm* m_cudaLbm;
    cudaGraphicsResource* m_cudaGraphicsResource;
    GLuint m_vao;
    GLuint m_vbo;
    GLuint m_elementArrayBuffer;
    ShaderProgram* m_shaderProgram;
    ShaderProgram* m_computeProgram;
public:
    FW_API Graphics();

    FW_API void CreateCudaLbm();
    FW_API CudaLbm* GetCudaLbm();
    FW_API cudaGraphicsResource* GetCudaSolutionGraphicsResource();
    FW_API GLuint GetVbo();
    FW_API GLuint GetElementArrayBuffer();
    FW_API void CreateVbo(unsigned int size, unsigned int vboResFlags);
    FW_API void DeleteVbo();
    FW_API void CreateElementArrayBuffer();
    FW_API void DeleteElementArrayBuffer();
    FW_API void CreateVboForCudaInterop(unsigned int size);
    FW_API void CleanUpGLInterOp();
    FW_API ShaderProgram* GetShaderProgram();
    FW_API ShaderProgram* GetComputeProgram();
    FW_API void CompileShaders();
    FW_API void RunComputeShader(const float3 cameraPosition);
    FW_API void RenderVbo(bool renderFloor, Domain &domain, glm::mat4 modelMatrix,
        glm::mat4 projectionMatrix);
    FW_API void RenderVboUsingShaders(bool renderFloor, Domain &domain, glm::mat4 modelMatrix,
        glm::mat4 projectionMatrix);
};



class GraphicsManager
{
private:
    float m_currentZ = -1000.f;
    //view transformations
    float3 m_rotate;
    float3 m_translate;
    int m_currentObstId = -1;
    float m_currentObstSize = 0.f;
    Obstruction::Shape m_currentObstShape = Obstruction::SQUARE;
    ViewMode m_viewMode;
    Obstruction* m_obstructions;
    Panel* m_parent;
    float m_scaleFactor = 1.f;
    GLint m_viewport[4];
    GLdouble m_modelMatrix[16];
    GLdouble m_projectionMatrix[16];
    float m_contourMinValue;
    float m_contourMaxValue;
    ContourVariable m_contourVar;
    Graphics* m_graphics;

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

    FW_API float GetContourMinValue();
    FW_API float GetContourMaxValue();
    FW_API void SetContourMinValue(const float contourMinValue);
    FW_API void SetContourMaxValue(const float contourMaxValue);
    FW_API ContourVariable GetContourVar();
    FW_API void SetContourVar(const ContourVariable contourVar);

    FW_API void SetObstructionsPointer(Obstruction* obst);

    FW_API float GetScaleFactor();
    FW_API void SetScaleFactor(const float scaleFactor);

    FW_API CudaLbm* GetCudaLbm();
    FW_API Graphics* GetGraphics();

    FW_API void CenterGraphicsViewToGraphicsPanel(const int leftPanelWidth);
    FW_API void SetUpGLInterop();
    FW_API void SetUpShaders();
    FW_API void SetUpCuda();
    FW_API void RunCuda();
    FW_API void RunComputeShader();
    FW_API void RenderVbo();
    FW_API void RenderVboUsingShaders();
    FW_API bool ShouldRenderFloor();

    FW_API void ClickDown(Mouse mouse);

    FW_API void Drag(const int xi, const int yi, const float dxf, const float dyf,
        const int button);
    FW_API void Wheel(const int button, const int dir, const int x, const int y);
    FW_API void Zoom(const int dir, const float mag);
    FW_API void Pan(const float dx, const float dy);
    FW_API void Rotate(const float dx, const float dy);
    FW_API int PickObstruction(const float mouseXf, const float mouseYf);
    FW_API int PickObstruction(const int mouseX, const int mouseY);
    FW_API void UnpickObstruction();
    FW_API void MoveObstruction(int obstId, const float mouseXf, const float mouseYf,
        const float dxf, const float dyf);
   
    FW_API void UpdateViewTransformations();
    FW_API void UpdateGraphicsInputs();
    FW_API glm::vec4 GetViewportMatrix();
    FW_API glm::mat4 GetModelMatrix();
    FW_API glm::mat4 GetProjectionMatrix();

    FW_API void GetSimCoordFromMouseCoord(int &xOut, int &yOut, const int mouseX, const int mouseY);
    FW_API void GetSimCoordFromFloatCoord(int &xOut, int &yOut, const float xf, const float yf);
    FW_API void GetMouseRay(float3 &rayOrigin, float3 &rayDir, const int mouseX, const int mouseY);
    FW_API int GetSimCoordFrom3DMouseClickOnObstruction(int &xOut, int &yOut, 
        const int mouseX, const int mouseY);
    FW_API void GetSimCoordFromMouseRay(int &xOut, int &yOut, const int mouseX, const int mouseY);
    FW_API void GetSimCoordFromMouseRay(int &xOut, int &yOut, const float mouseXf, const float mouseYf,
        const float planeZ);
    FW_API void GetSimCoordFromMouseRay(int &xOut, int &yOut, const int mouseX, const int mouseY,
        const float planeZ);
    FW_API void AddObstruction(const int simX, const int simY);
    FW_API void RemoveObstruction(const int simX, const int simY);
    FW_API void RemoveSpecifiedObstruction(const int obstId);
    FW_API void MoveObstruction(const int xi, const int yi, const float dxf, const float dyf);
    FW_API int FindUnusedObstructionId();
    FW_API int FindClosestObstructionId(const int simX, const int simY);
    FW_API int FindObstructionPointIsInside(const int x, const int y, const float tolerance=0.f);
    FW_API bool IsInClosestObstruction(const int mouseX, const int mouseY);
 
};

float GetDistanceBetweenTwoPoints(const float x1, const float y1,
    const float x2, const float y2);