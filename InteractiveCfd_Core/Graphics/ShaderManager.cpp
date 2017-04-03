#include "ShaderManager.h"
#include "Shader.h"
#include "CudaLbm.h"
#include "Domain.h"
#include "helper_cuda.h"
#include <glm/gtc/type_ptr.hpp>

ShaderManager::ShaderManager()
{
    m_shaderProgram = new ShaderProgram;
    m_lightingProgram = new ShaderProgram;
    m_obstProgram = new ShaderProgram;
}

void ShaderManager::CreateCudaLbm()
{
    m_cudaLbm = new CudaLbm;
}

CudaLbm* ShaderManager::GetCudaLbm()
{
    return m_cudaLbm;
}

cudaGraphicsResource* ShaderManager::GetCudaSolutionGraphicsResource()
{
    return m_cudaGraphicsResource;
}

void ShaderManager::CreateVboForCudaInterop(const unsigned int size)
{
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
    CreateVbo(size, cudaGraphicsMapFlagsWriteDiscard);
    CreateElementArrayBuffer();
}

void ShaderManager::CleanUpGLInterOp()
{
    DeleteVbo();
    DeleteElementArrayBuffer();
}

void ShaderManager::CreateVbo(const unsigned int size, const unsigned int vboResFlags)
{
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);

    cudaGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, m_vbo, vboResFlags);
}

void ShaderManager::DeleteVbo()
{
    cudaGraphicsUnregisterResource(m_cudaGraphicsResource);
    glBindBuffer(1, m_vbo);
    glDeleteBuffers(1, &m_vbo);
    glDeleteVertexArrays(1, &m_vao);
    m_vbo = 0;
    m_vao = 0;
}

void ShaderManager::CreateElementArrayBuffer()
{
    const int numberOfElements = (MAX_XDIM - 1)*(MAX_YDIM - 1);
    const int numberOfNodes = MAX_XDIM*MAX_YDIM;
    GLuint* elementIndices = new GLuint[numberOfElements * 4 * 2];
    for (int j = 0; j < MAX_YDIM-1; j++){
        for (int i = 0; i < MAX_XDIM-1; i++){
            //going clockwise, since y orientation will be flipped when rendered
            elementIndices[j*(MAX_XDIM-1)*4+i*4+0] = (i)+(j)*MAX_XDIM;
            elementIndices[j*(MAX_XDIM-1)*4+i*4+1] = (i+1)+(j)*MAX_XDIM;
            elementIndices[j*(MAX_XDIM-1)*4+i*4+2] = (i+1)+(j+1)*MAX_XDIM;
            elementIndices[j*(MAX_XDIM-1)*4+i*4+3] = (i)+(j+1)*MAX_XDIM;
        }
    }
    for (int j = 0; j < MAX_YDIM-1; j++){
        for (int i = 0; i < MAX_XDIM-1; i++){
            //going clockwise, since y orientation will be flipped when rendered
            elementIndices[numberOfElements*4+j*(MAX_XDIM-1)*4+i*4+0] = numberOfNodes+(i)+(j)*MAX_XDIM;
            elementIndices[numberOfElements*4+j*(MAX_XDIM-1)*4+i*4+1] = numberOfNodes+(i+1)+(j)*MAX_XDIM;
            elementIndices[numberOfElements*4+j*(MAX_XDIM-1)*4+i*4+2] = numberOfNodes+(i+1)+(j+1)*MAX_XDIM;
            elementIndices[numberOfElements*4+j*(MAX_XDIM-1)*4+i*4+3] = numberOfNodes+(i)+(j+1)*MAX_XDIM;
        }
    }
    glGenBuffers(1, &m_elementArrayBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elementArrayBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*numberOfElements*4*2, elementIndices, GL_DYNAMIC_DRAW);
    free(elementIndices);
}

void ShaderManager::DeleteElementArrayBuffer(){
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &m_elementArrayBuffer);
}

template <typename T>
void ShaderManager::CreateShaderStorageBuffer(T defaultValue, const unsigned int numberOfElements, const std::string name)
{
    GLuint temp;
    glGenBuffers(1, &temp);
    T* data = new T[numberOfElements];
    for (int i = 0; i < numberOfElements; i++)
    {
        data[i] = defaultValue;
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, temp);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numberOfElements*sizeof(T), data, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, temp);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    m_ssbos.push_back(Ssbo{ temp, name });
}

GLuint ShaderManager::GetShaderStorageBuffer(const std::string name)
{
    for (std::vector<Ssbo>::iterator it = m_ssbos.begin(); it != m_ssbos.end(); ++it)
    {
        if (it->m_name == name)
        {
            return it->m_id;
        }
    }
}

GLuint ShaderManager::GetElementArrayBuffer()
{
    return m_elementArrayBuffer;
}

GLuint ShaderManager::GetVbo()
{
    return m_vbo;
}

ShaderProgram* ShaderManager::GetShaderProgram()
{
    return m_shaderProgram;
}

ShaderProgram* ShaderManager::GetLightingProgram()
{
    return m_lightingProgram;
}

ShaderProgram* ShaderManager::GetObstProgram()
{
    return m_obstProgram;
}

void ShaderManager::CompileShaders()
{
    GetShaderProgram()->Initialize();
    GetShaderProgram()->CreateShader("VertexShader.glsl", GL_VERTEX_SHADER);
    GetShaderProgram()->CreateShader("FragmentShader.glsl", GL_FRAGMENT_SHADER);
    GetLightingProgram()->Initialize();
    GetLightingProgram()->CreateShader("ComputeShader.glsl", GL_COMPUTE_SHADER);
    GetObstProgram()->Initialize();
    GetObstProgram()->CreateShader("Obstructions.glsl", GL_COMPUTE_SHADER);
}

void ShaderManager::AllocateStorageBuffers()
{
    CreateShaderStorageBuffer(GLfloat(0), MAX_XDIM*MAX_YDIM*9, "LbmA");
    CreateShaderStorageBuffer(GLfloat(0), MAX_XDIM*MAX_YDIM*9, "LbmB");
    CreateShaderStorageBuffer(GLint(0), MAX_XDIM*MAX_YDIM, "Floor");
    CreateShaderStorageBuffer(Obstruction{}, MAXOBSTS, "Obstructions");
    CreateShaderStorageBuffer(float4{0,0,0,1e6}, 1, "RayIntersection");
}

void ShaderManager::InitializeObstSsbo()
{
    const GLuint obstSsbo = GetShaderStorageBuffer("Obstructions");
    CudaLbm* cudaLbm = GetCudaLbm();
    Obstruction* obst_h = cudaLbm->GetHostObst();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, obstSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, MAXOBSTS*sizeof(Obstruction), obst_h, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, obstSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

int ShaderManager::RayCastMouseClick(float3 &rayCastIntersection, const float3 rayOrigin,
    const float3 rayDir)
{
    Domain domain = *m_cudaLbm->GetDomain();
    int xDim = domain.GetXDim();
    int yDim = domain.GetYDim();
    glm::vec4 intersectionCoord{ 0, 0, 0, 0 };
    float intersectCoord[4];
    const GLuint ssbo_rayIntersection = GetShaderStorageBuffer("RayIntersection");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssbo_rayIntersection);

    ShaderProgram* const shader = GetLightingProgram();
    shader->Use();

    GLuint shaderID = shader->GetId();
    SetUniform(shaderID, "maxXDim", MAX_XDIM);
    SetUniform(shaderID, "maxyDim", MAX_YDIM);
    SetUniform(shaderID, "maxObsts", MAXOBSTS);
    SetUniform(shaderID, "xDim", xDim);
    SetUniform(shaderID, "yDim", yDim);
    SetUniform(shaderID, "xDimVisible", domain.GetXDimVisible());
    SetUniform(shaderID, "yDimVisible", domain.GetYDimVisible());
    SetUniform(shaderID, "rayOrigin", rayOrigin);
    SetUniform(shaderID, "rayDir", rayDir);

    RunSubroutine(shaderID, "ResetRayCastData", int3{ 1, 1, 1 });
    RunSubroutine(shaderID, "RayCast", int3{ xDim, yDim, 1 });

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_rayIntersection);
    GLfloat* intersect = (GLfloat*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
        sizeof(glm::vec4), GL_MAP_READ_BIT);
    std::memcpy(&intersectionCoord, intersect, sizeof(glm::vec4));
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    int returnVal;
    if (intersectionCoord.w > 1e5) //ray did not intersect with any objects
    {
        returnVal = 1;
    }
    else
    {
        RunSubroutine(shaderID, "ResetRayCastData", int3{ 1, 1, 1 });
        rayCastIntersection.x = intersectionCoord.x;
        rayCastIntersection.y = intersectionCoord.y;
        rayCastIntersection.z = intersectionCoord.z;
        returnVal = 0;
    }
    shader->Unset();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return returnVal;
}

void ShaderManager::RenderVbo(const bool renderFloor, Domain &domain, const glm::mat4 &modelMatrix,
    const glm::mat4 &projectionMatrix)
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBindVertexArray(m_vao);
    //Draw solution field
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elementArrayBuffer);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 16, (GLvoid*)(3 * sizeof(GLfloat)));
    glEnableClientState(GL_VERTEX_ARRAY);

    int yDimVisible = domain.GetYDimVisible();
    if (renderFloor)
    {
        //Draw floor
        glDrawElements(GL_QUADS, (MAX_XDIM - 1)*(yDimVisible - 1)*4, GL_UNSIGNED_INT, 
            BUFFER_OFFSET(sizeof(GLuint)*4*(MAX_XDIM - 1)*(MAX_YDIM - 1)));
    }
    //Draw water surface
    glDrawElements(GL_QUADS, (MAX_XDIM - 1)*(yDimVisible - 1)*4 , GL_UNSIGNED_INT, (GLvoid*)0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindVertexArray(0);
}

void ShaderManager::RunComputeShader(const float3 cameraPosition)
{
    const GLuint ssbo_lbmA = GetShaderStorageBuffer("LbmA");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_lbmA);
    const GLuint ssbo_lbmB = GetShaderStorageBuffer("LbmB");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_lbmB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_vbo);
    const GLuint ssbo_floor = GetShaderStorageBuffer("Floor");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssbo_floor);
    const GLuint ssbo_obsts = GetShaderStorageBuffer("Obstructions");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, ssbo_obsts);
    ShaderProgram* const shader = GetLightingProgram();

    shader->Use();

    Domain domain = *m_cudaLbm->GetDomain();
    const int xDim = domain.GetXDim();
    const int yDim = domain.GetYDim();
    GLuint shaderID = shader->GetId();
    SetUniform(shaderID, "maxXDim", MAX_XDIM);
    SetUniform(shaderID, "maxyDim", MAX_YDIM);
    SetUniform(shaderID, "maxObsts", MAXOBSTS);
    SetUniform(shaderID, "xDim", xDim);
    SetUniform(shaderID, "yDim", yDim);
    SetUniform(shaderID, "xDimVisible", domain.GetXDimVisible());
    SetUniform(shaderID, "yDimVisible", domain.GetYDimVisible());
    SetUniform(shaderID, "cameraPosition", cameraPosition);
    SetUniform(shaderID, "uMax", m_inletVelocity);
    SetUniform(shaderID, "omega", m_omega);

    for (int i = 0; i < 5; i++)
    {
        RunSubroutine(shaderID, "MarchLbm", int3{ xDim, yDim, 1 });
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_lbmA);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_lbmB);

        RunSubroutine(shaderID, "MarchLbm", int3{ xDim, yDim, 1 });
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_lbmA);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_lbmB);
    }
    
    RunSubroutine(shaderID, "UpdateFluidVbo", int3{ xDim, yDim, 1 });
    RunSubroutine(shaderID, "DeformFloorMeshUsingCausticRay", int3{ xDim, yDim, 1 });
    RunSubroutine(shaderID, "ComputeFloorLightIntensitiesFromMeshDeformation", int3{ xDim, yDim, 1 });
    RunSubroutine(shaderID, "ApplyCausticLightingToFloor", int3{ xDim, yDim, 1 });
    RunSubroutine(shaderID, "PhongLighting", int3{ xDim, yDim, 2 });
    RunSubroutine(shaderID, "UpdateObstructionTransientStates", int3{ xDim, yDim, 1 });
    RunSubroutine(shaderID, "CleanUpVbo", int3{ MAX_XDIM, MAX_YDIM, 2 });
    
    shader->Unset();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}



void ShaderManager::UpdateObstructionsUsingComputeShader(const int obstId, Obstruction &newObst)
{
    const GLuint ssbo_obsts = GetShaderStorageBuffer("Obstructions");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_obsts);
    ShaderProgram* const shader = GetObstProgram();
    shader->Use();

    GLuint shaderId = shader->GetId();

    SetUniform(shaderId, "targetObst.shape", newObst.shape);
    SetUniform(shaderId, "targetObst.x", newObst.x);
    SetUniform(shaderId, "targetObst.y", newObst.y);
    SetUniform(shaderId, "targetObst.r1", newObst.r1);
    SetUniform(shaderId, "targetObst.u", newObst.u);
    SetUniform(shaderId, "targetObst.v", newObst.v);
    SetUniform(shaderId, "targetObst.state", newObst.state);
    SetUniform(shaderId, "targetObstId", obstId);

    RunSubroutine(shader->GetId(), "UpdateObstruction", int3{ 1, 1, 1 });

    shader->Unset();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ShaderManager::InitializeComputeShaderData()
{
    const GLuint ssbo_lbmA = GetShaderStorageBuffer("LbmA");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_lbmA);
    const GLuint ssbo_lbmB = GetShaderStorageBuffer("LbmB");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_lbmB);
    const GLuint ssbo_obsts = GetShaderStorageBuffer("Obstructions");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, ssbo_obsts);
    ShaderProgram* const shader = GetLightingProgram();

    shader->Use();

    Domain domain = *m_cudaLbm->GetDomain();
    GLuint shaderId = shader->GetId();
    SetUniform(shaderId, "maxXDim", MAX_XDIM);
    SetUniform(shaderId, "maxYDim", MAX_YDIM);
    SetUniform(shaderId, "maxObsts", MAXOBSTS);//
    SetUniform(shaderId, "xDim", domain.GetXDim());
    SetUniform(shaderId, "yDim", domain.GetYDim());
    SetUniform(shaderId, "xDimVisible", domain.GetXDim());
    SetUniform(shaderId, "yDimVisible", domain.GetYDim());
    SetUniform(shaderId, "uMax", m_inletVelocity);

    RunSubroutine(shaderId, "InitializeDomain", int3{ MAX_XDIM, MAX_YDIM, 1 });

    shader->Unset();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}
    
void ShaderManager::SetOmega(const float omega)
{
    m_omega = omega;
}

float ShaderManager::GetOmega()
{
    return m_omega;
}

void ShaderManager::SetInletVelocity(const float u)
{
    m_inletVelocity = u;
}

float ShaderManager::GetInletVelocity()
{
    return m_inletVelocity;
}

void ShaderManager::UpdateLbmInputs(const float u, const float omega)
{
    SetInletVelocity(u);
    SetOmega(omega);
}

void ShaderManager::RenderVboUsingShaders(const bool renderFloor, Domain &domain,
    const glm::mat4 &modelMatrix, const glm::mat4 &projectionMatrix)
{
    ShaderProgram* shader = GetShaderProgram();

    shader->Use();//
    GLint modelMatrixLocation = glGetUniformLocation(shader->GetId(), "modelMatrix");
    GLint projectionMatrixLocation = glGetUniformLocation(shader->GetId(), "projectionMatrix");
    glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    RenderVbo(renderFloor, domain, modelMatrix, projectionMatrix);

    shader->Unset();
    glBindVertexArray(0);
}

void SetUniform(GLuint shaderId, const GLchar* varName, const int varValue)
{
    const GLint targetLocation = glGetUniformLocation(shaderId, varName);
    glUniform1i(targetLocation, varValue);
}

void SetUniform(GLuint shaderId, const GLchar* varName, const float varValue)
{
    const GLint targetLocation = glGetUniformLocation(shaderId, varName);
    glUniform1f(targetLocation, varValue);
}

void SetUniform(GLuint shaderId, const GLchar* varName, const bool varValue)
{
    const GLint targetLocation = glGetUniformLocation(shaderId, varName);
    glUniform1i(targetLocation, varValue);
}

void SetUniform(GLuint shaderId, const GLchar* varName, const float3 varValue)
{
    const GLint targetLocation = glGetUniformLocation(shaderId, varName);
    glUniform3f(targetLocation, varValue.x, varValue.y, varValue.z);
}

void RunSubroutine(GLuint shaderId, const GLchar* subroutineName, const int3 workGroupSize)
{
    const GLuint subroutine = glGetSubroutineIndex(shaderId, GL_COMPUTE_SHADER,
        subroutineName);
    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &subroutine);
    glDispatchCompute(workGroupSize.x, workGroupSize.y, workGroupSize.z);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

