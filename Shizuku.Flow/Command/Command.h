#pragma once

#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class GraphicsManager;

class FW_API Command
{
private:
    GraphicsManager* m_graphics;
protected:
    enum State {ACTIVE, INACTIVE};
    State m_state;
    Command();
    Command(GraphicsManager &graphicsManager);
    void Start();
    void Track();
    void End();
    GraphicsManager* GetGraphicsManager();
};
