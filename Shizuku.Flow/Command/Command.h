#pragma once

#ifdef SHIZUKU_FLOW_EXPORTS  
#define FLOW_API __declspec(dllexport)   
#else  
#define FLOW_API __declspec(dllimport)   
#endif  

class GraphicsManager;

class FLOW_API Command
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
