#pragma once

#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class Panel;
class Button;
class SliderBar;
class GraphicsManager;

class FW_API Command
{
protected:
    Panel* m_rootPanel;
    enum State {ACTIVE, INACTIVE};
    State m_state;
    Command();
    Command(Panel &rootPanel);
    void Start();
    void Track();
    void End();
    Panel* GetRootPanel();
    GraphicsManager* GetGraphicsManager();
};
