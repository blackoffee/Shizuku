#pragma once 
#include <string>
#include <vector>


#ifdef LBM_GL_CPP_EXPORTS  
#define FW_API __declspec(dllexport)   
#else  
#define FW_API __declspec(dllimport)   
#endif  

class Button;

class FW_API ButtonGroup
{
    std::string m_name;
    std::vector<Button*> m_buttons;
public:
    ButtonGroup();
    ButtonGroup(const std::string name, std::vector<Button*> &buttons);

    std::string GetName();
    void AddButton(Button* button);
    std::vector<Button*> GetButtons(Button* button);
    void ExclusiveEnable(Button* button);
    Button* GetCurrentEnabledButton();
};