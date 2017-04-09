#include "Panel.h"

class Slider;

class FW_API SliderBar : public Panel
{
public:
    enum Orientation {VERTICAL, HORIZONTAL};
private:
    float m_value = 0.5f;
    Orientation m_orientation = HORIZONTAL;
public:
    SliderBar();
    SliderBar(const RectFloat rectFloat, const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Slider* parent = NULL);
    SliderBar(const RectInt rectInt    , const SizeDefinitionMethod sizeDefinition,
        const std::string name, const Color color, Slider* parent = NULL);

    void Draw();
    void UpdateValue();
    float GetValue();
    Orientation GetOrientation();
    virtual void Drag(int x, int y, float dx, float dy, int button);
};


