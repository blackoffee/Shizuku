#version 430 core
struct Obstruction
{
    int shape; // {SQUARE,CIRCLE,HORIZONTAL_LINE,VERTICAL_LINE};
    float x;
    float y;
    float r1;
    float r2;
    float u;
    float v;
    int state; // {ACTIVE,INACTIVE,NEW,REMOVED};
};
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) buffer ssbo_obsts
{
    Obstruction obsts[];
};

uniform int maxObsts;
uniform int targetObstId;
uniform Obstruction targetObst;

subroutine void ObstUpdate_t();

subroutine uniform ObstUpdate_t ObstUpdate;

subroutine(ObstUpdate_t) void UpdateObstruction()
{
    obsts[targetObstId].shape = targetObst.shape;
    obsts[targetObstId].r1 = targetObst.r1;
    obsts[targetObstId].x = targetObst.x;
    obsts[targetObstId].y = targetObst.y;
    obsts[targetObstId].u = targetObst.u;
    obsts[targetObstId].v = targetObst.v;
    obsts[targetObstId].state = targetObst.state;
}


subroutine(ObstUpdate_t) void DoNothing2()
{

}

void main()
{
    ObstUpdate();

}