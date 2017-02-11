#pragma once

#define MAXOBSTS 100
#define MAX_XDIM 512
#define MAX_YDIM 512
#define UMAX 0.06f
#define BLOCKSIZEX 128
#define BLOCKSIZEY 1
#define REFRESH_DELAY 10
#define LINE_OBST_WIDTH 1
#define PI 3.141592653589793238463

enum ContourVariable{VEL_MAG,VEL_U,VEL_V,PRESSURE,STRAIN_RATE,WATER_RENDERING};
enum ViewMode{TWO_DIMENSIONAL,THREE_DIMENSIONAL};

struct Obstruction
{
    enum Shape{SQUARE,CIRCLE,HORIZONTAL_LINE,VERTICAL_LINE};
    enum State{ACTIVE,INACTIVE,NEW,REMOVED};
    //Obstruction() : state(NEW) {}
    Shape shape;
    float x;
    float y;
    float r1;
    float r2;
    float u;
    float v;
    State state;
};