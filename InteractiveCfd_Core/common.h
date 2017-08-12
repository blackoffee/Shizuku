#pragma once
#define MAXOBSTS 100
#define MAX_XDIM 512
#define MAX_YDIM 512

#define INITIAL_UMAX 0.125f
#define BLOCKSIZEX 128
#define BLOCKSIZEY 1
#define REFRESH_DELAY 10
#define LINE_OBST_WIDTH 1
#define TIMESTEPS_PER_FRAME 30
#define PI 3.141592653589793238463
#define SMAG_CONST 1.f

enum ContourVariable{VEL_MAG,VEL_U,VEL_V,PRESSURE,STRAIN_RATE,WATER_RENDERING};
enum ViewMode{TWO_DIMENSIONAL,THREE_DIMENSIONAL};
enum Shape{SQUARE=0,CIRCLE=1,HORIZONTAL_LINE=2,VERTICAL_LINE=3};
enum State{ACTIVE=0,INACTIVE=1,NEW=2,REMOVED=3};

struct Obstruction
{
    int shape;
    float x;
    float y;
    float r1;
    float r2;
    float u;
    float v;
    int state;
};