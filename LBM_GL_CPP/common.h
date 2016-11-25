#pragma once

#define MAXOBSTS 100
#define XDIM 512
#define YDIM 384
#define UMAX 0.06f
#define BLOCKSIZEX 64
#define BLOCKSIZEY 1
#define REFRESH_DELAY 10

enum ContourVariable{VEL_U,VEL_V,PRESSURE,STRAIN_RATE};

struct Obstruction
{
	enum Shape{SQUARE,CIRCLE,HORIZONTAL_LINE,VERTICAL_LINE};
	Shape shape;
	float x;
	float y;
	float r1;
	float r2;
};