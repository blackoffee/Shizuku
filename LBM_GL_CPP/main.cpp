#include <GL/glew.h>
#include <GL/freeglut.h>
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_cuda_gl.h"

#include <stdio.h>
#include <iostream>
#include <ostream>
#include <fstream>

#include "kernel.h"
#include "Mouse.h"
#include "Panel.h"
#include "common.h"


int winw = 640;
int winh = 480;
int g_xDim = XDIM;
int g_yDim = YDIM;
float rotate_x = 45.f;

Obstruction g_obstructions[MAXOBSTS];

Panel theWindow;
Mouse theMouse;

GLuint g_vboSolutionField;
GLuint g_elementArrayIndexBuffer;
cudaGraphicsResource *g_cudaSolutionField;

int* g_elementArrayIndices;

float* g_fA_h;
float* g_fA_d;
float* g_fB_h;
float* g_fB_d;
int* g_im_h;
int* g_im_d;
Obstruction* g_obst_d;




void Init()
{
	glEnable(GL_LIGHT0);

	glewInit();
}

void SetUpWindow()
{
	int leftPanelWidth(100);
	theWindow.m_rectInt_abs = RectInt(100, 100, winw, winh);
	theWindow.RectIntAbsToRectFloatAbs();

	theWindow.CreateSubPanel(RectInt(0, 0, 100, winh), Panel::DEF_ABS, "CDV", Color(Color::LIGHT_GRAY));
	theWindow.m_subPanels[0]->CreateButton(RectFloat(-0.9f, 0.f, 1.8f, 0.2f), Panel::DEF_REL, "Button1", Color(Color::RED));
	theWindow.m_subPanels[0]->CreateButton(RectFloat(-0.9f,-0.5f, 1.8f, 0.2f), Panel::DEF_REL, "Button2", Color(Color::DARK_GRAY));

}


/*----------------------------------------------------------------------------------------
 *	GL Interop Functions
 */
void CreateVBO(GLuint *vbo, cudaGraphicsResource **vbo_res, unsigned int size, unsigned int vbo_res_flags)
{
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);
}

void DeleteVBO(GLuint *vbo, cudaGraphicsResource *vbo_res)
{
	cudaGraphicsUnregisterResource(vbo_res);
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}


void GenerateIndexList(){

	g_elementArrayIndices = new int[(g_xDim-1)*(g_yDim-1) * 4];
	for (int j = 0; j < g_yDim-1; j++){
		for (int i = 0; i < g_xDim-1; i++){
			//going clockwise, since y orientation will be flipped when rendered
			g_elementArrayIndices[j*(g_xDim-1)*4+i * 4 + 0] = (i)+(j)*g_xDim;
			g_elementArrayIndices[j*(g_xDim-1)*4+i * 4 + 1] = (i + 1) + (j)*g_xDim;
			g_elementArrayIndices[j*(g_xDim-1)*4+i * 4 + 2] = (i+1)+(j + 1)*g_xDim;
			g_elementArrayIndices[j*(g_xDim-1)*4+i * 4 + 3] = (i)+(j + 1)*g_xDim;
		}
	}

	glGenBuffers(1, &g_elementArrayIndexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_elementArrayIndexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*(g_xDim-1)*(g_yDim-1)*4, g_elementArrayIndices, GL_DYNAMIC_DRAW);
}

void CleanUpIndexList(){
	free(g_elementArrayIndices);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &g_elementArrayIndexBuffer);
}


void SetUpGLInterop()
{
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	GenerateIndexList();
	unsigned int solutionMemorySize = g_xDim*g_yDim * 4 * sizeof(float);
	CreateVBO(&g_vboSolutionField, &g_cudaSolutionField, solutionMemorySize, cudaGraphicsMapFlagsWriteDiscard);
}

void CleanUpGLInterop()
{
	CleanUpIndexList();
	DeleteVBO(&g_vboSolutionField, g_cudaSolutionField);
}

/*----------------------------------------------------------------------------------------
 *	CUDA calls
 */

//BC function for host side
int ImageFcn_h(int x, int y, Obstruction* obstructions){
	//if(y == 0 || x == XDIM-1 || y == YDIM-1)
	if (x < 0.1f)
	return 3;//west
	else if ((XDIM - x) < 1.1f)
	return 2;//east
	else if ((YDIM - y) < 1.1f)
	return 11;//11;//xsymmetry top
	else if (y < 0.1f)
	return 12;//12;//xsymmetry bottom
	return 0;
}

void SetUpCUDA()
{
	size_t memsize, memsize_int, memsize_inputs;

	int domainSize = ((g_xDim + BLOCKSIZEX - 1) / BLOCKSIZEX)*(g_yDim / BLOCKSIZEY)
						*BLOCKSIZEX*BLOCKSIZEY;
	memsize = domainSize*sizeof(float)*9;
	memsize_int = domainSize*sizeof(int);
	memsize_inputs = sizeof(g_obstructions);

	g_fA_h = (float *)malloc(memsize);
	g_fB_h = (float *)malloc(memsize);
	g_im_h = (int *)malloc(memsize_int);
	//obstructions = (input_values *)malloc(memsize_inputs);

	cudaMalloc((void **)&g_fA_d, memsize);
	cudaMalloc((void **)&g_fB_d, memsize);
	cudaMalloc((void **)&g_im_d, memsize_int);
	cudaMalloc((void **)&g_obst_d, memsize_inputs);

	for (int i = 0; i < domainSize*9; i++)
	{
		g_fA_h[i] = 0;
		g_fB_h[i] = 0;
	}
	//u_max = UMAX;
	for (int i = 0; i < MAXOBSTS; i++)
	{
		g_obstructions[i].r1 = 0;
		g_obstructions[i].x = 0;
		g_obstructions[i].y = -1000;
	}	
	g_obstructions[0].r1 = 10;
	g_obstructions[0].x = XDIM*0.33f;
	g_obstructions[0].y = YDIM*0.5f;
	g_obstructions[0].shape = Obstruction::CIRCLE;

	for (int i = 0; i < domainSize; i++)
	{
		int x = i%XDIM;
		int y = i/XDIM;
		g_im_h[i] = ImageFcn_h(x, y, g_obstructions);
	}
	
	cudaMemcpy(g_fA_d, g_fA_h, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(g_fB_d, g_fB_h, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(g_im_d, g_im_h, memsize_int, cudaMemcpyHostToDevice);
	cudaMemcpy(g_obst_d, g_obstructions, memsize_inputs, cudaMemcpyHostToDevice);

	//writeInputs();
	InitializeDomain(g_fA_d, g_im_d, g_xDim, g_yDim);
	InitializeDomain(g_fB_d, g_im_d, g_xDim, g_yDim);
}

void RunCuda(struct cudaGraphicsResource **vbo_resource)
{
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	cudaGraphicsMapResources(1, vbo_resource, 0);
	size_t num_bytes,num_bytes2;
	cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, *vbo_resource);

	MarchSolution(dptr, g_fA_d, g_fB_d, g_im_d, g_obst_d, ContourVariable::VEL_U, g_xDim, g_yDim, 5);

	// unmap buffer object
	cudaGraphicsUnmapResources(1, vbo_resource, 0);
}



/*----------------------------------------------------------------------------------------
 *	This function will be used to draw the 3D scene
 */
void Draw3D()
{
	gluLookAt(0,1,5,0,0,0,0,1,0);
	glutSolidTeapot(1);
}

void DrawRectangle(RectInt rect)
{
	glBegin(GL_QUADS);
		glVertex2f(rect.m_x         ,rect.m_y+rect.m_h);
		glVertex2f(rect.m_x         ,rect.m_y         );
		glVertex2f(rect.m_x+rect.m_w,rect.m_y         );
		glVertex2f(rect.m_x+rect.m_w,rect.m_y+rect.m_h);
	glEnd();
}


void Draw2D()
{
	theWindow.m_subPanels[0]->Draw();
	theWindow.m_subPanels[0]->m_buttons[0]->Draw();
	theWindow.m_subPanels[0]->m_buttons[1]->Draw();
}


void MouseButton(int button, int state, int x, int y)
{
	theMouse.Update(x, y, button, state);

}

void MouseMotion(int x, int y)
{
	int dx, dy;

	theMouse.GetChange(x, y);

	theMouse.Update(x, y);

}



void Draw()
{
	RunCuda(&g_cudaSolutionField);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);

	/*
	 *	Set perspective viewing transformation
	 */
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//gluPerspective(45,(winh==0)?(1):((float)winw/winh),1,100);
	glOrtho(-10,10,-10,10,0,100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(0.0, 0.0, -1.0);
	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, g_vboSolutionField);
	//glVertexPointer(4, GL_FLOAT, 0, 0);
	glVertexPointer(3, GL_FLOAT, 16, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(4, GL_UNSIGNED_BYTE, 16, (char *)NULL + 12);
	//glDrawArrays(GL_POINTS, 0, XDIM * YDIM);
	glDrawElements(GL_QUADS, (XDIM - 1)*(YDIM - 1) * 4, GL_UNSIGNED_INT, (GLvoid*)0);
	glDisableClientState(GL_VERTEX_ARRAY);

	//glRotatef(rotate_x,1.f,0.f,0.f);

	/*
	 *	Draw the 3D elements in the scene
	 */
	//Draw3D();

	/*
	 *	Disable depth test and lighting for 2D elements
	 */
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	/*
	 *	Set the orthographic viewing transformation
	 */
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//glOrtho(0,winw,winh,0,-1,1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	/*
	 *	Draw the 2D overlay
	 */
	Draw2D();

	/*
	 *	Bring the back buffer to the front and vice-versa.
	 */
	glutSwapBuffers();

}

int main(int argc,char **argv)
{



	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_RGB|GLUT_DEPTH|GLUT_DOUBLE);
	glutInitWindowSize(winw,winh);
	glutInitWindowPosition(200,100);
	glutCreateWindow("Interactive CFD");

	glutDisplayFunc(Draw);
//	glutReshapeFunc(Resize);
	glutMouseFunc(MouseButton);
	glutMotionFunc(MouseMotion);
//	glutPassiveMotionFunc(MousePassiveMotion);

	Init();

	SetUpWindow();

	SetUpGLInterop();

	SetUpCUDA();

	glutMainLoop();

	return 0;
}