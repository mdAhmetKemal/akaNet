#ifndef NETGL_H
#define NETGL_H
#include <stdlib.h>
#include <stdio.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#define GLEW_STATIC 
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>  
typedef struct
{
	GLfloat x;
}pointData;
static uchar4 *d_out =0;
static uchar4 *other_out = 0;
static float*  trainErrorArray=0;
static float* testErrorArray = 0;
static float* MSErrorArray = 0;
static GLuint pbo;
static GLuint tex;
static struct cudaGraphicsResource *cuda_pbo_resource;
static int H;
static  int W;
static  int sayac=0;
typedef struct
{
	GLfloat x, y;
	GLfloat r, g, b, a;
}Vertex;

static void drawLine(Vertex a1, Vertex a2, GLfloat size){
	glLineWidth(size);
	glBegin(GL_LINES);
	glColor4f(a1.r, a1.g, a1.b, a1.a);
	glVertex2f(a1.x, a1.y);
	glColor4f(a2.r, a2.g, a2.b, a2.a);
	glVertex2f(a2.x, a2.y);
	glEnd();  
}
static void drawPoint(Vertex a1, GLfloat size){
	glPointSize(size);
	glBegin(GL_POINTS);
	glColor4f(a1.r, a1.g, a1.b, a1.a);
	glVertex2f(a1.x, a1.y);
	glEnd();
}
static float returnLine(float value,int W){
	return W - (value / 2)*W;
}
static float returnLog(float value, int W){
	value = -fabs(value)*10 + (W/2);
	//printf("\n mseError %.4f", value);
	//return 0,5*W-(20*(value/450));
	//return (W - (value / 200)*W)-W/2;
	return value;
}
static void plotDrawer(GLfloat thin){
	for (int s = 1; s < W; s++){
		if (testErrorArray[s + 1] != 0.){
			Vertex	P1 = { s, returnLine(testErrorArray[s - 1], W), 1.0, 0.0, 0.5, 0.7 };
			Vertex		P2 = { s + 1, returnLine(testErrorArray[s], W), 1.0, 0.0, 0.5, 0.7 };
			drawLine(P1, P2, thin);
			P1 = { s, returnLog(MSErrorArray[s - 1], W), 0.5, 0.3, 0.5, 0.7 };
			P2 = { s + 1, returnLog(MSErrorArray[s], W), 0.5, 0.3, 0.5, 0.7 };
			drawLine(P1, P2, thin);
		}
		if (trainErrorArray[s + 1] != 0.){
			Vertex P1 = { s, returnLine(trainErrorArray[s - 1], W), 0.5, 1.0, 0.5, 0.7 };
			Vertex P2 = { s + 1, returnLine(trainErrorArray[s], W), 0.5, 1.0, 0.5, 0.7 };
			drawLine(P1, P2, thin);
			//printf("\n point %.6f ", trainErrorArray[s - 1]);
		}
	}
}
static void layerDrawer(GLfloat thin){
	for (int w = 0; w < W; w++){
		for (int h = 0; h < (H /2); h++){
			float piksValueX = fminf(0.98,float(d_out[h*(W)+w].x) / 256);
			float piksValueY = fminf(0.98,float(d_out[h*(W)+w].y) / 256);
			float piksValueZ = fminf(0.98,float(d_out[h*(W)+w].z) / 256);
			Vertex P1 = { w, h, piksValueX, piksValueY, piksValueZ, 1.0 };
			drawPoint(P1, thin);
		}

	}
}
static void Background(GLfloat rgb){
	glColor3f(rgb, rgb, rgb);
	glBegin(GL_POLYGON);
	glVertex2f(0, 0);
	glVertex2f(0, H);
	glVertex2f(W, H);
	glVertex2f(W, 0);
	glEnd();

}
static void backGrid(int parca,GLfloat rgb,int thin){
	for (int g = 0; g < (parca*parca); g++){
		if ((g%parca) == 0){
			Vertex P1 = { 0, g *(W / (parca*parca)), rgb, rgb, rgb, 0.4 };
			Vertex P2 = { H, g *(W / (parca*parca)), rgb, rgb, rgb, 0.4 };
			drawLine(P1, P2, thin*1.5);
		}
		else{
			Vertex P1 = { 0, g *(W / (parca*parca)), rgb, rgb, rgb, 0.2 };
			Vertex P2 = { H, g *(W / (parca*parca)), rgb, rgb, rgb, 0.2 };
			drawLine(P1, P2, thin );
		}
		Vertex P1 = { 0, (W / 2), rgb, rgb, rgb, 0.4 };
		Vertex P2 = { H, (W / 2), rgb, rgb, rgb, 0.4 };
		drawLine(P1, P2, thin*1.8);
		if ((g%parca) == 0){
			Vertex P1 = { g *(H / (parca*parca)),0, rgb, rgb, rgb, 0.2 };
			Vertex P2 = { g *(H / (parca*parca)),W, rgb, rgb, rgb, 0.2 };
			drawLine(P1, P2, thin);
		}
		else{
			Vertex P1 = { g *(H / (parca*parca)), 0, rgb, rgb, rgb, 0.2 };
			Vertex P2 = { g *(H / (parca*parca)), W, rgb, rgb, rgb, 0.2 };
			drawLine(P1, P2, thin);
		}
	}
}
static void createVBO()
{
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, W * H *  sizeof(uchar4), 0, GL_STREAM_DRAW);
	
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,cudaGraphicsMapFlagsWriteDiscard);
	SDK_CHECK_ERROR_GL();
}
static void runCuda()
{
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));

	cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,cuda_pbo_resource);
	cudaMemcpy(d_out, other_out, W * H *  sizeof(uchar4), cudaMemcpyDeviceToDevice);
	//printf("uptade\n");
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
};

static void display()
{

	Background(0.6);
	backGrid(10, 0.5, 0.6);
	plotDrawer(.7);
	layerDrawer(1.);
	/* runCuda();
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
	glEnd();
	glDisable(GL_TEXTURE_2D); */
	glutSwapBuffers();
}

static void initGLUT(int He,int We) {
	
	H = He;
	W = We;

	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	int argc = 1;
	char * argv[] = { "" };
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(W, H);
	glutCreateWindow("aka");
	
	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//gluOrtho2D(0, W, H, 0);
	glOrtho(0, W,H, 0, -1.f, 1.f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glutDisplayFunc(display);
	glewInit();
	glDisable(GL_DEPTH_TEST);
	glClearColor(1., 0.5, 1., 0.);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	
	SDK_CHECK_ERROR_GL();
	
	//createVBO( );	
}
static void updateNetGL(){

	glutPostRedisplay(); 
	glutMainLoopEvent();
};


#endif