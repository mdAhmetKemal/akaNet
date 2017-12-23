#ifndef HYPERTENSOR_H
#define HYPERTENSOR_H
#include <stdlib.h>
#include <stdexcept>
extern void  randTensor(float* data, int size,float fac);
class HyperTensor{
public:
	HyperTensor(int sampl, int fea, int he, int wi, int ch, float fac);
	~HyperTensor();
	float *Tensor;
	float *paralelTensor;
	float *pastTensor;
	int axis;
	int hTsample;
	int hTfeatureNum;
	int hTheigth;
	int hTwidth;
	int hTchanal;
	void refreshTensor(float *newTensor);
	void loadTensor(float *newTensor);
	void loadTensorFromGPU(float* newTensor);
	int sizePiksel;
	void changeHyperTensor(int sam, int fea, int he, int wi, int ch, float fac=0.0);
	void TensorRandom(float factor);
	void addPastTensor();
	void setZero();
 private:
	 bool isAddPastTensor;
	 void allocateCu();
	 void freeCu();
};

#endif




