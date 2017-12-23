#ifndef HYPERTENSOR_CPP
#define HYPERTENSOR_CPP

#include "HyperTensor.h"
#include <algorithm>
#include <cuda_runtime.h>

void HyperTensor::changeHyperTensor(int sam, int fea, int he, int wi, int ch, float fac){
	axis = 5;
	if (sam == 0) sam++; if (sam == 1) axis--;
	if (fea == 0) fea++; if (fea == 1) axis--;
	if (he == 0) he++; if (he == 1) axis--;
	if (wi == 0) wi++; if (wi == 1) axis--;
	if (ch == 0) ch++; if (ch == 1) axis--;
	hTsample = sam;
	hTfeatureNum = fea;
	hTheigth = he;
	hTwidth = wi;
	hTchanal = ch;
	sizePiksel = sam*fea*he*wi*ch;
	freeCu();
	allocateCu();
	if (fac != 0.0)
		TensorRandom(fac);
	else
		setZero();
}

void HyperTensor::refreshTensor(float *newTensor){
	for (int r = 0; r < sizePiksel; ++r){
		paralelTensor[r] = Tensor[r];
		Tensor[r] = newTensor[r];
	}
	cudaError_t target = cudaMemcpy(paralelTensor, Tensor, sizeof(float) * sizePiksel, cudaMemcpyDeviceToDevice);
	cudaError_t target2 = cudaMemcpy(Tensor, newTensor, sizeof(float) * sizePiksel, cudaMemcpyDeviceToDevice);
	if (target != cudaSuccess || target2 != cudaSuccess){
		printf("\nSorun:Hyperdata::refreshTensor");
	}
	else{
		//printf("\nloadTensor pieces : %.6f ", newTensor[1]);
	}
}
void HyperTensor::loadTensor(float *newTensor){
	
	cudaError_t target = cudaMemcpy(Tensor,newTensor, sizeof(float) * sizePiksel, cudaMemcpyHostToDevice);
	if (target != cudaSuccess){
		printf("\nSorun:Hyperdata::load2Tensorfirst");
	}
	else{
		//printf("\nBasarili:Hyperdata::loadTensor");
		//printf("\nloadTensor pieces : %.6f ", newTensor[1]);
	}
}
void HyperTensor::loadTensorFromGPU(float *newTensor){

	cudaError_t target = cudaMemcpy(Tensor, newTensor, sizeof(float) * sizePiksel, cudaMemcpyDeviceToDevice);
	if (target != cudaSuccess){
		printf("\nSorun:Hyperdata::load2TensorFromGPU");
	}
	else{
		//printf("\nBasarili:Hyperdata::loadTensorFromGPU");
		//printf("\nloadTensor pieces : %.6f ", newTensor[1]);
	}
}
void HyperTensor::addPastTensor(){
	cudaError_t target = cudaMalloc(&pastTensor, sizePiksel * sizeof(float));
	if ( target != cudaSuccess){
		pastTensor = 0;
		printf("\nSorun:Hyperdata::addPastTensor");
	}
	else{ //printf("\nBasarili:Hyperdata::addPastTensor");
	}
	isAddPastTensor = true;
	//randTensor(pastTensor, sizePiksel, 0.001f);
}
void HyperTensor::TensorRandom(float fac = 0.0001f){
	//printf("\nDe:randtensor fac:%.4f", fac);
	randTensor(Tensor, sizePiksel,fac);
	/*randTensor(paralelTensor, sizePiksel, 0.001f);
	if (isAddPastTensor)
		randTensor(pastTensor, sizePiksel, 0.001f); */

};
HyperTensor::HyperTensor(int sam = 1, int fea = 1, int he = 5,
	int wi = 5, int ch = 6, float fac = 0.01f){
	axis = 5;
	if (sam == 0) sam++; if (sam == 1) axis--;
	if (fea == 0) fea++; if (fea == 1) axis--;
	if (he == 0) he++; if (he == 1) axis--;
	if (wi == 0) wi++; if (wi == 1) axis--;
	if (ch == 0) ch++; if (ch == 1) axis--;
	hTsample = sam;
	hTfeatureNum = fea;
	hTheigth = he;
	hTwidth = wi;
	hTchanal = ch;
	sizePiksel = sam*fea*he*wi*ch;
	isAddPastTensor = false;
	allocateCu();
	setZero();
}
HyperTensor::~HyperTensor(){
	freeCu();
};

void HyperTensor::allocateCu(){
	cudaError_t target = cudaMalloc(&Tensor, sizePiksel * sizeof(float));
	cudaError_t target2 = cudaMalloc(&paralelTensor, sizePiksel * sizeof(float));
	if (isAddPastTensor)
		 cudaMalloc(&pastTensor, sizePiksel * sizeof(float));
	
	if (target != cudaSuccess && target2 != cudaSuccess)
	{
		Tensor = paralelTensor = 0;
		printf("\nSorun:Hyperdata::allocateCu");
	}
	else{ 
		// printf("\nBasarili:Hyperdata::allocateCu");
	}
}
void HyperTensor::freeCu(){
	if (Tensor != 0)
	{
		cudaFree(Tensor);
		cudaFree(paralelTensor);
		if (isAddPastTensor)
			cudaFree(pastTensor);
	}
}
void HyperTensor::setZero(){
	cudaError_t target = cudaMemset(Tensor,0, sizePiksel * sizeof(float));
	cudaError_t target2 = cudaMemset(paralelTensor,0, sizePiksel * sizeof(float));
	if (target != cudaSuccess && target2 != cudaSuccess)
	{
		Tensor = paralelTensor = 0;
		printf("\nSorun:Hyperdata::setZero");
	}
	else{ 
		//printf("\nBasarili:Hyperdata::setZero");
	}
}
#endif