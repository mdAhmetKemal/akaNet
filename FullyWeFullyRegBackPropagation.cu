#include <cuda_runtime.h>
#include "ConnectCuda.h"

void FullyWeigthFullyRegBackProCu(
	float* inputWeLayerTensor, int inpLaySamp, int inpLayFea,
	int inpLayHe, int inpLayWi, int inpLayCh,
	float * regulatorLayerTensor, int regLayFea, int regLayHe,
	int regLayWi, int regLayCh,
	float * outLayerTensor, int outLayFea, int outLayHe,
	int outLayWi, int outLayCh,
	float* kernelWeigthTensor,float* kernelWePastTensor,
	float* kernelActivatorTensor){

};

