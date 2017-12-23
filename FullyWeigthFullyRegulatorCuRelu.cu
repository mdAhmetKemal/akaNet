#include <cuda_runtime.h>
#include "ConnectCuda.h"
#include <cuda.h>
#include<math.h>

	


void FullyWeigthFullyRegulatorCuRelu(
	float* inputWeLayerTensor, int inpLaySamp, int inpLayFea,
	int inpLayHe, int inpLayWi, int inpLayCh,
	float * regulatorLayerTensor, int regLayFea, int regLayHe,
	int regLayWi, int regLayCh,
	float * outLayerTensor, int outLayFe, int outLayHe,
	int outLayWi, int outLayCh,////*************************************************kernelWeigthpast ekle 
	float* kernelWeigthTensor,//////////******bu pasta  reg ve act çarpýmýsonucu toplamý kaydet ,changeWe yaparken kolaylýk olsun 
	float* kernelPastTensor,
	float* kernelActivatorTensor, float * biasData){

};

