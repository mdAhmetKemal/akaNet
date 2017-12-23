#include <cuda_runtime.h>
#include "ConnectCudaVaryasyon.h"
#include <cuda.h>
#include "curand.h"
#include "curand_kernel.h"


void otherWeigthChanger(float* inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * regulatorLayerTensor, int regLayFea, int regLayHe,
	int regLayWi, int regLayCh,
	float * ErrorOutParalelTensor, int errOutFe, int errOutHe, int errOutWi, int errOutCh,
	float* kernelWeigthTensor, float* kernelParalelTensor, float* kernelWePasttensor,
	float* kernelActivTensor, float* kernelActivParllTensor, float *biasTensor,
	float* biasParalelTensor, float learning, float momentum, float noise, float lear2,
	float momen2, float nois2)
{
	

};