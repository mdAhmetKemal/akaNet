#ifndef CONNECTCUDAVARYASYON_H
#define CONNECTCUDAVARYASYON_H
#include <cuda_runtime.h>
#include "Connect.h"


extern void FullyWeigthFullyRegulatorCuRelu(
	float* inputWeLayerTensor, int inpLaySamp, int inpLayFea,
	int inpLayHe, int inpLayWi, int inpLayCh,
	float * regulatorLayerTensor, int regLayFea, int regLayHe,
	int regLayWi, int regLayCh,
	float * outLayerTensor, int outLayFe, int outLayHe,
	int outLayWi, int outLayCh,
	float* kernelWeigthTensor, float* kernelWePastTensor,
	float* kernelActivatorTensor, float* biasTensor);
extern void FullyWeigthFullyRegBackProCuRelu(float* inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * regulatorLayerTensor, int regLayFea, int regLayHe,
	int regLayWi, int regLayCh,
	float * totalOutLayerTensor, int totalOutLayFea, int totalOutLayHe, int totalOutLayWi, int totalOutLayCh,
	float* kernelWeigthTensor, float* kernelWeigthPastTensor,
	float* kernelActivatorTensor);
extern void otherWeigthChanger(float* inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * regulatorLayerTensor, int regLayFea, int regLayHe,
	int regLayWi, int regLayCh,
	float * ErrorOutParalelTensor, int errOutFe, int errOutHe, int errOutWi, int errOutCh,
	float* kernelWeigthTensor, float* kernelParalelTensor, float* kernelWePastTensor,
	float* kernelActivTensor, float* kernelActivParllTensor, float *biasTensor,
	float* biasParalelTensor, float learning, float momentum, float noise, float lear2,
	float momen2, float nois2);

#endif