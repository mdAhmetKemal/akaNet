#include <cuda_runtime.h>
#include "ConnectCuda.h"
#include <cuda.h>

__global__ void ErrorCalculateDevice(const  float* __restrict__ outLayer,
	const  float* __restrict__ targetLayer,
	float * outParalelTensor,
	int totalPikselSize){
	int sampleNum = blockIdx.x*blockDim.x + threadIdx.x;
	float totalEr = 0;
	for (int p = 0; p < totalPikselSize; p++){
		totalEr += powf((outLayer[sampleNum*totalPikselSize + p] - targetLayer[sampleNum*totalPikselSize + p]), 2);
		printf("\nout:%.5f    target%.5f    in:%d", outLayer[sampleNum*totalPikselSize + p],
			targetLayer[sampleNum*totalPikselSize + p], p);
	}
//	totalEr = -sqrtf(totalEr);
	printf("\ntotalError %.4f ", totalEr);
	for (int p = 0; p < totalPikselSize; p++){
		outParalelTensor[sampleNum*totalPikselSize + p] = totalEr;
	}
	printf("\n");
}



void ErrorCalculateCu(float * outLayer, float* targetLayer, float * outParalelTensor, int sampleNum,int outFea,
	int outHe, int outWi, int outCha){
	int totalPikselSize =  outFea*outHe*outWi*outCha;

	dim3 BlockSize(1);
	dim3 GridSize(sampleNum );
	ErrorCalculateDevice << <GridSize, BlockSize >> >(outLayer, targetLayer, outParalelTensor, totalPikselSize); 

}