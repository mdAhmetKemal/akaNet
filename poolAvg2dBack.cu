#include <cuda_runtime.h>
#include "ConnectCuda.h"

__global__ void avgPoolBackCu(const  float* __restrict__ inputData, float* inputError, const  float* __restrict__  outputError, int sample, int feature,
	int chanal, int outHe, int outWi, int poolScale){
	int forHe = blockIdx.x*blockDim.x + threadIdx.x;
	int forWi = blockIdx.y*blockDim.y + threadIdx.y;
	float maxPiks;
	int maxX, maxY;
	if (forHe < outHe &&forWi < outWi){
		for (int forSam = 0; forSam < sample; forSam++){
			for (int forFea = 0; forFea < feature; forFea++){
				for (int forCha = 0; forCha < chanal; forCha++){
					int outIndex = sample*(feature*(outHe*(outWi*(forCha)+forWi) + forHe) + forFea) + forSam;
					maxPiks = 0.0;
					for (int poolX = 0; poolX < poolScale; poolX++){
						for (int poolY = 0; poolY < poolScale; poolY++){
							//	printf("\n %.6f ", inputData[inIndex]);
							int inIndex = sample*(feature*(outHe*(outWi*(forCha)+forWi + poolY) + forHe + poolX) + forFea) + forSam;
							inputError[inIndex] = 0.0;
							if (inputData[inIndex] > maxPiks){
								maxX = poolX;
								maxY = poolY;
								//maxPiks = inputData[inIndex];
							}
						}
					}
					inputError[sample*(feature*(outHe*(outWi*(forCha)+forWi + maxY) + forHe + maxX) + forFea) + forSam] =
						outputError[outIndex];
					//printf("\n-%.6f ", outData[outIndex]);
				}
			}
		}
	}

}

void poolAvg2dBack(Layer * inputLayer, Layer* outLayer, int poolScale){
	float * inputData = inputLayer->Tensor;
	float * inputError = inputLayer->paralelTensor;
	float * outputError = outLayer->paralelTensor;
	int inSam = inputLayer->hTsample;
	int inFea = inputLayer->hTfeatureNum;
	int inHe = inputLayer->hTheigth;
	int inWi = inputLayer->hTwidth;
	int inCha = inputLayer->hTchanal;
	int outSam = outLayer->hTsample;
	int outFea = outLayer->hTfeatureNum;
	int outHe = outLayer->hTheigth;
	int outWi = outLayer->hTwidth;
	int outCha = outLayer->hTchanal;
	if (inSam == outSam && inFea == outFea  &&  inCha == outCha){
		if (inHe == outHe*poolScale  &&  inWi == outWi*poolScale){
			dim3 bloksize(16, 16);
			dim3 gridsize((outHe + bloksize.x - 1) / bloksize.x, (outWi + bloksize.y - 1) / bloksize.y);
			avgPoolBackCu << <gridsize, bloksize >> >(inputData, inputError, outputError, inSam, inFea, inCha, outHe, outWi, poolScale);
		}
		else printf("\n PoolLayer Scale Problem");
	}
	else printf("\n PoolLayer Scale Problem");
}