#include <cuda_runtime.h>
#include "ConnectCuda.h"

__global__ void maxPoolCu(const  float* __restrict__ inputData, float * outData, int sample, int feature,
	int chanal, int outHe, int outWi, int poolScale){
	int forHe = blockIdx.x*blockDim.x + threadIdx.x;
	int forWi = blockIdx.y*blockDim.y + threadIdx.y;
	float maxPiks;
	if (forHe < outHe &&forWi < outWi){
		for (int forSam = 0; forSam < sample; forSam++){
			for (int forFea = 0; forFea < feature; forFea++){
				for (int forCha = 0; forCha < chanal; forCha++){
					int outIndex =sample*(feature*(outHe*(outWi*(forCha)+forWi) + forHe) + forFea) + forSam;
					maxPiks = 0.0;
					for (int poolX = 0; poolX < poolScale; poolX++){
						for (int poolY = 0; poolY < poolScale; poolY++){
							int inIndex=sample*(feature*(outHe*(outWi*(forCha)+(forWi*poolScale)+poolY) + (forHe*poolScale)+poolX) + forFea) + forSam;
						//	printf("\n %.6f ", inputData[inIndex]);
							if (inputData[inIndex] > maxPiks)
								maxPiks = inputData[inIndex];

						}
					}
					outData[outIndex] = maxPiks*100;
					//printf("\n-%.6f ", outData[outIndex]);
				}
			}
		}
	}

}

void poolMax2d(Layer * inputLayer, Layer* outLayer, int poolScale){
	float * inputData = inputLayer->Tensor;
	float * outputData = outLayer->Tensor;
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
			maxPoolCu << <gridsize, bloksize >> >(inputData, outputData, inSam, inFea, inCha, outHe, outWi, poolScale);
		}
		else printf("\n PoolLayer Scale Problem");
	}
	else printf("\n PoolLayer Scale Problem");
}