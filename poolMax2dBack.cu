#include <cuda_runtime.h>
#include "ConnectCuda.h"

__global__ void maxPoolBackCu(const  float* __restrict__ inputData, float* inputError, const  float* __restrict__  outputError, int sample, int feature,
	int chanal, int outHe, int outWi, int poolScale){
	int forHe = blockIdx.x*blockDim.x + threadIdx.x;
	int forWi = blockIdx.y*blockDim.y + threadIdx.y;
	float maxPiks;
	int maxX, maxY;
	if (forHe < outHe && forWi < outWi){
		for (int forSam = 0; forSam < sample; forSam++){
			for (int forFea = 0; forFea < feature; forFea++){
				for (int forCha = 0; forCha < chanal; forCha++){
					int outIndex = sample*(feature*(outHe*(outWi*(forCha)+forWi) + forHe) + forFea) + forSam;
					maxPiks = 0.0;
					maxX = 0; maxY = 0;
					for (int poolX = 0; poolX < poolScale; poolX++){
						for (int poolY = 0; poolY < poolScale; poolY++){
							//	printf("\n %.6f ", inputData[inIndex]);
							int inIndex = sample*(feature*(outHe*(outWi*(forCha)+(forWi*poolScale) + poolY) + (forHe*poolScale) + poolX) + forFea) + forSam;
							inputError[inIndex] = 0.0;
							if (inputData[inIndex] > maxPiks){ 
								maxX = poolX;
								maxY = poolY;
								maxPiks = inputData[inIndex];
							}
						}
					}
					inputError[sample*(feature*(outHe*(outWi*(forCha)+(forWi*poolScale) + maxY) + (forHe*poolScale) + maxX) + forFea) + forSam] =
						outputError[outIndex];
					//printf("\n******- %.12f   %d ", outputError[outIndex], outIndex);
				}
			}
		}
	}

}
__global__ void pool_showouterror(
	const  float* __restrict__ erroroutparalel
	){
	int index = blockIdx.x*blockDim.x + threadIdx.x;


	//printf("\n++^^ %.16f   %d", erroroutparalel[index], index);
	//kernelWeigthTensor[kernelFea*kernelIndex + forFea] -= kernelParalelTensor[kernelFea*kernelIndex + forFea];

}

void poolMax2dBack(Layer * inputLayer, Layer* outLayer, int poolScale){
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
	//printf("\n poool  %d ", outLayer->sizePiksel);

	if (inSam == outSam && inFea == outFea  &&  inCha == outCha){
		if (inHe == outHe*poolScale  &&  inWi == outWi*poolScale){
			dim3 bloksize(16, 16);
			dim3 gridsize((outHe + bloksize.x - 1) / bloksize.x, (outWi + bloksize.y - 1) / bloksize.y);
			maxPoolBackCu << <gridsize, bloksize >> >(inputData,inputError, outputError, inSam, inFea, inCha, outHe, outWi, poolScale);
		}
		else printf("\n PoolLayer Scale Problem");
	}
	else printf("\n PoolLayer Scale Problem");
	//dim3 grido(inputLayer->sizePiksel);
	//pool_showouterror << <grido, 1 >> >(inputError);
}