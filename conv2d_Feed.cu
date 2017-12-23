#include <cuda_runtime.h>
#include "ConnectCuda.h"




__global__ void conv2d_FeedCu(const  float* __restrict__ inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * totalOutLayerTensor,
	int totalOutLayFe, int totalOutLayHe, int totalOutLayWi, int  totalOutLayCh,
	const  float* __restrict__ kernelWeigthTensor,
	int kernelFea, int kernelHeWeSc, int kernelDepth, int striX, int striY, int padX, int padY){
	int outHe = blockIdx.x*blockDim.x + threadIdx.x;
	int outWi = blockIdx.y*blockDim.y + threadIdx.y;
	int outCha = blockIdx.z*blockDim.z + threadIdx.z;
	if (outHe < totalOutLayHe && outWi < totalOutLayWi && outCha < totalOutLayCh){


		int outPiks3dIndex = totalOutLayHe*(totalOutLayWi*(outCha)+outWi) + outHe;
		int halfKernelSize = (kernelHeWeSc - 1) / 2;

		int outCentralIndex;
		int inHe, inWi;
		inHe = (outHe*striX) - padX - halfKernelSize;
		inWi = (outWi*striY) - padY - halfKernelSize;
		for (int samNu = 0; samNu < inWeigthLaySamp; samNu++){
			for (int feaNum = 0; feaNum < inWeigthLayFea; feaNum++){
				//hem out feature //hemde kernelFeature


				for (int keFe = 0; keFe < kernelFea; keFe++){
					outCentralIndex = inWeigthLaySamp*(inWeigthLayFea*(kernelFea*(totalOutLayHe*(totalOutLayWi*(outCha)
						+outWi) + outHe) + keFe) + feaNum) + samNu;
					for (int keHe = 0; keHe < kernelHeWeSc; keHe++){
						for (int keWi = 0; keWi < kernelHeWeSc; keWi++){
							for (int keCha = 0; keCha < kernelDepth; keCha++){
								if (inHe >= 0 && inHe < inWeigthLayHe && inWi >= 0 && inWi < inWeigthLayWi){
									totalOutLayerTensor[outCentralIndex] +=
										inputWeLayerTensor[inWeigthLaySamp*(inWeigthLayFea*(inWeigthLayHe*(inWeigthLayWi*(keCha)
										+inWi + keWi) + inHe + keHe) + feaNum) + samNu] *
										kernelWeigthTensor[kernelFea*(kernelHeWeSc*(kernelHeWeSc*(keCha)+keWi) + keHe) + keFe];
								}
								else{

								}
							}
						}
					}
				}
			}

		}
	}
}

__global__ void conv2d_FeedBias(
	int inpLaySamp, int totalOutFea, int kernelFea,
	float * outLayerTensor,
	int  outFastSize,
	const  float* __restrict__ biasData){

	int forSamp = blockIdx.x*blockDim.x + threadIdx.x;
	int forFea = blockIdx.y*blockDim.y + threadIdx.y;
	int kerFe = blockIdx.z*blockDim.z + threadIdx.z;
	if (forSamp < inpLaySamp && forFea < totalOutFea && kerFe<kernelFea){

		float bias = biasData[kerFe];
		for (int out = 0; out < outFastSize; out++){
			outLayerTensor[inpLaySamp*(totalOutFea*(kernelFea*(out)+kerFe) + forFea) + forSamp] += bias;
		}

	}
};

void conv2d_Feed(float* inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * totalOutLayerTensor, int totalOutLayFe, int totalOutLayHe, int totalOutLayWi, int  totalOutLayCh,
	float* kernelWeigthTensor, int kernelFea, int kernelHeWeSc, int kernelDepth, int striX, int striY, int padX, int padY,
	float* biasTensor){
	int inpfastSize = inWeigthLaySamp*inWeigthLayFea*inWeigthLayHe*inWeigthLayWi*inWeigthLayCh;
	int outFastSize = inWeigthLaySamp*totalOutLayFe*totalOutLayHe*totalOutLayWi*totalOutLayCh;

	if (inWeigthLayCh == kernelDepth){
		int blokX = 4;
		int blokY = 16;
		int blokZ = 2;
		dim3 BlockSize(blokX, blokY, blokZ);
		dim3 GridSize((totalOutLayHe + BlockSize.x - 1) / BlockSize.x,
			(totalOutLayWi + BlockSize.y - 1) / BlockSize.y,
			(totalOutLayCh + BlockSize.z - 1) / BlockSize.z);
		cudaMemset(totalOutLayerTensor, 0, outFastSize * sizeof(float));
		conv2d_FeedCu << <GridSize, BlockSize >> >(inputWeLayerTensor, inWeigthLaySamp,
			inWeigthLayFea, inWeigthLayHe, inWeigthLayWi, inWeigthLayCh,
			totalOutLayerTensor,
			totalOutLayFe, totalOutLayHe, totalOutLayWi, totalOutLayCh,
			kernelWeigthTensor,
			kernelFea, kernelHeWeSc, kernelDepth, striX, striY, padX, padY);
		dim3 blok2(8, 8, 8);
		dim3 grid2((inWeigthLaySamp + BlockSize.x - 1) / BlockSize.x,
			(totalOutLayFe + BlockSize.y - 1) / BlockSize.y,
			(kernelDepth + BlockSize.z - 1) / BlockSize.z);
		outFastSize = totalOutLayHe*totalOutLayWi*totalOutLayCh;
	//	conv2d_FeedBias << <grid2, blok2 >> >(inWeigthLaySamp, totalOutLayFe, kernelFea, totalOutLayerTensor, outFastSize, biasTensor);
		//ActivationFunction(_RELU, totalOutLayerTensor, outFastSize);
	}
};