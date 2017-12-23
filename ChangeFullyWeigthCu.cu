#include <cuda_runtime.h>
#include "ConnectCuda.h"
#include <math.h>


__global__ void  ChangeFullyWeigthDe(const  float* __restrict__  inputWeLayerTensor,
	int inWeigthLaySamp,
	const  float* __restrict__  errorParalelTensor,
	float*  kernelParalelTensor, int inpFastSize, int outfastSize,
	float learning, float momentum, float* kernelWeigthTensor){
	int inputIndex, kernelIndex;
	int forSamp = blockIdx.x*blockDim.x + threadIdx.x;
	int numberOut = blockIdx.y*blockDim.y + threadIdx.y;
	float errorOutPiksel;
	
	if (forSamp < inWeigthLaySamp && numberOut < outfastSize)
	{
		errorOutPiksel = errorParalelTensor[inWeigthLaySamp*numberOut + forSamp];

		for (int fastInput = 0; fastInput < inpFastSize; fastInput++)
		{
			inputIndex = inWeigthLaySamp*(fastInput)+forSamp;
			kernelIndex = outfastSize*(fastInput)+numberOut;
			kernelParalelTensor[kernelIndex] += inputWeLayerTensor[inputIndex] * errorOutPiksel*learning;
		/*	printf("\n*** errorOutTarg: %.5f OutIndex %d  -inputX: %.5f  inInd %d  -kernelW: %.5f - KernelIndx: %d",
				errorOutPiksel, inWeigthLaySamp*numberOut + forSamp,
				inputWeLayerTensor[inputIndex], inputIndex,
				kernelWeigthTensor[kernelIndex], kernelIndex);
				*/
				
		}
	}
};
__global__ void  ChangeFullyWeBiasDe(
	int inWeigthLaySamp,
	const  float* __restrict__ errorParalelTensor,
	int outfastSize, float * biasParalelTensor,float learning,float momentum){

	int forSamp = blockIdx.x*blockDim.x + threadIdx.x;
	int numberOut = blockIdx.y*blockDim.y + threadIdx.y;
	float errorOutPiksel;

	if (forSamp < inWeigthLaySamp && numberOut < outfastSize)
	{
		//errorOutPiksel = errorParalelTensor[inWeigthLaySamp*numberOut + forSamp];
		biasParalelTensor[numberOut] += errorParalelTensor[inWeigthLaySamp*numberOut + forSamp] * learning;
		
	}
};
__global__ void  ChangeFullyWeUpdate(float* kernelWeigthTensor,
	const  float* __restrict__  kernelParalelTensor,
	int kernelFastSize){
	int kernelIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if (kernelIndex < kernelFastSize){
	/*	printf("\n*** kernelW: %.4f -kernelDelW: %.6f - -KernelIndx:%d ",
			kernelWeigthTensor[kernelIndex],
			kernelParalelTensor[kernelIndex],
			kernelIndex); */
		kernelWeigthTensor[kernelIndex] = kernelWeigthTensor[kernelIndex] - (kernelParalelTensor[kernelIndex]);
	}
}

__global__ void  ChangeFullyWeMomentumDe(
	float*  kernelParalelTensor, int KernelFastSize,float momentum){
	int kernelIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if (kernelIndex < KernelFastSize){
			kernelParalelTensor[kernelIndex] *= momentum;
			//printf("\nkernelParalelTensor[kernelIndex] : %.6f- kernelIndex %d ", kernelParalelTensor[kernelIndex], kernelIndex);
	}
}

void ChangeFullyWeigthCu(float* inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * ErrorOutParalelTensor, int errOutFe, int errOutHe, int errOutWi, int errOutCh,
	float* kernelWeigthTensor, float* kernelParalelTensor,
	float *biasTensor,
	float* biasParalelTensor,float learning,float momentum){
	
	int inpfastSize = inWeigthLayFea*inWeigthLayHe*inWeigthLayWi*inWeigthLayCh;
	int outFastSize = errOutFe*errOutHe*errOutWi*errOutCh;
	int kernelFastSize = outFastSize* inpfastSize;
	int blockNum = 256;
	dim3 BlockSize(blockNum);
	dim3 GridSize((kernelFastSize + BlockSize.x - 1) / BlockSize.x);
	ChangeFullyWeMomentumDe << <GridSize, BlockSize >> >(kernelParalelTensor, kernelFastSize, momentum);
	GridSize.x = (outFastSize + BlockSize.x - 1) / BlockSize.x;
	ChangeFullyWeMomentumDe << <GridSize, BlockSize >> >(biasParalelTensor, outFastSize, momentum);
	
	BlockSize.x = 2; BlockSize.y = 256;
	GridSize.x = (inWeigthLaySamp + BlockSize.x - 1) / BlockSize.x;
	GridSize.y = (outFastSize + BlockSize.y - 1) / BlockSize.y;
	ChangeFullyWeigthDe << < GridSize, BlockSize >> >(inputWeLayerTensor, inWeigthLaySamp,
		ErrorOutParalelTensor, kernelParalelTensor, inpfastSize, outFastSize, learning, momentum,
		kernelWeigthTensor);
	
	ChangeFullyWeBiasDe << < GridSize, BlockSize >> >(inWeigthLaySamp,ErrorOutParalelTensor,
		outFastSize,biasParalelTensor,learning,momentum);
	
	dim3 BlockSize2(256);
	dim3 GridSize2((kernelFastSize + BlockSize2.x - 1) / BlockSize2.x);
	ChangeFullyWeUpdate << < GridSize2, BlockSize2 >> >(kernelWeigthTensor, kernelParalelTensor,
		kernelFastSize);
	
	GridSize2.x = (outFastSize + BlockSize2.x - 1) / BlockSize2.x;
	ChangeFullyWeUpdate << < GridSize2, BlockSize2 >> >(biasTensor, biasParalelTensor,
		outFastSize);
};

