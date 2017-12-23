#include <cuda_runtime.h>
#include "ConnectCuda.h"

__global__ void  FullyLayerFeedworkDeNoShaAlterna(
	const  float* __restrict__ inputWeLayerTensor,
	int inWeigthLaySamp,
	float * totalOutLayerTensor, const  float* __restrict__ kernelWeigthTensor,
	int inputfastSize, int kernelFastSize, int outFastSize,
	const  float* __restrict__ biasTensor){
	int forSamp = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int numberOut = blockIdx.y*blockDim.y + threadIdx.y;

	if (forSamp < inWeigthLaySamp && numberOut < outFastSize){
		float outPiksel = 0.;
		int kernelIndex, inputIndex;
#pragma unroll 
			for (int fastInput = 0; fastInput < inputfastSize; fastInput++)
			{
				inputIndex = inWeigthLaySamp*(fastInput)+forSamp;
				kernelIndex = outFastSize*(fastInput)+numberOut;
				outPiksel += inputWeLayerTensor[inputIndex] * kernelWeigthTensor[kernelIndex];
	/*		printf("\n-->Flfeed  in inPiVa : %.5f - inInd %d  kernelWe: %.5f - keInd %d   outPiVa:%.5f - outInd %d ",
					inputWeLayerTensor[inputIndex], inputIndex,
					kernelWeigthTensor[kernelIndex], kernelIndex,
					outPiksel,(inWeigthLaySamp*numberOut + forSamp)); */
			}
			totalOutLayerTensor[inWeigthLaySamp*numberOut + forSamp] = outPiksel;// +biasTensor[numberOut];
		/*	printf("\n>Flfeed outPiNoBi : %.5f - outWitBias %.5f  outIndx: - %d  bias: %.5f -biasIndx %d ", outPiksel,
				totalOutLayerTensor[inWeigthLaySamp*numberOut + forSamp],
				(inWeigthLaySamp*numberOut + forSamp),
				 biasTensor[numberOut], numberOut); */
	}
};
__global__ void  FullyBiasAdd(
	int inpLaySamp,
	float * outLayerTensor,
	int  outFastSize,
	const  float* __restrict__ biasData){

	int forSamp = blockIdx.x*blockDim.x + threadIdx.x;
	int numberOut = blockIdx.y*blockDim.y + threadIdx.y;
	if (forSamp < inpLaySamp && numberOut < outFastSize){
		outLayerTensor[inpLaySamp*numberOut + forSamp] += biasData[numberOut];
	}
};
void FullyLayerFeedworkCu(
	float* inputWeLayerTensor, int inWeigthLaySamp, int inWeigthLayFea,
	int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh, 
	float * totalOutLayerTensor,int totalOutLayFe, int totalOutLayHe, int totalOutLayWi,
	int totalOutLayCh,
	float* kernelWeigthTensor, float* biasTensor){
	int inpfastSize = inWeigthLayFea*inWeigthLayHe*inWeigthLayWi*inWeigthLayCh;
	int outFastSize = totalOutLayFe*totalOutLayHe*totalOutLayWi*totalOutLayCh;
	int kernelFastSize = outFastSize* inpfastSize;

	dim3 BlockSize(2, 512);
	dim3 GridSize((inWeigthLaySamp + 2 - 1) / BlockSize.x, (outFastSize + 512 - 1) / BlockSize.y);
	FullyLayerFeedworkDeNoShaAlterna << < GridSize, BlockSize >> >(inputWeLayerTensor, inWeigthLaySamp,
		totalOutLayerTensor,
		kernelWeigthTensor, inpfastSize, kernelFastSize, outFastSize, biasTensor);
	FullyBiasAdd << < GridSize, BlockSize >> >
		(inWeigthLaySamp, totalOutLayerTensor, outFastSize, biasTensor);
//	ActivationFunction(_RELU, totalOutLayerTensor, outFastSize);
};

/*buna benzer fully connect witt activitation için  4-4-3 boyurlu input-regulator-output  layerlarýndan
oluþan 3 baðlantýlý konnect alan ve input-inputweigth-regulator-regulatorActivite-output datalarý olan bir 
connect sýnýfý lazým bunlarý iþlem yaparken özellikle input -regulator kýsmý için shared meymory kullanma 
þansýný kullanmak gerekli olacak gibi durutyor */
/*
__global__ void  FullyLayerFeedworkDeNoShare(
const  float* __restrict__ inputWeLayerTensor,
int inWeigthLaySamp, int inWeigthLayFea,
int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
float * totalOutLayerTensor, int totalOutLayHe, int totalOutLayWi,
int totalOutLayCh, const  float* __restrict__ kernelWeigthTensor,
int forSamp,int outHe,int outWi,int outCh){
int inHe = blockIdx.x*blockDim.x + threadIdx.x;
int inWi = blockIdx.y*blockDim.y + threadIdx.y;
int inFe = blockIdx.z*blockDim.z + threadIdx.z;

if (inHe < inWeigthLayHe && inWi < inWeigthLayWi && inFe < inWeigthLayFea){
int inputIndex, kernelIndex;
float outPiksel = 0;
for (int inCh = 0; inCh < inWeigthLayCh; inCh++)
{
//inputIndex = inWeigthLaySamp*(inWeigthLayFea*(inWeigthLayHe*(inWeigthLayWi*(inCh)+inWi) + inHe) + inFe) + forSamp;
//kernelIndex = totalOutLayHe*(totalOutLayWi*(totalOutLayCh*(inWeigthLayFea*(inWeigthLayHe*(inWi)
//	+inHe) + inFe)+outCh) + outWi) + outHe;
//outPiksel += inputWeLayerTensor[inputIndex] * kernelWeigthTensor[kernelIndex];
outPiksel += inputWeLayerTensor[inWeigthLaySamp*(inWeigthLayFea*(inWeigthLayHe*(inWeigthLayWi*(inCh)+inWi) + inHe) + inFe) + forSamp] *
kernelWeigthTensor[totalOutLayHe*(totalOutLayWi*(totalOutLayCh*(inWeigthLayFea*(inWeigthLayHe*(inWi)+inHe) + inFe)+outCh) + outWi) + outHe];
}
totalOutLayerTensor[inWeigthLaySamp*(totalOutLayHe*(totalOutLayWi*(outCh)+outWi) + outHe) + forSamp] = outPiksel;
}
};

__global__ void  FullyLayerFeedworkDeShared(
const  float* __restrict__ inputWeLayerTensor,
int inWeigthLaySamp, int inWeigthLayFea,
int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
float * totalOutLayerTensor, int totalOutLayHe, int totalOutLayWi,
int totalOutLayCh, const  float* __restrict__ kernelWeigthTensor,
int inputfastSize, int kernelFastSize, int outFastSize){
int forSamp = blockIdx.x*blockDim.x + threadIdx.x;
int numberOut = blockIdx.y*blockDim.y + threadIdx.y;
//int numberOut = inWeigthLaySamp*(totalOutLayHe*(totalOutLayWi*(outCh)+outWi) + outHe) + forSamp;
__shared__  float inputFastData[1000];

if (forSamp < inWeigthLaySamp && numberOut < outFastSize){

for (int inputIndex = 0; inputIndex < inputfastSize; inputIndex++){
inputFastData[inputIndex] = inputWeLayerTensor[inWeigthLaySamp*(inputIndex)+forSamp];
__syncthreads();
}

float outPiksel = 0;
int kernelIndex, inputIndex;
for (int fastInput = 0; fastInput < inputfastSize; fastInput++)
{
#pragma unroll inWeigthLaySamp
kernelIndex = outFastSize*(fastInput)+numberOut;
//printf("\nkernel %d", kernelIndex);
outPiksel += inputFastData[fastInput] * kernelWeigthTensor[kernelIndex];
}
//printf("out index %d", numberOut);
totalOutLayerTensor[numberOut] = outPiksel;
}
};
*/