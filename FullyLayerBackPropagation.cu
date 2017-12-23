#include <cuda_runtime.h>
#include "ConnectCuda.h"


__global__ void  FullyLayerBackPropagationDeShared(float*  inputParalelTensor,
	int inWeigthLaySamp, int inWeigthLayFea,
	int inWeigthLayHe, int inWeigthLayWi, int inWeigthLaySize,
	const  float* __restrict__  errorParalelTensor, int totalOutLayHe, int totalOutLayWi, int totalOutLaySize,
	const  float* __restrict__ kernelWeigthTensor){
	int outIndex, kernelIndex, inputIndex;
	float inPiksel;
	const int fastOutSize = totalOutLayHe*totalOutLayWi;
	__shared__  float outFastData[10100];
	int idxHe = blockIdx.x*blockDim.x + threadIdx.x;
	int idxWi = blockIdx.y*blockDim.y + threadIdx.y;
	int idxFea = blockIdx.z*blockDim.z + threadIdx.z;
	int totalPiks = inWeigthLayFea*(inWeigthLayHe*(inWeigthLayWi*idxWi) + idxHe) + idxFea;
	if (totalPiks < inWeigthLaySize)
	{
		for (int inSam = 1; inSam < inWeigthLaySamp; inSam++)
		{
			inPiksel = 0;
			for (int fast = 0; fast < fastOutSize; fast++)
				for (int errHe = 1; errHe < totalOutLayHe; errHe++)
				{
					for (int errWi = 1; errHe < totalOutLayWi; errWi++)
					{
						outIndex = inWeigthLaySamp*(totalOutLayHe*(errWi)+errHe) + inSam;
						outFastData[fast] = errorParalelTensor[outIndex];
						__syncthreads();
					}
				}
			
			inputIndex = inWeigthLaySamp*(inWeigthLayFea*(inWeigthLayHe*(idxWi)+idxHe) + idxFea) + inSam;
			for (int errHe = 1; errHe < totalOutLayHe; errHe++)
			{
				for (int errWi = 1; errWi < totalOutLayWi; errWi++)
				{
					outIndex = totalOutLayHe*(errWi)+errHe;
					kernelIndex = inWeigthLayFea*(inWeigthLayHe*(inWeigthLayWi*(totalOutLayHe*(errWi)+errHe) + idxWi) + idxHe) + idxFea;
					inPiksel += outFastData[outIndex] * kernelWeigthTensor[kernelIndex];
					__syncthreads();
				}
			}
		
			inputParalelTensor[inputIndex] = inPiksel;
		}
	}
};

__global__ void  FullyLayerBackPropagationDeNoshared(float*  inputParalelTensor,
	int inWeigthLaySamp,
	const  float* __restrict__  errorParalelTensor, 
	const  float* __restrict__ kernelWeigthTensor, int inpFastSize,int KernelFastSize,int outfastSize){
	int errOutIndex, kernelIndex;
	float errInPiksel;
	int forSamp = blockIdx.x*blockDim.x + threadIdx.x;
	int numberInp = blockIdx.y*blockDim.y + threadIdx.y;

	if (forSamp < inWeigthLaySamp &&numberInp < inpFastSize)
	{
		errInPiksel = 0.;
#pragma unroll 
		for (int fastOut = 0; fastOut < outfastSize; fastOut++)
		{
			//kernelIndex = outFastSize*(fastInput)+numberOut;
			errOutIndex = inWeigthLaySamp*(fastOut)+forSamp;
			kernelIndex = inpFastSize*(fastOut)+numberInp;
			errInPiksel += errorParalelTensor[errOutIndex] * kernelWeigthTensor[kernelIndex];
		/*	printf("\n-->FlBackProfast out InPiksel : %.6f %d  kernelWe: %.6f toplam errorIn:%.6f", errorParalelTensor[errOutIndex],
				errOutIndex,
				kernelWeigthTensor[kernelIndex],
				kernelIndex, errInPiksel); */
		}
		inputParalelTensor[inWeigthLaySamp*numberInp + forSamp] = errInPiksel;
	//	printf("\n>FlBaP  : %.6f  inputIndex %d ", inputParalelTensor[inWeigthLaySamp*numberInp + forSamp],
		//	inWeigthLaySamp*numberInp + forSamp); 
	} 
};

void FullyLayerBackPropagationCu(float* inputErrorTensor, int inErrorLaySamp, int inErrorLayFea,
	int inErrorLayHe, int inErrorLayWi, int inErrorLayCh,
	float * errorOutLayerTensor, int totalOutLayFe, int totalOutLayHe, int totalOutLayWi, int totalOutLayCh,
	float* kernelWeigthTensor){
	int inpfastSize = inErrorLayFea*inErrorLayHe*inErrorLayWi*inErrorLayCh;
	int outFastSize = totalOutLayFe*totalOutLayHe*totalOutLayWi*totalOutLayCh;
	int kernelFastSize = outFastSize* inpfastSize;
	dim3 BlockSize(2, 512);
	dim3 GridSize((inErrorLaySamp + 2 - 1) / BlockSize.x, (inpfastSize + 512 - 1) / BlockSize.y);

	FullyLayerBackPropagationDeNoshared << < GridSize, BlockSize >> >(inputErrorTensor, inErrorLaySamp,
		errorOutLayerTensor, kernelWeigthTensor, inpfastSize, kernelFastSize, outFastSize);
	//DerivationFunction(_RELU, inputErrorTensor, inpfastSize);
};
