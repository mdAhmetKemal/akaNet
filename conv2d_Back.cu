#include <cuda_runtime.h>
#include "ConnectCuda.h"




__global__ void conv2d_BackCu(float*  inputWeParTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	const  float* __restrict__ totalOutParTensor,
	int totalOutLayFe, int totalOutLayHe, int totalOutLayWi, int  totalOutLayCh,
	const  float* __restrict__ kernelWeigthTensor,
	int kernelFea, int kernelHeWeSc, int kernelDepth, int striX, int striY, int padX, int padY){
	int outHe = blockIdx.x*blockDim.x + threadIdx.x;
	int outWi = blockIdx.y*blockDim.y + threadIdx.y;
	int chanal = blockIdx.z*blockDim.z + threadIdx.z;
	if (outHe >= totalOutLayHe || outWi >= totalOutLayWi || chanal >= totalOutLayCh)
		return;

	int inputCentreIndex;
	int outPiks3dIndex = totalOutLayHe*(totalOutLayWi*(chanal)+outWi) + outHe;
	int halfKernelSize = (kernelHeWeSc - 1) / 2;
	int kernelIndex, inputIndex, outCentralIndex;
	int inHe = (outHe*striX) - padX - halfKernelSize;
	int inWi = (outWi*striY) - padY - halfKernelSize;
	for (int samNu = 0; samNu < inWeigthLaySamp; samNu++){
		for (int feaNum = 0; feaNum < inWeigthLayFea; feaNum++){

			for (int keFe = 0; keFe < kernelFea; keFe++){
				outCentralIndex = inWeigthLaySamp*(inWeigthLayFea*(kernelFea*(outPiks3dIndex)+keFe) + feaNum) + samNu;
				for (int keHe = 0; keHe < kernelHeWeSc; keHe++){
					for (int keWi = 0; keWi < kernelHeWeSc; keWi++){
						for (int keCha = 0; keCha < kernelDepth; keCha++){
							kernelIndex = totalOutLayFe*(kernelHeWeSc*(kernelHeWeSc*(keCha)+keWi) + keHe) + feaNum;
							inputCentreIndex = inWeigthLaySamp*(inWeigthLayFea*(inWeigthLayHe*(inWeigthLayWi*(chanal)+inWi + keHe) + inHe + keHe) + feaNum) + samNu;
							if (inHe < 0 || inHe >= totalOutLayHe || inWi < 0 || inWi >= totalOutLayWi){

							}
							else{
								inputWeParTensor[inputCentreIndex] += totalOutParTensor[outCentralIndex] * kernelWeigthTensor[kernelIndex];

							}
						}
					}
				}
			}

		}
	}
}

__global__ void _showouterror(
	const  float* __restrict__ erroroutparalel
	){
	int index = blockIdx.x*blockDim.x + threadIdx.x;


	printf("\n second %.10f   %d", erroroutparalel[index], index);
	//kernelWeigthTensor[kernelFea*kernelIndex + forFea] -= kernelParalelTensor[kernelFea*kernelIndex + forFea];

}
__global__ void _showinerror(
	const  float* __restrict__ erroroutparalel
	){
	int index = blockIdx.x*blockDim.x + threadIdx.x;


	printf("\n first %.10f   %d", erroroutparalel[index], index);
	//kernelWeigthTensor[kernelFea*kernelIndex + forFea] -= kernelParalelTensor[kernelFea*kernelIndex + forFea];

}

void conv2d_Back(float* inputWeLayerParTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * totalOutLayerParTensor, int totalOutLayFe, int totalOutLayHe, int totalOutLayWi, int  totalOutLayCh,
	float* kernelWeigthTensor, int kernelFea, int kernelHeWeSc, int kernelDepth, int striX, int striY, int padX, int padY,
	float* biasTensor){
	int inpfastSize = inWeigthLaySamp*inWeigthLayFea*inWeigthLayHe*inWeigthLayWi*inWeigthLayCh;
	int outFastSize = inWeigthLaySamp*totalOutLayFe*totalOutLayHe*totalOutLayWi*totalOutLayCh;
	int blokX = 16;
	int blokY = 16;
	int blokZ = 2;
	dim3 BlockSize(blokX, blokY, blokZ);
	dim3 grido(inpfastSize);
	//_showinerror << <grido, 1 >> >(inputWeLayerParTensor);
	// burdansonrasý iyi bir optimizasyon gerektirebilir
	dim3 GridSize((totalOutLayHe + BlockSize.x - 1) / BlockSize.x,
		(totalOutLayWi + BlockSize.y - 1) / BlockSize.y,
		(totalOutLayCh + BlockSize.z - 1) / BlockSize.z);
	cudaMemset(inputWeLayerParTensor, 0, inpfastSize * sizeof(float));
	conv2d_BackCu << <GridSize, BlockSize >> >(inputWeLayerParTensor, inWeigthLaySamp,
		inWeigthLayFea, inWeigthLayHe, inWeigthLayWi, inWeigthLayCh,
		totalOutLayerParTensor,
		totalOutLayFe, totalOutLayHe, totalOutLayWi, totalOutLayCh,
		kernelWeigthTensor,
		kernelFea, kernelHeWeSc, kernelDepth, striX, striY, padX, padY);
	
	//_showinerror << <grido, 1 >> >(inputWeLayerParTensor);
	//_showinerror << <grido, 1 >> >(inputWeLayerParTensor);
};