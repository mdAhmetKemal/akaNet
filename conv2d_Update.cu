#include <cuda_runtime.h>
#include "ConnectCuda.h"





__global__ void conv2d_UpMom(
	float* kernelParalelTensor,
	int kernelFea,int kernelSize,float momentum
	){
	int kernelFeaInd = blockIdx.x*blockDim.x + threadIdx.x;
	int kernelIndex= blockIdx.y*blockDim.y + threadIdx.y;
	if (kernelFeaInd < kernelFea && kernelIndex < kernelSize){
		kernelParalelTensor[kernelFea*kernelIndex + kernelFeaInd] *= momentum;
		//printf("\n kernelIndex %d ", kernelFea*kernelIndex + kernelFeaInd);
	}
}
__global__ void conv2d_DelWeigth(
	const  float* __restrict__ inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	const  float* __restrict__ ErrorOutParalelTensor, int errOutFe, int errOutHe, int errOutWi, int errOutCh,
	float* kernelParalelTensor, int kernelFea, int kernelHeWeSc, int kernelDepth, int striX, int striY, int padX, int padY,
	int halfKernelSize, float learning){
	int forSamp = blockIdx.x*blockDim.x + threadIdx.x;
	int kerFea = blockIdx.y*blockDim.y + threadIdx.y;
	int outCentralIndex;
	int kernelIndex;
	int inHe, inWi, outHe, outWi;
	if (forSamp < inWeigthLaySamp && kerFea < kernelFea){
		for (int forFea = 0; forFea < inWeigthLayFea; forFea++){
			for (int keCha = 0; keCha < kernelDepth; keCha++){
				for (int keHe = 0; keHe < kernelHeWeSc; keHe++){
					for (int keWi = 0; keWi < kernelHeWeSc; keWi++){
						kernelIndex = kernelFea*(kernelHeWeSc*(kernelHeWeSc*(keCha)+keWi) + keHe) + kerFea;
						for (outHe = 0; outHe < errOutHe; outHe++){
							for (outWi = 0; outWi < errOutHe; outWi++){

								inHe = (outHe*striX) - padX - halfKernelSize + keHe;
								inWi = (outWi*striY) - padY - halfKernelSize + keWi;
								kernelParalelTensor[kernelIndex] +=
									inputWeLayerTensor[inWeigthLaySamp*(inWeigthLayFea*(inWeigthLayHe*(inWeigthLayWi*(keCha)+inWi) + inHe) + forFea) + forSamp] *
									ErrorOutParalelTensor[inWeigthLaySamp*(inWeigthLayFea*(kernelFea*(errOutHe*(errOutWi*(keCha)+outWi) + outHe) + kerFea) + forFea) + forSamp] * learning;
							}
						}
					}
				}
			}
		}
	}
}

__global__ void conv2d_UpDelta(
	const  float* __restrict__ kernelParalelTensor,
	float* kernelWeigthTensor,
	int kernelFea, int kernelSize){
	int forFea = blockIdx.x*blockDim.x + threadIdx.x;
	int kernelIndex = blockIdx.y*blockDim.y + threadIdx.y;
	if (forFea < kernelFea && kernelIndex<kernelSize){
		//printf("\n++ %f   %d", kernelWeigthTensor[kernelFea*kernelIndex + forFea], kernelFea*kernelIndex + forFea);
		kernelWeigthTensor[kernelFea*kernelIndex + forFea] -= kernelParalelTensor[kernelFea*kernelIndex + forFea];
		//printf("\n++ %f   %f", kernelWeigthTensor[kernelFea*kernelIndex + forFea], kernelParalelTensor[kernelFea*kernelIndex + forFea]);
		//kernelWeigthTensor[kernelFea*kernelIndex + forFea] = 0.5;
	}
}
__global__ void conv2d_showouterror(
	const  float* __restrict__ erroroutparalel
){
	int index = blockIdx.x*blockDim.x + threadIdx.x;


	//printf("\n+conUpd %.10f   %d", erroroutparalel[index], index);
		//kernelWeigthTensor[kernelFea*kernelIndex + forFea] -= kernelParalelTensor[kernelFea*kernelIndex + forFea];

}

__global__ void conv2d_DelBias(
	const  float* __restrict__ ErrorOutParalelTensor,
	int inWeigthLaySamp,int errOutFe, int errOutHe, int errOutWi, int errOutCh,
	float* biasParalelTensor, int kernelFea, float learning){
	int forSamp = blockIdx.x*blockDim.x + threadIdx.x;
	int forFea = blockIdx.y*blockDim.y + threadIdx.y;
	int forErrLaySize = blockIdx.z*blockDim.z + threadIdx.z;
	if (forSamp < inWeigthLaySamp &&forFea < kernelFea){
		biasParalelTensor[forFea] += ErrorOutParalelTensor[inWeigthLaySamp*(errOutFe*forErrLaySize + forFea) + forSamp] * learning;
	}
}

void conv2d_Update(float* inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * ErrorOutParalelTensor, int errOutFe, int errOutHe, int errOutWi, int errOutCh,
	float* kernelWeigthTensor, float* kernelParalelTensor, int kernelFea, int kernelHeWeSc, int kernelDepth, int striX, int striY, int padX, int padY,
	float *biasTensor, float* biasParalelTensor, float learning, float momentum){
/*	int inpfastSize = inWeigthLaySamp*inWeigthLayFea*inWeigthLayHe*inWeigthLayWi*inWeigthLayCh;
	int outFastSize = inWeigthLaySamp*totalOutLayFe*totalOutLayHe*totalOutLayWi*totalOutLayCh;
	int blokX = 16;
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
	dim3 blok2(16, 32);
	dim3 grid2((inWeigthLaySamp + BlockSize.x - 1) / BlockSize.x,
		(kernelFea + BlockSize.y - 1) / BlockSize.y);
	outFastSize = totalOutLayHe*totalOutLayWi*totalOutLayCh;
	conv2d_FeedBias << <grid2, blok2 >> >(inWeigthLaySamp, kernelFea, totalOutLayerTensor, outFastSize, biasTensor);
	*/
	int errorLayerSize = inWeigthLaySamp*errOutFe*errOutHe*errOutWi* errOutCh;
	int kernelSize = kernelHeWeSc*kernelHeWeSc*kernelDepth;
	int halfKernelSize = (kernelHeWeSc - 1) / 2;
	int blokA = 16;
	int blokB = 16;
	dim3 grido(errorLayerSize);
	conv2d_showouterror << <grido, 1 >> >(ErrorOutParalelTensor);
	dim3 BlockSize(blokA,blokB,1);
	dim3 GridSize((kernelFea + BlockSize.x - 1) / BlockSize.x, (kernelSize + BlockSize.y- 1) / BlockSize.y);
	conv2d_UpMom <<< GridSize, BlockSize >> > (kernelParalelTensor,kernelFea,kernelSize,momentum);
	
	GridSize.x = (kernelFea + BlockSize.x - 1) / BlockSize.x;
	GridSize.y = 1; BlockSize.y = 1;
	conv2d_UpMom << < GridSize, BlockSize >> > (biasParalelTensor, kernelFea, 1, momentum);
	//******buraya kadar doðruuuuu
	//cudaMemset(kernelParalelTensor, 0, kernelSize * sizeof(float));..s
	GridSize.x = (inWeigthLaySamp + BlockSize.x - 1) / BlockSize.x;
	BlockSize.y = blokB;
	GridSize.y = (kernelFea + BlockSize.y - 1) / BlockSize.y;
	conv2d_DelWeigth << <GridSize, BlockSize >> >(inputWeLayerTensor, inWeigthLaySamp, inWeigthLayFea,inWeigthLayHe,inWeigthLayWi,
		inWeigthLayCh, ErrorOutParalelTensor, errOutFe, errOutHe, errOutWi, errOutCh, kernelParalelTensor, kernelFea, kernelHeWeSc,
		kernelDepth, striX, striY, padX, padY, halfKernelSize,learning);
	GridSize.z = errorLayerSize ;
	//cudaMemset(biasParalelTensor, 0, kernelFea * sizeof(float));
	/*conv2d_DelBias << <GridSize, BlockSize >> >(ErrorOutParalelTensor, inWeigthLaySamp,errOutFe, errOutHe, errOutWi, errOutCh,
		biasParalelTensor,
		kernelFea,
		learning); */
	dim3 BlockSize2(16, 16);
	dim3 GridSize2((kernelFea + BlockSize.x - 1) / BlockSize.x, (kernelSize + BlockSize.y - 1) / BlockSize.y);
	conv2d_UpDelta << <GridSize2, BlockSize2 >> >(kernelParalelTensor,kernelWeigthTensor, kernelFea, kernelSize);
	GridSize2.y = 1;
	//conv2d_UpDelta << <GridSize2, BlockSize2 >> >(biasTensor, biasParalelTensor, kernelFea, 1);

};