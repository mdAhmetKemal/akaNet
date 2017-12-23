#include <cuda_runtime.h>
#include "ConnectCuda.h"




__global__ void flipKernelCu(const float* __restrict__ kernel3dData, float* flippedKernel3data, int TkerFea, int TkerHe, int TkerCha){
	int kerFea_N = blockIdx.x*blockDim.x + threadIdx.x;
	int kerHe_N = blockIdx.y*blockDim.y + threadIdx.y;
	int kerWi_N = blockIdx.z*blockDim.z + threadIdx.z;
	for (int kerCha_N = 0; kerCha_N < TkerCha; kerCha_N++){
		flippedKernel3data[TkerFea*(TkerHe*(TkerHe*(kerCha_N)+kerWi_N) + kerHe_N) + kerFea_N] =
			kernel3dData[TkerFea*(TkerHe*(TkerHe*(kerCha_N)+TkerHe - 1 - kerWi_N) + TkerHe - 1 - kerHe_N) + kerFea_N];
	}

}

__global__ void conv2dV2cu(const float * __restrict__ inputData, int totalSam, int TinFea, int TinHe, int TinWi, int TinCha,
	float * outData, int ToutFea, int ToutHe, int ToutWi, int ToutCha,
	const float* __restrict__ kernel3data, int TkerFea, int TkerHe, int TkerCha,
	int stride, int pad){

	int outHe_N = blockIdx.x*blockDim.x + threadIdx.x;
	int outWi_N = blockIdx.y*blockDim.y + threadIdx.y;
	int outCha_N = blockIdx.z*blockDim.z + threadIdx.z;
	if (outHe_N < ToutHe && outWi_N < ToutWi && outCha_N < ToutCha){

		int outMinIndex = ToutHe*(ToutWi*(outCha_N)+outWi_N) + outHe_N;
		int halfKerSize = (TkerHe - 1) / 2;
		int outGlobalIndex;
		int inHe_N = outHe_N*stride - pad - halfKerSize;
		int inWi_N = outWi_N*stride - pad - halfKerSize;
		int	inCha_N;
		for (int inSam_N = 0; inSam_N < totalSam; inSam_N++){
			for (int inFea_N = 0; inFea_N < TinFea; inFea_N++){
				for (int kerFea_N = 0; kerFea_N < TkerFea; kerFea_N++){
					outGlobalIndex = totalSam*(TinFea*(TkerFea*(ToutHe*(ToutWi*(outCha_N)+outWi_N) + outHe_N) + kerFea_N) + inFea_N) + inSam_N;
					for (int keHe_N = 0; keHe_N < TkerHe; keHe_N++){
						for (int keWi_N = 0; keWi_N < TkerHe; keWi_N++){
							for (int keCha_N = 0; keCha_N < TkerCha; keCha_N++){
								if (inHe_N  >= 0 && inHe_N  < TinHe && inWi_N >= 0 && inWi_N  < TinWi){
									outData[outGlobalIndex] =
										inputData[totalSam*(TinFea*(TinHe*(TinWi*(keCha_N/*inCha_N*/)+inWi_N + keWi_N) + inHe_N + keHe_N) + inFea_N) + inSam_N] *
										kernel3data[TkerFea*(TkerHe*(TkerHe*(keCha_N)+keWi_N) + keHe_N) + kerFea_N];
								}
							}
						}
					}
				}
			}
		}
	}
}



void conv2dV2Back(float* inputDifData, int totalSam,
	int inFea, int inHe, int inWi, int inCha,
	float * outputDifData, int outFea, int outHe, int outWi, int  outCha,
	float* kernelData, int kerFea, int kerHe, int kerCha, int striX, int striY, int padX, int padY,
	float* biasData) {
	if (inCha == kerCha){

		float * flipKernel;
		int kernelSize = kerFea*kerHe*kerHe*kerCha;
		cudaMalloc(&flipKernel, kernelSize * sizeof(float));

		dim3 blockKernel(1,1,1);
		dim3 gridKernel(kerFea, kerHe, kerHe);
		flipKernelCu << <gridKernel, blockKernel >> >(kernelData, flipKernel,kerFea,kerHe,kerCha);

		int inputDataSize = totalSam*inCha*inHe*inWi*inCha;
		cudaMemset(inputDifData, 0, inputDataSize * sizeof(float));
		int blokX = 16;
		int blokY = 16;
		int blokZ = 1;
		dim3 BlockSize(blokX, blokY, blokZ);
		dim3 GridSize(inHe / BlockSize.x + 1, inWi / BlockSize.y + 1, inCha);

		conv2dV2cu << <GridSize, BlockSize >> >(outputDifData, totalSam, outFea, outHe, outWi, outCha,
			inputDifData, inFea, inHe, inWi, inCha,
			flipKernel, kerFea, kerHe, kerCha, striX, padX);
		cudaFree(flipKernel);
		//	conv2d_FeedBias << <grid2, blok2 >> >(inWeigthLaySamp, totalOutLayFe, kernelFea, totalOutLayerTensor, outFastSize, biasTensor);
		//ActivationFunction(_RELU, totalOutLayerTensor, outFastSize);
	}
};
void conv2dV2Feed(float* inputData, int totalSam,
	int inFea, int inHe, int inWi, int inCha,
	float * outputData, int outFea, int outHe, int outWi, int  outCha,
	float* kernelData, int kerFea, int kerHe, int kerCha, int striX, int striY, int padX, int padY,
	float* biasData) {
	int outputDataSize = totalSam*outFea*outHe*outWi*outCha;
	if (inCha == kerCha){
		cudaMemset(outputData, 0, outputDataSize * sizeof(float));
		int blokX = 16;
		int blokY = 16;
		int blokZ = 1;
		dim3 BlockSize(blokX, blokY, blokZ);
		dim3 GridSize(outHe / BlockSize.x + 1, outWi / BlockSize.y + 1, outCha);

		conv2dV2cu << <GridSize, BlockSize >> >(inputData, totalSam, inFea, inHe, inWi, inCha,
			outputData, outFea, outHe, outWi, outCha,
			kernelData, kerFea, kerHe, kerCha, striX, padX);

		//	conv2d_FeedBias << <grid2, blok2 >> >(inWeigthLaySamp, totalOutLayFe, kernelFea, totalOutLayerTensor, outFastSize, biasTensor);
		//ActivationFunction(_RELU, totalOutLayerTensor, outFastSize);
	}
};