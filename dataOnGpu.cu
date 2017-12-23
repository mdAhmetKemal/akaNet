#include <cuda_runtime.h>
#include "ConnectCuda.h"




__global__ void shuffleGpuCu(const float* __restrict__ sourceData, float * shuffledData,int sampleNum,
	int sampleSize,  int*  shuffleArray){
	int step = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (step < sampleNum){
		int tempStep = shuffleArray[step];
		for (int a = 0; a < sampleSize; a++){
			shuffledData[sampleNum*a + step] = sourceData[sampleNum*a + tempStep];
		}
		
	}
};

void shuffleGpu(float* sourceData,float * shuffledData, int totalSample, int heigth,
	int width,int chanal, int * shuffleArray){

	int * arrayGpu;
	cudaError_t target = cudaMalloc(&arrayGpu, totalSample*sizeof(int));
	cudaMemcpy(arrayGpu, shuffleArray, totalSample*sizeof(int), cudaMemcpyHostToDevice);

	dim3 Bloksize(512);
	dim3 Gridsize(totalSample / Bloksize.x + 1);

	shuffleGpuCu << <Gridsize, Bloksize >> >(sourceData, shuffledData, totalSample,
		heigth*width*chanal, arrayGpu);
	cudaFree(arrayGpu);

}


__global__ void datashowCu(float * data, int size,int sample){

	for (int b = 0; b < sample; b++){
		for (int a = 0; a < size; a++){

			if (data[sample*(a) + b] > 0){
				printf(" \x6A");
			}
			else{
				printf("  ");
			}

			if ((a + 1) % 28 == 0){
				printf("\n");
			}
		}
	}
}
void showonGpuData(float * sourceData, int size, int totalsample){
	printf("\n  %d  %d   \n", size, totalsample);
	if (size == 784){
		datashowCu << <1, 1 >> >(sourceData, size, totalsample);
	}
}


__global__ void partialSet(const float* __restrict__ sourceData, float * targetData, int sample, int size, int piece, int iSamTotal){

	for (int forSamp = 0; forSamp < sample; forSamp++){
		for (int forSiz = 0; forSiz < size; forSiz++){
			targetData[sample*(forSiz)+forSamp] = sourceData[iSamTotal*(forSiz)+forSamp + piece];
		}
	}
}


void loadPieceOnGpu(float * sourceData, float* loadedData, int heigth, int totalsample, int ProcesStepSample, int inputSampleTotal){
	partialSet << <1, 1 >> >(sourceData, loadedData, totalsample, heigth, ProcesStepSample*totalsample, inputSampleTotal);
}
