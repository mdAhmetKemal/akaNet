#include <cuda_runtime.h>
#include "ConnectCuda.h"
#include <cuda.h>




__global__ void accuracyMultinominalCompareCu(const  float* __restrict__  outLayer,
	const  int* __restrict__  whichBigIndex,
	float * ErrorArray, int sampleTotal, int sizePiece, int totalPikselSize, int step){
int samplNumIn = blockIdx.x*blockDim.x + threadIdx.x;

if (samplNumIn < sizePiece){
	if ( outLayer[whichBigIndex[samplNumIn]] == 1.0){
		atomicAdd(&ErrorArray[step], 1.0 / sampleTotal);
	//	printf("\n%d    target: %.5f ", whichBigIndex[samplNumIn], outLayer[whichBigIndex[samplNumIn]]);
	}
}
}

__global__ void accuracyMultinominalCu(const  float* __restrict__  targetLayer,
	const  float* __restrict__ outLayer, float * ErrorArray,
	int totalPikselSize, int sizePiece, int sampleTotal,int step){
	int samplNumIn = blockIdx.x*blockDim.x + threadIdx.x;
	int bigIndex = 0;
	float bigValue = 0.0;
	if (samplNumIn < sizePiece){
		
		for (int p = 0; p<totalPikselSize; p++){
		//	printf("\nbigValue %.4f  outlayer%.4f  index%d", bigValue, outLayer[samplNumIn*totalPikselSize + p], p);

			if (outLayer[samplNumIn*totalPikselSize + p] > bigValue){
				bigValue = outLayer[samplNumIn*totalPikselSize + p];
				bigIndex = p;
				
			}
		}
		// printf("\n--bigValue %.4f  outlayer%.4f  index%d", targetLayer[samplNumIn*totalPikselSize + bigIndex],
			// outLayer[samplNumIn*totalPikselSize + bigIndex], bigIndex);
		if (targetLayer[bigIndex] == 1.0){
			atomicAdd(&ErrorArray[step], 1.0 / sampleTotal);
		}
	}
}
void accuracyMultinominal(float* OutLayer, float * targetLayer, float* errorArray, int sizePiksel,
	int sampleTotal, int sizePiece, int epoch, int widthScreen){

	if (epoch%widthScreen == 0){
		cudaMemset(errorArray, 0, widthScreen* sizeof(float));
	}
	int totalPikselSize = sizePiksel / sizePiece;
	int block = 8;
	dim3 BlockSize(block);
	dim3 GridSize((sizePiece + block - 1) / BlockSize.x);
	accuracyMultinominalCu << <GridSize, BlockSize >> >(OutLayer, targetLayer,errorArray, totalPikselSize,
		sizePiece, sampleTotal, (epoch%widthScreen));

};




