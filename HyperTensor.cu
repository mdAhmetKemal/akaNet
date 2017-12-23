#include "curand.h"
#include "curand_kernel.h"
#include <cuda_runtime.h>
#include "HyperTensor.h"

__global__ void  randTensorD(float * data,int size,float fac,unsigned int seed){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size){
		curandState states;
		curand_init(i, 1, 0, &states);
		data[i] = (curand_uniform(&states) * fac) - (fac / float(2));
		//printf("val:%.5f  ",  data[i]);
	}
};
__global__ void TensorFactor(float * data,const int size,const float fac){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size){
		data[i] = (data[i]*fac)-(fac/float(2));
		
	}
};


void  randTensor(float* data, int size,float fac){

	//printf("\nCu:randtensor fac:%.4f", fac);
	int block = 32;
	dim3 BlockSize(block);
	dim3 GridSize((size + block - 1) / BlockSize.x);
	randTensorD << < GridSize, BlockSize >> >(data,size,fac, 13142737);

};