#include <cuda_runtime.h>
#include "ConnectActFunc.h"
#include <math.h>
#include <cuda.h>

__global__ void lineerAct(float * layer, int layerSize){

}
__global__ void ReluAct(float * layer, int layerSize){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
		if (layer[index] < 0){
			layer[index] = 0.0;
		}
	//	else{
			// layer[index] = layer[index];
	//	}
	}
}
__global__ void LReluAct(float * layer, int layerSize, float parameter){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
		if (layer[index] < 0){
			layer[index] = layer[index] * parameter;
		}
	//	else{
		//	layer[index] = layer[index];
	//	}
	}
}
__global__ void EluAct(float * layer, int layerSize,float parameter){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
		if (layer[index] < 0){
			layer[index] = parameter*(expf(layer[index]) - 1);
		}
		//	else{
		// layer[index] = layer[index];
		//	}
	}
}
__global__ void SigmAct(float * layer, int layerSize){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
			layer[index] = 1/(expf(-layer[index]) + 1);	
	}
}
__global__ void TanhAct(float * layer, int layerSize){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
		layer[index] = (2 / (expf(-2 * layer[index]) + 1)) - 1;
	}
}
__global__ void StepAct(float * layer, int layerSize){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
		if (layer[index] < 0){
			layer[index] = 0.0;
		}
			else{
		 layer[index] = 1.0;
		}
	}
}
__global__ void ArctanAct(float * layer, int layerSize){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
		layer[index] = atanf( layer[index]);
	}
}
__global__ void SoftplusAct(float * layer, int layerSize){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
		layer[index] = log1pf(expf(layer[index])) ;
	}
}
extern void ActivationFunction(_ConnectActType layerAct, float * layer, int layerSize,float parameter){
	int block = 512;
	dim3 BlockSize(block);
	dim3 GridSize(layerSize + BlockSize.x - 1 / BlockSize.x);
	if (layerAct == _NOACTV || layerAct == _LINEER){
		return;
	}
	if (layerAct == _RELU){
		ReluAct << <GridSize, BlockSize >> >(layer, layerSize);
	}
	if (layerAct == _ELU){
		EluAct << <GridSize, BlockSize >> >(layer, layerSize, parameter);
	}
	if (layerAct == _LRELU){
		LReluAct << <GridSize, BlockSize >> >(layer, layerSize,parameter);
	}
	if (layerAct == _SIGM){
		SigmAct << <GridSize, BlockSize >> >(layer, layerSize);
	}
	if (layerAct == _TANH){
		TanhAct << <GridSize, BlockSize >> >(layer, layerSize);
	}
	if (layerAct == _STEP){
		StepAct << <GridSize, BlockSize >> >(layer, layerSize);
	}
	if (layerAct == _ARCTAN){
		ArctanAct << <GridSize, BlockSize >> >(layer, layerSize);
	}
	if (layerAct == _SOFTPLUS){
		SoftplusAct << <GridSize, BlockSize >> >(layer, layerSize);
	}
}