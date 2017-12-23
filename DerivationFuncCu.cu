#include <cuda_runtime.h>
#include "ConnectActFunc.h"
#include <math.h>
#include <cuda.h>

__global__ void lineerDer(float * layer, int layerSize){

}
__global__ void ReluDer(float * layer, int layerSize){
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
__global__ void LReluDer(float * layer, int layerSize, float parameter){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
		if (layer[index] < 0){
			layer[index] =  parameter;
		}
			else{
		layer[index] = 1.;
			}
	}
}
__global__ void EluDer(float * layer, int layerSize, float parameter){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
		if (layer[index] < 0){
			layer[index] = parameter+layer[index] ;
		}
			else{
		layer[index] = 1.;
		}
	}
}
__global__ void SigmDer(float * layer, int layerSize){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
		layer[index] = layer[index] * (1. - layer[index]);
	}
}
__global__ void TanhDer(float * layer, int layerSize){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
		layer[index] = 1.-(layer[index] * layer[index]);
	}
}
__global__ void StepDer(float * layer, int layerSize){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
		if (layer[index] > 0){
			layer[index] = 1.0;
		}
		else{
			layer[index] = 0.0;
		}
	}
}
__global__ void ArctanDer(float * layer, int layerSize){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
		layer[index] =1./ (layer[index] * layer[index]+1.);
	}
}
__global__ void SoftplusDer(float * layer, int layerSize){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < layerSize){
		layer[index] = 1./(1.+expf(-layer[index]));
	}
}
extern void DerivationFunction(_ConnectActType layerDer, float * layer, int layerSize, float parameter){
	int block = 512;
	dim3 BlockSize(block);
	dim3 GridSize(layerSize + BlockSize.x - 1 / BlockSize.x);
	if (layerDer == _NOACTV || layerDer == _LINEER){
		return;
	}
	if (layerDer == _RELU){
		ReluDer << <GridSize, BlockSize >> >(layer, layerSize);
	}
	if (layerDer == _ELU){
		EluDer << <GridSize, BlockSize >> >(layer, layerSize, parameter);
	}
	if (layerDer == _LRELU){
		LReluDer << <GridSize, BlockSize >> >(layer, layerSize, parameter);
	}
	if (layerDer == _SIGM){
		SigmDer << <GridSize, BlockSize >> >(layer, layerSize);
	}
	if (layerDer == _TANH){
		TanhDer << <GridSize, BlockSize >> >(layer, layerSize);
	}
	if (layerDer == _STEP){
		StepDer << <GridSize, BlockSize >> >(layer, layerSize);
	}
	if (layerDer == _ARCTAN){
		ArctanDer << <GridSize, BlockSize >> >(layer, layerSize);
	}
	if (layerDer == _SOFTPLUS){
		SoftplusDer << <GridSize, BlockSize >> >(layer, layerSize);
	}
}