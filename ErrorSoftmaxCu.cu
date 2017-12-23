#include <cuda_runtime.h>
#include "ConnectCuda.h"
#include <cuda.h>
/*
__global__ void errorSoftmaxtargetDiv(const  float* __restrict__ outLayer,
	const  float* __restrict__ targetLayer, float* __restrict__ sampleErrorSumExpo,
	int sampleNum,int totalSize){
	int samplNumIn = blockIdx.x ;
	int Index =  threadIdx.x;
	//printf("\n Index %d", Index);
	if (samplNumIn<sampleNum){
		if (Index < totalSize){
			if (targetLayer[samplNumIn*totalSize + Index] == 1.){
				//printf("\n out%.6f  calc:%.5f      res:%.5f     inde:%d", expf(outLayer[samplNumIn*totalSize + Index]), sampleErrorSumExpo[samplNumIn],
					//targetLayer[samplNumIn*totalSize + Index], samplNumIn*totalSize + Index);
				sampleErrorSumExpo[samplNumIn] = expf(outLayer[samplNumIn*totalSize + Index]) / sampleErrorSumExpo[samplNumIn];
				//printf("\n out%.6f  calc:%.5f      res:%.5f     inde:%d", expf(outLayer[samplNumIn*totalSize + Index]), sampleErrorSumExpo[samplNumIn],
					//targetLayer[samplNumIn*totalSize + Index], samplNumIn*totalSize + Index);
			}
			
		}
	}
}

__global__ void errorSoftmaxSumExpo(const  float* __restrict__  outLayer,
	float * sampleErrorSumExpo, int sampleNum, int totalPikselSize){
	int samplNumIn = blockIdx.x*blockDim.x + threadIdx.x;
	int Index = blockIdx.y*blockDim.y + threadIdx.y;
	if (samplNumIn < sampleNum){
		atomicAdd(&sampleErrorSumExpo[samplNumIn], expf(outLayer[samplNumIn*totalPikselSize + Index]));
		//printf("\nooo:  %.4f  exp%.4f", sampleErrorSumExpo[samplNumIn], expf(outLayer[samplNumIn*totalPikselSize + Index]) );
	}
	//ErrorArray[step] += powf( errorLayer[Index] ,2)/ float(sizePiece);
	
} */

__global__ void errorSoftmaxGenelalize(float* __restrict__ sampleErrorSumExpo,
	float * outParalelTensor,
	int sampleNum, int totalSize){
	int samplNumIn = blockIdx.x*blockDim.x + threadIdx.x;
	int Index = blockIdx.y*blockDim.y + threadIdx.y;
	if (samplNumIn < sampleNum){
		if (Index < totalSize){
			outParalelTensor[samplNumIn*totalSize + Index] = sampleErrorSumExpo[samplNumIn];
		//	printf("\nooo:  %.4f  exp%.4f  in:%d", outParalelTensor[samplNumIn*totalSize + Index], sampleErrorSumExpo[samplNumIn],Index);
		}
	}
}


__global__ void errorSoftmaxModify(const  float* __restrict__ outLayer,
	const  float* __restrict__ targetLayer, float* __restrict__ sampleErrorSumExpo,
	float* __restrict__ outParalelTensor,
	int sampleNum, int totalPikselSize){
	int samplNumIn = blockIdx.x*blockDim.x + threadIdx.x;
	float totalExpo = 0;
	float totalError = 0;
	float tempSoft = 0;
	float temptarget = 0;
	float tempOut = 0;
	float tempError = 0;
	if (samplNumIn < sampleNum){
		for (int p = 0; p < totalPikselSize; p++){
			totalExpo += expf(outLayer[samplNumIn*totalPikselSize + p]);
		}
	
		for (int p = 0; p < totalPikselSize; p++){
			tempOut = outLayer[samplNumIn*totalPikselSize + p];//calculateed
			tempSoft = expf(tempOut) / totalExpo;
			temptarget = targetLayer[samplNumIn*totalPikselSize + p];//true Labels
			//tempError = -1*((temptarget*logf(tempSoft) + (1 - temptarget)*logf(1.00000 - tempSoft)));
			//if (tempOut < 0) tempOut = 0;
			//aþaðýsý harika çalýþýyor
			//tempError = -1 * ((temptarget*(temptarget - tempSoft)) + ((1 - temptarget)*(temptarget - tempSoft)));
			tempError = -1*(temptarget - tempSoft);
			//totalError += fabs(tempError);
			outParalelTensor[samplNumIn*totalPikselSize + p] = tempError;
			//printf("\n--  output:%.4f  %.2f :targ    error%.4f  ",tempOut,
				// temptarget, outParalelTensor[samplNumIn*totalPikselSize + p]);
		}
		/*for (int p = 0; p < totalPikselSize; p++){
			totalExpo += expf(outLayer[samplNumIn*totalPikselSize + p]);
			trueValExp += expf((outLayer[samplNumIn*totalPikselSize + p])*targetLayer[samplNumIn*totalPikselSize + p])+
				((1-outLayer[samplNumIn*totalPikselSize + p])*(1-targetLayer[samplNumIn*totalPikselSize + p]) );
			printf("\n----expOut %.6f   res:%.6f in:%d", expf(outLayer[samplNumIn*totalPikselSize + p]),
				targetLayer[samplNumIn*totalPikselSize + p],p);
		}
*/
	//	sampleErrorSumExpo[samplNumIn] = totalError/totalPikselSize;
	//	printf("\n error:%.4f  ", totalError / totalPikselSize);
	
	}
}

void errorSoftmaxCrEn(float * outLayer, float* targetLayer, float * outParalelTensor, int sampleNum, int outFea,
	int outHe, int outWi, int outCha){
	float * sampleErrorSumExpo;
	cudaMalloc(&sampleErrorSumExpo, sampleNum * sizeof(float));
	cudaMemset(sampleErrorSumExpo, 0, sampleNum * sizeof(float));
	int totalPikselSize =  outFea*outHe*outWi*outCha;
	int block = 8;
	dim3 BlockSize(block);
	dim3 GridSize((sampleNum + block - 1) / BlockSize.x);
	errorSoftmaxModify << <GridSize, BlockSize >> >(outLayer, targetLayer, sampleErrorSumExpo, outParalelTensor, sampleNum, totalPikselSize);
	//printf("\n SOftmax totalpiksel %d", totalPikselSize);
	//errorSoftmaxtargetDiv <<<sampleNum, totalPikselSize >> >(outLayer, targetLayer, sampleErrorSumExpo, sampleNum, totalPikselSize);
	BlockSize.x = 8; BlockSize.y = 32;
	GridSize.x = (sampleNum + BlockSize.x - 1) / BlockSize.x;
	GridSize.y = (totalPikselSize + BlockSize.y - 1) / BlockSize.y;
//	errorSoftmaxGenelalize << <GridSize, BlockSize >> >(sampleErrorSumExpo, outParalelTensor, sampleNum, totalPikselSize);
	cudaFree(sampleErrorSumExpo);
}


///////****************************** 
/*

softmax düzgün çalýþýyor mu ayrýca negatif log unutma 

Netdata fonnksiyonu son out datasý nominal ise týpký diðerlerinde olduðu gibi bir sýnýflandýrma yapmýyor
ikili üçlü sýnýflandýrma yerine sadece 1 2 diyor sorun da bu 



*/
