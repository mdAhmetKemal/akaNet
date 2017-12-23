#include <cuda_runtime.h>
#include <cuda.h>
#include "ConnectCuda.h"

__global__ void layerShowerCuda(uchar4 * other_out,int pikSize,int W,int H,const float * __restrict__ layer,
	int layerSample,int layerFeature,int layerChanal,int layerHeigth,int layerWeigth,
	int maxSample,int maxFeature){
	const int col = blockIdx.x*blockDim.x + threadIdx.x;
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	const int cha = blockIdx.z*blockDim.z + threadIdx.z;
	int thisSample, thisFeature,thisHeigth,thisWeigth;
	
	if (col < W && row < H){
		
		int outIndex = H*(row+6)+col+2;
		if (row < (H/2)&& col<(W/1)){
		
			thisSample = (col / (pikSize*layerWeigth));
			thisWeigth  = (col % (pikSize*layerWeigth)) / pikSize;
			thisFeature = (row / (pikSize*layerHeigth));
			thisHeigth = (row % (pikSize*layerHeigth)) / pikSize;
			if (thisSample < maxSample && thisFeature < maxFeature){
				//int inIndex = (layerWeigth *(layerHeigth *(layerSample*(0) + thisSample) + thisHeigth) + thisWeigth);
				int inIndex = layerSample*(layerFeature*(layerHeigth*(layerWeigth*(cha) +thisWeigth) + thisHeigth) + thisFeature) + thisSample;
				other_out[outIndex].x = unsigned(layer[inIndex] * 255);
				other_out[outIndex].y = unsigned(layer[inIndex] * 255);
				other_out[outIndex].z = unsigned(layer[inIndex] * 255);
				other_out[outIndex].w = 255;
				//*******sorunlarý burdaki göstergecler ile bulabilirsin 
				if (false){
					printf("\n lS:%d tS:%d mS:%d lF:%d tF:%d mF:%d lH:%d tH:%d lW:%d tW:%d col:%d row:%d H:%d W:%d piSi:%d ",
						layerSample, thisSample, maxSample, layerFeature, thisFeature, maxFeature, layerHeigth, thisHeigth, layerWeigth,
						thisWeigth, col, row, H, W, pikSize);
				}
					
			} else{
			other_out[outIndex].x = unsigned(240);
			other_out[outIndex].y = unsigned(240);
			other_out[outIndex].z = other_out[outIndex].x;
			other_out[outIndex].w = 255;
			}
			/*  eskisi 
			thisSample = (row / (pikSize*layerHeigth));
			thisWeigth = row % (pikSize*layerHeigth);
			thisFeature = (col / (pikSize*layerWeigth));
			thisHeigth = col % (pikSize*layerWeigth);
			if (thisSample < maxSample && thisFeature < maxFeature){
				int inIndex = layerSample*(layerFeature*(layerHeigth*(layerWeigth*(cha)+thisWeigth) + thisHeigth) + thisFeature) + thisSample;
				other_out[outIndex].x = unsigned(layer[inIndex] * 256);
				other_out[outIndex].y = other_out[outIndex].x;
				other_out[outIndex].z = other_out[outIndex].x;
				other_out[outIndex].w = 255;
				//*******sorunlarý burdaki göstergecler ile bulabilirsin 
				 printf("\n lS:%d tS:%d mS:%d lF:%d tF:%d mF:%d lH:%d tH:%d lW:%d tW:%d col:%d row:%d H:%d W:%d piSi:%d ",
					layerSample, thisSample, maxSample, layerFeature, thisFeature, maxFeature, layerHeigth, thisHeigth, layerWeigth,
					thisWeigth, col, row, H, W, pikSize); 
			}
		*/
			
		}
		else
		{
			
			other_out[outIndex].x = unsigned(240);
			other_out[outIndex].y = unsigned(240);
			other_out[outIndex].z = other_out[outIndex].x;
			other_out[outIndex].w = 255;
		}
	}
}

extern void LayerShower(int pikSize, uchar4 * other_out, int W, int H, Layer * showingLayer){

	float * dataPtr = showingLayer->Tensor;
	int layerSample = showingLayer->hTsample;
	int layerFeature = showingLayer->hTfeatureNum;
	int layerChanal = showingLayer->hTchanal;
	int layerHeigth = showingLayer->hTheigth;
	int layerWidth = showingLayer->hTwidth;
	int maxSample = W / (pikSize*layerSample);
	int maxFeature = (H / 2) / (pikSize*layerFeature);
	if (maxSample > layerSample) 
		maxSample = layerSample;
	if (maxFeature > layerFeature)
		maxFeature = layerFeature;
	if (maxSample >= layerSample && maxFeature >= layerFeature && layerChanal<=3){
		if (layerChanal == 1){
			
			int block = 8;
			dim3 blockSize(block, block,1);
			dim3 gridSize((W + blockSize.x - 1) / blockSize.x, (H + blockSize.y - 1) / blockSize.y,layerChanal);
		
			layerShowerCuda << <gridSize, blockSize >> > (other_out,pikSize, W, H, dataPtr, layerSample,
				layerFeature, layerChanal, layerHeigth, layerWidth,maxSample,maxFeature);
		}
	}

}