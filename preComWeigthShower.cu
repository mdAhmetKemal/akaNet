#include <cuda_runtime.h>
#include "ConnectCuda.h"
#include <cuda.h>



__device__ inline float logError(float error){
	if (error > 1){
		error =(error/10)+200;
	}
	else if (error < 1){
		error *= 200;
	}
	return  error ;
}

__global__ void weigthShower(int boyut, uchar4 * other_out, int W, int H,
	const  float* __restrict__ WeigthData, int sizeWeigth) {

	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	const int r = blockIdx.y*blockDim.y + threadIdx.y;

	if ((c < W) && (r < H) && (r < boyut) && (c < boyut*sizeWeigth)) {

		int i = (c)+(r)*W; 
		int we = int(r/boyut);
		int he = int(c/ boyut);
		
		if (he <= sizeWeigth && we == 0){
			float value = WeigthData[he];
			if (value < 0){
				value *= -1;
				other_out[i].x = 0;
				other_out[i].y = 0;
				other_out[i].z = unsigned(fminf(254.,(logError( value)))) ;
				other_out[i].w = 255;
				//printf(" \n c: %d,    ", other_out[i].z );
			}
			else if (value == 0){
				other_out[i].x = 127;
				other_out[i].y = 127;
				other_out[i].z = 127;
				other_out[i].w = 255;
			}
			else if (value > 0){
				other_out[i].x = unsigned(fminf(254., (logError( value))));
				other_out[i].y = 0;
				other_out[i].z = 0;
				other_out[i].w = 255;
			}
		}
	
	}
}


__global__ void weigthActivShower(int boyut, uchar4 * other_out, int W, int H,
	const  float* __restrict__ WeigthData, int sizeWeigth, const  float* __restrict__ ActData, 
	int actSize) {

	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	const int r = blockIdx.y*blockDim.y + threadIdx.y;

	if ((c < W) && (r < H) && (r < boyut*(actSize+1)) && (c < boyut*actSize)) {
		int square = actSize;
		int i = (c)+(r)*W;
		int we = int(r / boyut);
		int he = int(c / boyut);

		if (he <= sizeWeigth && we ==1){
			float value = WeigthData[he];
			if (value < 0){
				value *= -1;
				other_out[i].x = 255 - unsigned(fminf(254., (logError(value))));
				other_out[i].y = 255 - unsigned(fminf(254., (logError(value))));
				other_out[i].z = 255;
				other_out[i].w = 255;
				//printf(" \n c: %d,    ", other_out[i].z );
			}
			else if (value == 0){
				other_out[i].x = 127;
				other_out[i].y = 127;
				other_out[i].z = 127;
				other_out[i].w = 255;
			}
			else if (value > 0){
				other_out[i].x = 255;
				other_out[i].y = 255 - unsigned(fminf(254., (logError(value))));
				other_out[i].z = 255 - unsigned(fminf(254., (logError(value))));
				other_out[i].w = 255;
			}
		}
		i = (c)+(r+boyut*3)*W;
		we = int(r / boyut);
		he = int(c / boyut);

		if (he < square && we < square){
			float value = ActData[he + square*we];
			if (value < 0){
				value *= -1;
				other_out[i].x = 255 - unsigned(fminf(254., (logError(value))));
				other_out[i].y = 255 - unsigned(fminf(254., (logError(value))));
				other_out[i].z = 255;
				other_out[i].w = 255;
				//printf(" \n c: %d,    ", other_out[i].z );
			}
			else if (value > 0){
				other_out[i].x = 255;
				other_out[i].y = 255 - unsigned(fminf(254., (logError(value))));
				other_out[i].z = 255 - unsigned(fminf(254., (logError(value))));
				other_out[i].w = 255;
			}else 
			{
				other_out[i].x = 127;
				other_out[i].y = 127;
				other_out[i].z = 127;
				other_out[i].w = 255;
			}
			
		}

	}
}





extern void preComWeigthShower(int boyut, uchar4 * other_out, int W, int H, float * WeigthData,
	int sizeWeigth, float *ActivData, int actSize){
	if (actSize == 0){
		int block = 32;
		dim3 blockSize(block, block);
		dim3 gridSize((W + block - 1) / block, (H + block - 1) / block);
		weigthShower << <gridSize, blockSize >> >(boyut,other_out, W, H,WeigthData,sizeWeigth);
	}
	else{
		actSize = sqrt(actSize);
		int block = 32;
		dim3 blockSize(block, block);
		dim3 gridSize((W + block - 1) / block, (H + block - 1) / block);
		weigthActivShower << <gridSize, blockSize >> >(boyut, other_out, W, H, WeigthData, sizeWeigth,ActivData,actSize);
	}
}