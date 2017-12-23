#include <cuda_runtime.h>
#include "ConnectCuda.h"
#include <cuda.h>

__device__ inline float logError(float error){
	return  error=  (__log10f(error));//15
}
__global__ void error2TextureDevice(const  float* __restrict__ accuracyTrain,
	const  float* __restrict__ accuracyTest,
	const  float* __restrict__  errorArray,
	uchar4 *d_out, int w, int h) {
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	const int r = blockIdx.y*blockDim.y + threadIdx.y;
	float WE = float(w), HE = float(h), Re = float(r),
		Co = float(c);
	float  pixAcc;
	int basamak = 20;//20
	
	if ((c < w) && (r < h)) {
		//printf("\nxxxx:  %.10f", errorArray[c]);
		int i = (c)+(h - r)*w; // 1D indexing
		//printf("  c: %d,   r: %d     i: %d  ", c, r,i);
		d_out[i].x = 255;
		d_out[i].y = 255;
		d_out[i].z = 255;
		d_out[i].w = 255;
		if ((accuracyTrain[c] / 2) >= (Re / HE)){

				d_out[i].x = 60;
				d_out[i].y = 170;
				d_out[i].z = 80;
				}
		/*else if (r < w / 2 && (accuracyTrain[c] / 2) != 0){
				d_out[i].x = 180;
				d_out[i].y = 200;
				d_out[i].z = 210;
			} */

			if ((accuracyTest[c ] / 2) >= (Re / HE)){
				d_out[i].x += 60;
				d_out[i].y -= 100;
				d_out[i].z += 45;
			} 
				
	/*	pixAcc = logError(errorArray[c]) / (basamak/2);
		//pixAcc = logError(10.0) / (basamak);  //****error 1. nereye denk
		if (pixAcc >= (Re / HE)){
			d_out[i].x +=40;
			d_out[i].y -= 70;
		} */
		
		if (r % (w / (basamak * 5)) == 0){
			d_out[i].y -= 10;
			if (r % ((w / basamak)) == 0){
				d_out[i].x -= 15;
				d_out[i].y -= 15;
				if (r % (int(w / (basamak / 5))) == 0){
					d_out[i].x -= 25;
					d_out[i].y -= 30;
					d_out[i].z -= 25;
				}
			}
		} 
		if (c % (w / (basamak * 5)) == 0){
			d_out[i].y -= 15;
			
		}
	}
}
__global__ void find2order(const  float* __restrict__  errorArray,
	uchar4 *d_out, int w, int h) {
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	const int r = blockIdx.y*blockDim.y + threadIdx.y;
	if ((c > 1) && (c < w - 1) && (r < h)) {
		int i = (c)+(h - r)*w; // 1D indexing
		float order = -errorArray[c - 1] + errorArray[c + 1];
		//order = logError(order) / (10);
		if (r < w / 2){
			if (order >= (float(r) / h)){
				d_out[i].x -= 20;
				d_out[i].y += 30;
			}
		}
	}
}

void error1D2D(float* accuracyTrain,
	float* accuracyTest,
	float* errorArray, uchar4 *out,
	int w, int h) {
	int block = 16;
	dim3 blockSize(block, block);
	dim3 gridSize((w + block - 1) / block, (h + block - 1) / block);
	error2TextureDevice << <gridSize, blockSize >> >(accuracyTrain, accuracyTest, errorArray, out, w, h);
	//printf("\nxx");
	//find2order << <gridSize, blockSize >> >( errorArray, out, w, h);
}




__global__ void errorSumDevice(const  float* __restrict__  paralelTensor,
	float * ErrorArray, int sizePiece, int step){
	 int Index = blockIdx.x*blockDim.x + threadIdx.x;
	// atomicAdd(&ErrorArray[step], (powf(errorLayer[Index], 2) / sizePiece));
	 atomicAdd(&ErrorArray[step], fabs(paralelTensor[Index]) / 500);
	//ErrorArray[step] += powf( errorLayer[Index] ,2)/ float(sizePiece);
 	// printf("\nooo  ooo :  %.10f", ErrorArray[step]);
}
void errorSummer(float* paralelTensor, float* errorArray,
	int sizePiece, int epoch, int widthScreen){
	if (epoch%widthScreen == 0){
		cudaMemset(errorArray, 0, widthScreen* sizeof(float));
	}
	errorSumDevice << <sizePiece, 1 >> >(paralelTensor, errorArray, sizePiece, (epoch%widthScreen));
};



__global__ void errorPercentDe(const  float* __restrict__  outLayer,
	const  float* __restrict__ targetLayer,
	float * ErrorArray, int sampleTotal, int step){

	int Index = blockIdx.x*blockDim.x + threadIdx.x;
	if (targetLayer[Index]>0.5 && outLayer[Index] == 1.0){
		atomicAdd(&ErrorArray[step], 1.0 / sampleTotal);
	}
	else if (targetLayer[Index]< 0.5 && outLayer[Index] == 0.0){
		atomicAdd(&ErrorArray[step], 1.0 / sampleTotal);
	}
	
	//printf("\nooo  ooo :  out%.5f    target: %.5f ", outLayer[Index], targetLayer[Index]);
}

void errorPercent(float* OutLayer, float * targetLayer, float* errorArray,
	int sampleTotal,int sizePiece, int epoch,int widthScreen){
	if (epoch%widthScreen == 0){
		cudaMemset(errorArray, 0, widthScreen* sizeof(float));
	}
	errorPercentDe << <sizePiece, 1 >> >(OutLayer, targetLayer, errorArray, sampleTotal, (epoch%widthScreen));
};



 __global__ void errorShow(const  float* __restrict__  Out,
	 const  float* __restrict__  target,int size){
	 int Index = blockIdx.x*blockDim.x + threadIdx.x;
	 printf("\nooo  ooo :  out%.5f    target: %.5f ", Out[Index], target[Index]);
 }
 void showCompare(float * outLay, float *target, int size){
	 errorShow << <size, 1 >> >(outLay, target, size);
 }
