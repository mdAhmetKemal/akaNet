#include <cuda_runtime.h>
#include <cuda.h>
#include "ShowNetCuda.h"

__global__ void connectShowerCuda(uchar4 * other_out, int pikSize, int W, int H,int areaW,int areaH,
	const float * __restrict__ kernel, const float * __restrict__ paralelKernel,
	 int kernelFeature, int kernelHeWi, int kernelChanal,int maxSquare, int maxFeatureBorder){
	const int scrW = blockIdx.x*blockDim.x + threadIdx.x;
	const int scrH = blockIdx.y*blockDim.y + threadIdx.y;
	const int cha = blockIdx.z*blockDim.z + threadIdx.z;
	int  thisFeature, thisHeigth, thisWeigth;

	if (scrW <areaW && scrH < areaH){
		int outIndex = H*(scrH+2*pikSize)+(W-areaW+scrW);
		int thisFeature = maxSquare * int(scrH / (pikSize*(kernelHeWi + 1))) + int(scrW / (pikSize*(kernelHeWi + 1)));
	
			thisHeigth = (scrH % (pikSize*(kernelHeWi + 1))) / pikSize;
			thisWeigth = (scrW % (pikSize*(kernelHeWi + 1))) / pikSize;
			if (false){
				printf("\n  keF:%d kethF:%d maxFe:%d kH:%d thiH:%d kerW:%d thiW:%d H:%d W:%d piSi:%d ",
					kernelFeature, thisFeature, maxSquare, kernelHeWi, thisHeigth, kernelHeWi,
					thisWeigth, scrH, scrW, areaW);
			}
			if (thisFeature < kernelFeature){
				if (thisHeigth == kernelHeWi || thisWeigth == kernelHeWi){
					other_out[outIndex].x = unsigned(240);
					other_out[outIndex].y = unsigned(100);
					other_out[outIndex].z = other_out[outIndex].x;
					other_out[outIndex].w = 255;
				}
				else {
					//int inIndex = (kernelWeigth *(kernelHeigth *(kernelSample*(0) + thisSample) + thisHeigth) + thisWeigth);
					int inIndex = kernelFeature*(kernelHeWi*(kernelHeWi*(cha)+thisWeigth) + thisHeigth) + thisFeature;
					other_out[outIndex].x = unsigned(kernel[inIndex] *25500);
					other_out[outIndex].y = unsigned(paralelKernel[inIndex] *1280000);
					other_out[outIndex].z = 0;
					other_out[outIndex].w = 255;
					//printf("\n %.12f", kernel[inIndex]);
					//*******sorunlarý burdaki göstergecler ile bulabilirsin 
				}
			}

			else{
				other_out[outIndex].x = unsigned(120);
				other_out[outIndex].y = unsigned(240);
				other_out[outIndex].z = other_out[outIndex].x;
				other_out[outIndex].w = 255;
			}

	}
}

extern void ConnectShower(int pikSize, uchar4 * other_out, int W, int H, Connect * showingConnect){

	float * dataPtr = showingConnect->Tensor;
	float * dataParalelPtr = showingConnect->paralelTensor;
	int kernelFeature = showingConnect->hTfeatureNum;
	int kernelChanal = showingConnect->hTchanal;
	int kernelHeWi = showingConnect->hTheigth;


	int maxFeatureBorder = (H / 2) / ((kernelHeWi + 1)*pikSize);
	int maxSquare = 1;
	while (maxSquare*maxSquare <= kernelFeature &&   maxSquare <= maxFeatureBorder){
		maxSquare++;
	}
	int areaW = maxSquare* ((kernelHeWi + 1)*pikSize);
	int areaH = maxSquare* ((kernelHeWi + 1)*pikSize);
	if (kernelChanal == 1){

		int block = 8;
		dim3 blockSize(block, block, 1);
		dim3 gridSize((areaW + blockSize.x - 1) / blockSize.x, (areaH + blockSize.y - 1) / blockSize.y, kernelChanal);

		connectShowerCuda << <gridSize, blockSize >> > (other_out, pikSize, W, H, areaW, areaH, dataPtr, dataParalelPtr,
			kernelFeature, kernelHeWi, kernelChanal, maxSquare, maxFeatureBorder);
	}


}