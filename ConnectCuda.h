#ifndef CONNECTCUDA_H
#define CONNECTCUDA_H
#include <cuda_runtime.h>
#include "Connect.h"
#include "ConnectActFunc.h"

//****** Error   Functions
extern void ErrorCalculateCu(float * outLayer, float* targetLayer,float * outParalelTensor, int sampleNum,int outFea,
	int outHe, int outWi, int outCha);
extern void errorSoftmaxCrEn(float * outLayer, float* targetLayer, float * outParalelTensor, int sampleNum, int outFea,
	int outHe, int outWi, int outCha);
//******  FC Connect Functions
extern void FullyLayerFeedworkCu(float* inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * totalOutLayerTensor, int totalOutLayerFe, int totalOutLayHe, int totalOutLayWi, int  totalOutLayCh,
	float* kernelWeigthTensor, float* biasTensor);
extern void FullyLayerBackPropagationCu(float* inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * totalOutLayerTensor,int totalOutLayFea, int totalOutLayHe, int totalOutLayWi, int totalOutLayCh,
	float* kernelWeigthTensor);
extern void ChangeFullyWeigthCu(float* inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * ErrorOutParalelTensor, int errOutFe, int errOutHe, int errOutWi, int errOutCh,
	float* kernelWeigthTensor, float* kernelParalelTensor ,float *biasTensor,
	float* biasParalelTensor,float learning, float momentum);
//****** FCwReg Connection  Functions
extern void FullyWeigthFullyRegulatorCu(
	float* inputWeLayerTensor, int inpLaySamp, int inpLayFea,
	int inpLayHe, int inpLayWi, int inpLayCh,
	float * regulatorLayerTensor, int regLayFea, int regLayHe,
	int regLayWi, int regLayCh,
	float * outLayerTensor, int outLayFe, int outLayHe,
	int outLayWi, int outLayCh,
	float* kernelWeigthTensor, float* kernelWePastTensor,
	float* kernelActivatorTensor, float* biasTensor);
extern void FullyWeigthFullyRegBackProCu(float* inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * regulatorLayerTensor, int regLayFea, int regLayHe,
	int regLayWi, int regLayCh,
	float * totalOutLayerTensor, int totalOutLayFea, int totalOutLayHe, int totalOutLayWi, int totalOutLayCh,
	float* kernelWeigthTensor,float* kernelWeigthPastTensor,
	float* kernelActivatorTensor);
extern void ChangeFullyWeFullyRegCu(float* inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * regulatorLayerTensor, int regLayFea, int regLayHe,
	int regLayWi, int regLayCh,
	float * ErrorOutParalelTensor, int errOutFe, int errOutHe, int errOutWi, int errOutCh,
	float* kernelWeigthTensor, float* kernelParalelTensor,float* kernelWePastTensor,
	float* kernelActivTensor,float* kernelActivParllTensor, float *biasTensor,
	float* biasParalelTensor, float learning, float momentum,float noise, float lear2,
	float momen2, float nois2);
//******  Showing  Functions
extern void error1D2D(float* accuracyTrain,float* accuracyTest, float* errorArray, uchar4 * other_out, int W, int H);
extern void errorSummer(float* ErrorLayer, float* errorArray, int sizePiece, int epoch, int widthScreen);
extern void errorPercent(float*OutLayer, float*targetLayer, float* errorArray,int sampleTotal,
	int sizePiece, int epoch, int widthScreen);
extern void showCompare(float * outLay, float *target, int size);
extern void preComWeigthShower(int pikSize, uchar4 * other_out, int W, int H, float * WeigthData, int sizeWeigth, float *ActivData, int actSize);
extern void accuracyMultinominal(float* OutLayer, float * targetLayer, float* errorArray,int sizePiksel,
	int sampleTotal, int sizePiece, int epoch, int widthScreen);
//******  Showing LayerANDConv Functions

extern void LayerShower(int pikSize, uchar4 * other_out, int W, int H, Layer * showingLayer);

///***      CONV2D   functions
extern void conv2d_Feed(float* inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * totalOutLayerTensor, int totalOutLayerFe, int totalOutLayHe, int totalOutLayWi, int  totalOutLayCh,
	float* kernelWeigthTensor,int kernelFea,int kernelHeWeSc,int kernelDepth,int striX,int striY,int padX,int padY,
	float* biasTensor);

extern void conv2d_Back(float* inputWeLayerParalelTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * totalOutLayerParalelTensor, int totalOutLayFea, int totalOutLayHe, int totalOutLayWi, int totalOutLayCh,
	float* kernelWeigthTensor, int kernelFea, int kernelHeWeSc, int kernelDepth, int striX, int striY, int padX, int padY,
	float* biasTensor);
extern void conv2d_Update(float* inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * ErrorOutParalelTensor, int errOutFe, int errOutHe, int errOutWi, int errOutCh,
	float* kernelWeigthTensor, float* kernelParalelTensor, int kernelFea, int kernelHeWeSc, int kernelDepth, int striX, int striY, int padX, int padY,
	float *biasTensor,float* biasParalelTensor, float learning, float momentum);

extern void conv2dV2Feed(float* inputWeLayerTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * totalOutLayerTensor, int totalOutLayerFe, int totalOutLayHe, int totalOutLayWi, int  totalOutLayCh,
	float* kernelWeigthTensor, int kernelFea, int kernelHeWeSc, int kernelDepth, int striX, int striY, int padX, int padY,
	float* biasTensor);
extern void conv2dV2Back(float* inputWeLayerParalelTensor, int inWeigthLaySamp,
	int inWeigthLayFea, int inWeigthLayHe, int inWeigthLayWi, int inWeigthLayCh,
	float * totalOutLayerParalelTensor, int totalOutLayFea, int totalOutLayHe, int totalOutLayWi, int totalOutLayCh,
	float* kernelWeigthTensor, int kernelFea, int kernelHeWeSc, int kernelDepth, int striX, int striY, int padX, int padY,
	float* biasTensor);

////===     POOL2D  function
extern void poolMax2d(Layer * inputLayer,Layer * outLayer,int poolScale);
extern void poolAvg2d(Layer * inputLayer, Layer * outLayer, int poolScale);
extern void poolMax2dBack(Layer * inputLayer, Layer * outLayer, int poolScale);
extern void poolAvg2dBack(Layer * inputLayer, Layer * outLayer, int poolScale);
///====     ALLDATA On GPU function
extern void shuffleGpu(float* sourceData,float * shuffledData, int totalSample, int heigth, int width,int chanal, int * shuffleArray);
extern void loadPieceOnGpu(float * sourceData, float* loadedData,int heigthWidth,int sample, int loadedPiece,int inputSampletotal);
extern void showonGpuData(float * sourceData, int size, int totalsample);
#endif