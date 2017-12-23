#ifndef CONNECT_H
#define CONNECT_H
#include "HyperTensor.h"
#include "EnumType.h"
#include "Layer.h"
#include "ConnectCuda.h"
#include "ConnectCudaVaryasyon.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
class Connect :public HyperTensor{
public:
	// Layer * acWeigthLayPtr;
	Connect(Layer* inWeLaNum, Layer* OutLaNu,
		_ConnectConvType _CoConvTy = _CONV2D, _ConnectPoolType _CoPoTy = _NOPOOL,
		_ConnectActType _CoActTy = _LINEER, int k_Fea = 12, int k_Scale = 3, int k_Dep = 1
		, int strideConv = 1, int paddConv = 1, float  fac = 0.01, int poolR = 1);
	Connect(Layer* inWeLaNum, Layer* regLaNum, Layer* OutLaNu,
		_ConnectConvType _WeConvTy = _CONV2D, _ConnectPoolType _WePoTy = _NOPOOL,
		_ConnectActType _WeActTy = _LINEER, int k_Fea = 12, int k_Scale = 3, int k_Dep = 1
		, int k_strideConv = 1, int k_paddConv = 1, float  k_fac = 0.01, int k_poolR = 1,
		_ConnectConvType _ReConvTy = _CONV2D, _ConnectPoolType _RePoTy = _NOPOOL,
		_ConnectActType _ReActTy = _LINEER, int a_Fea = 12, int a_Scale = 3, int a_Dep = 1
		, int a_strideConv = 1, int a_paddConv = 1, float  a_fac = 0.01, int a_poolR = 1 );
	~Connect();
	void FeedWorkConnect();
	void ErrorConnect();
	void backPropagationConnect();
	void changeWeigthConnect(float learning, float momentum, float noise,
		float learn2, float momen2, float nois2);
	void setConnectProcess(_ConnectProcessType _CoProcesTyp);
	void setNewConv(_ConnectConvType _WeConvTy, int k_Fea,
		int k_Scale, int k_Dep, int k_strConv, int k_padCon, float k_fac);
	void setNewRegul(_ConnectConvType _RegConvTy, int a_Fea,
		int a_Scale, int a_Dep, float a_fac);
	void setNewPool(_ConnectPoolType _CoPoolTy, int poolR);
	void setAct(_ConnectActType _CoActTy);
	void setLearn(_ConnectLearningType _CoLearningTy);
	void setMatrixKernelType(_ConnectMatrixType _CoMatrixTy);
	HyperTensor  *RegulatorKernel;
private:
	void buildTotalOutLayer();
	int controlConnect();
	void setNewKernel(int k_Fea, int k_He, int k_Wi, int k_Ch, float fac);
	void setNewActKernel(int k_Fea, int k_He, int k_Wi, int k_Ch, float fac);
	void setFCConnect(float fac);
	void setFCActivatorConnect();
	void catchIAllDim(Layer* inWeLaNum, Layer* totOutLaNu);
	void catchIAllDim(Layer* inWeLaNum, Layer* regulatorLay,Layer* totOutLaNu);
	void catchinWeigthLay(const Layer* inWeLaNum);
	void catchtotalOutLay(const Layer* totOutLaNu);
	void catchRegulatorLayer(const Layer* Regulator);
	void catchRegActivatorKernel(const HyperTensor* Activator);
	Layer * inWeigthLayPtr;
	Layer * regulatorLayPtr;
	Layer * totalOutLayPtr;
	_ConnectProcessType _RegProcesTyp;
	_ConnectConvType _RegConvTyp;
	_ConnectPoolType _RegPoolTyp;
	_ConnectActType _RegActTyp;
	_ConnectNormType _RegNormTyp;
	_ConnectLearningType _RegLearningTyp;
	_ConnectMatrixType _RegMatrixTyp;
	_ConnectProcessType _WeProcesTyp;
	_ConnectConvType _WeConvTyp;
	_ConnectPoolType _WePoolTyp;
	_ConnectActType _WeActTyp;
	_ConnectNormType _WeNormTyp;
	_ConnectLearningType _WeLearningTyp;
	_ConnectMatrixType _WeMatrixTyp;
	float momentum;
	float learning;
	float noise;
	float momentum2;
	float learning2;
	float noise2;
	float  randomizeFactorWe;
	float * kernelWeigthTensor;
	float * kernelWeigthParalelTensor;
	float * kernelWeigthPastTensor;
	HyperTensor  *biasConnect;
	float * biasTensor;
	float * biasParalelTensor;
	float * biasPastTensor;
	float  randomizeFactorReg;
	float * kernelActivTensor;
	float * kernelActivParalelTensor;
	float * kernelActivPastTensor;
	int convStrideX, convStrideY;
	int convPadX, convPadY;
	int poolSizeWe;
	int poolSizeReg;
	int k_WeFeature;
	int k_WeHeWiScale;
	int k_WeDepth;
	int a_RegFeature;
	int a_RegHeWiScale;
	int a_RegDepth;
	float * inputWeLayerTensor;
	float * inputWeLayerparalelTensor;
	float * RegulatorLayerTensor;
	float * RegulatorLayerParalelTensor;
	int  RegulatorLayNum;
	int  RegulatorLayAxis;
	int  RegulatorLaySamp;
	int  RegulatorLayFea;
	int  RegulatorLayHe;
	int  RegulatorLayWi;
	int  RegulatorLayCh;
	int  RegulatorLaySize;
	int  inWeigthLayNum;
	int  inWeigthLayAxis;
	int  inWeigthLaySamp;
	int  inWeigthLayFea;
	int  inWeigthLayHe;
	int  inWeigthLayWi;
	int  inWeigthLayCh;
	int  inWeigthLaySize;
	float * totalOutLayerTensor;
	float * totalOutLayerparalelTensor;
	int  totalOutLayNum;
	int  totalOutLayAxis;
	int  totalOutLaySamp;
	int  totalOutLayFea;
	int  totalOutLayHe;
	int  totalOutLayWi;
	int  totalOutLayCh;
	int  totalOutLaySize;
	bool isKernelHex;
	bool reShape;
	bool totalOutLayerNoConnected;
	bool isRegulatorConnect;
	bool cuBlass = true;

};

#endif


