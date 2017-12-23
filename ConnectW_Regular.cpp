#ifndef CONNECTWREGULAR_CPP
#define CONNECTWREGULAR_CPP
#include "Connect.h"
Connect::Connect(Layer* inWeLaNum, Layer* regLaNum, Layer* OutLaNu,
	_ConnectConvType _WeConvTy, _ConnectPoolType _WePoTy,
	_ConnectActType _WeActTy, int k_Fea, int k_Scale, int k_Dep
	, int k_strideConv, int k_paddConv, float  k_fac, int k_poolR,
	_ConnectConvType _ReConvTy, _ConnectPoolType _RePoTy,
	_ConnectActType _ReActTy, int a_Fea, int a_Scale, int a_Dep
	, int a_strideConv, int a_paddConv, float  a_fac, int a_poolR) :HyperTensor(1, 1, 5, 5, 3, 0.01f)
{
	isRegulatorConnect = true;
	regLaNum->TensorRandom(k_fac);
	biasConnect = new HyperTensor(1, 1, 5, 5, 3, 0.01f);
	biasConnect->TensorRandom(k_fac);
	RegulatorKernel = new HyperTensor(1, 1, 5, 5, 3, 0.01f);
	RegulatorKernel->addPastTensor();
	catchRegActivatorKernel(RegulatorKernel);
	addPastTensor();
	RegulatorKernel->TensorRandom(0.01f);
	TensorRandom(k_fac);
	catchIAllDim(inWeLaNum, regLaNum, OutLaNu);
	randomizeFactorWe = k_fac;
	randomizeFactorReg = a_fac;
	poolSizeWe = k_poolR;
}
void Connect::setNewRegul(_ConnectConvType _RegConvTy=_CONV_1A, int a_Fea=1,
	int a_Scale=1, int a_Dep=1,float a_fac=1){
	a_RegFeature = a_Fea;
	a_RegHeWiScale = a_Scale;
	a_RegDepth = a_Dep;
	randomizeFactorWe = a_fac;
	_RegConvTyp = _RegConvTy;
	switch (_RegConvTyp)
	{
	case _NOCONV:
		break;
	case _CONV1D:
		setNewActKernel(a_Fea, 1, 1, 1, a_fac);
		break;
	case _CONV2D:
		setNewActKernel(a_Fea, a_Scale, a_Scale, 1, a_fac);
		break;
	case _CONV2DtoCH:
		setNewActKernel(a_Fea, a_Scale, a_Scale, 1, a_fac);
		break;
	case _CONV3C:
		setNewActKernel(a_Fea, a_Scale, a_Scale, inWeigthLayCh, a_fac);
		break;
	case _CONV3F:
		setNewActKernel(a_Fea, a_Scale, a_Scale, inWeigthLayFea, a_fac);
		break;
	case _CONV3A:
		setNewActKernel(a_Fea, a_Scale, a_Scale, inWeigthLayCh*inWeigthLayFea, a_fac);
		break;
	case _CONV_1C:
		setNewActKernel(a_Fea, 1, 1, inWeigthLayCh, a_fac);
		break;
	case _CONV_1F:
		setNewActKernel(a_Fea, 1, 1, inWeigthLayFea, a_fac);
		break;
	case _CONV_1A:
		setNewActKernel(a_Fea, 1, 1, inWeigthLayCh*inWeigthLayFea, a_fac);
		break;
	case _FULLYLAYERCON:
		setFCActivatorConnect();
		break;
	default:
		break;
	}
};
void Connect::setFCActivatorConnect(){
	
	biasConnect->changeHyperTensor(1, 1, 1, 1, totalOutLaySize / totalOutLaySamp, 0.01f);
	biasTensor = biasConnect->Tensor;
	biasParalelTensor = biasConnect->paralelTensor;
	biasPastTensor = biasConnect->pastTensor;
	

	RegulatorKernel->changeHyperTensor(1, 1, RegulatorLaySize/RegulatorLaySamp,
		inWeigthLaySize/inWeigthLaySamp, totalOutLaySize/totalOutLaySamp, 0.01f);
	kernelActivTensor = RegulatorKernel->Tensor;
	kernelActivParalelTensor = RegulatorKernel->paralelTensor;
	kernelActivPastTensor = RegulatorKernel->pastTensor;
	buildTotalOutLayer();
};
void Connect::setNewActKernel(int a_Fea, int a_He, int a_Wi, int a_Ch, float a_fac){

	RegulatorKernel->changeHyperTensor(1, a_Fea, a_He, a_Wi, a_Ch, a_fac);
	kernelActivTensor = RegulatorKernel->Tensor;
	kernelActivParalelTensor = RegulatorKernel->paralelTensor;
	kernelActivPastTensor = RegulatorKernel->pastTensor;

}
#endif