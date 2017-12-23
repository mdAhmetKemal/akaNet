#ifndef CONNECT_CPP
#define CONNECT_CPP
#include "Connect.h"



Connect::Connect(Layer* inWeLaNum, Layer* totOutLaNu,
	_ConnectConvType _CoConvTy, _ConnectPoolType _CoPoTy,
	_ConnectActType _CoActTy, int k_Fea, int k_Scale, int k_Dep,
	int strideConv, int paddConv, float  fac, int poolR) :HyperTensor(1, 1, 5, 5, 3, fac)
{
	isRegulatorConnect = false;
	TensorRandom(fac);
	biasConnect = new HyperTensor(1, 1, 5, 5, 3,fac);
	biasConnect->addPastTensor();
	addPastTensor();
	catchIAllDim(inWeLaNum, totOutLaNu);
	_WePoolTyp = _NOPOOL;
	_WeConvTyp = _NOCONV;
	//setNewConv(_CoConvTy, k_Fea, k_Scale, k_Dep, strideConv, paddConv, fac);
	//setNewPool(_CoPoTy, poolR);
	//setAct(_CoActTy);
	randomizeFactorWe = fac;
	poolSizeWe = poolR;
}
Connect::~Connect() {
	HyperTensor::~HyperTensor();
	biasConnect->~HyperTensor();
	if (isRegulatorConnect)
		RegulatorKernel->~HyperTensor();
}
void Connect::catchRegActivatorKernel(const HyperTensor* Activator){
	kernelActivTensor = Activator->Tensor;
	kernelActivParalelTensor = Activator->paralelTensor;
	kernelActivPastTensor = Activator->pastTensor;
}
void Connect::catchIAllDim(Layer* inWeLaNum, Layer* totOutLaNu){
	biasTensor = biasConnect->Tensor;
	biasParalelTensor = biasConnect->paralelTensor;
	biasPastTensor = biasConnect->pastTensor;
	inWeigthLayPtr = inWeLaNum;
	totalOutLayPtr = totOutLaNu;
	catchtotalOutLay(totOutLaNu);
	catchinWeigthLay(inWeLaNum);
}
void Connect::catchIAllDim(Layer* inWeLaNum, Layer* regulatorLay, Layer* totOutLaNu){
	
	biasTensor = biasConnect->Tensor;
	biasParalelTensor = biasConnect->paralelTensor;
	biasPastTensor = biasConnect->pastTensor;
	inWeigthLayPtr = inWeLaNum;
	regulatorLayPtr = regulatorLay;
	totalOutLayPtr = totOutLaNu;
	catchtotalOutLay(totOutLaNu);
	catchRegulatorLayer(regulatorLay);
	catchinWeigthLay(inWeLaNum);
}
void Connect::catchRegulatorLayer(const Layer* Regulator){
	RegulatorLayerTensor = Regulator->Tensor;
	RegulatorLayerParalelTensor = Regulator->paralelTensor;
	RegulatorLayNum = Regulator->layerStepNum;
	RegulatorLayAxis = Regulator->axis;
	RegulatorLaySamp = Regulator->hTsample;
	RegulatorLayFea = Regulator->hTfeatureNum;
	RegulatorLayHe = Regulator->hTheigth;
	RegulatorLayWi = Regulator->hTwidth;
	RegulatorLayCh = Regulator->hTchanal;
	RegulatorLaySize = Regulator->sizePiksel;
}
void Connect::catchinWeigthLay(const Layer* inWeLaNum){
	biasTensor = biasConnect->Tensor;
	biasParalelTensor = biasConnect->paralelTensor;
	biasPastTensor = biasConnect->pastTensor;
	inputWeLayerTensor = inWeLaNum->Tensor;
	inputWeLayerparalelTensor = inWeLaNum->paralelTensor;
	inWeigthLayAxis = inWeLaNum->axis;
	inWeigthLayNum = inWeLaNum->layerStepNum;
	inWeigthLaySamp = inWeLaNum->hTsample;
	inWeigthLayFea = inWeLaNum->hTfeatureNum;
	inWeigthLayHe = inWeLaNum->hTheigth;
	inWeigthLayWi = inWeLaNum->hTwidth;
	inWeigthLayCh = inWeLaNum->hTchanal;
	inWeigthLaySize = inWeLaNum->sizePiksel;
}
void Connect::catchtotalOutLay(const Layer* totOutLaNu){
	totalOutLayerTensor = totOutLaNu->Tensor;
	totalOutLayerparalelTensor = totOutLaNu->paralelTensor;
	totalOutLayNum = totOutLaNu->layerStepNum;
	totalOutLayAxis = totOutLaNu->axis;
	totalOutLaySamp = totOutLaNu->hTsample;
	totalOutLayFea = totOutLaNu->hTfeatureNum;
	totalOutLayHe = totOutLaNu->hTheigth;
	totalOutLayWi = totOutLaNu->hTwidth;
	totalOutLayCh = totOutLaNu->hTchanal;
	totalOutLaySize = totOutLaNu->sizePiksel;
	totalOutLayerNoConnected = totOutLaNu->isNoConnected;
}
void Connect::setLearn(_ConnectLearningType _CoLearningTy){
	switch (_CoLearningTy)
	{
	case _STABIL:
		_WeLearningTyp = _STABIL;
		break;
	case _DYNAMIC:
		_WeLearningTyp = _DYNAMIC;
		break;
	case _OTHER:
		_WeLearningTyp = _DYNAMIC;
		break;
	default:
		_WeLearningTyp = _DYNAMIC;
		break;
	}
}
void Connect::setAct(_ConnectActType _CoActTy){
	switch (_CoActTy)
	{
	case _NOACTV:
		_WeActTyp = _NOACTV;
		break;
	case _RELU:
		_WeActTyp = _RELU;
		break;
	case _ELU:
		_WeActTyp = _ELU;
		break;
	case _LRELU:
		_WeActTyp = _LRELU;
		break;
	case _LINEER:
		_WeActTyp = _LINEER;
		break;
	case _SIGM:
		_WeActTyp = _SIGM;
		break;
	case _TANH:
		_WeActTyp = _TANH;
		break;
	case _ARCTAN:
		_WeActTyp = _ARCTAN;
		break;
	case _STEP:
		_WeActTyp = _STEP;
		break;
	case _SOFTPLUS:
		_WeActTyp = _SOFTPLUS;
		break;
	default:
		_WeActTyp = _RELU;
		break;
	}

}

void Connect::setMatrixKernelType(_ConnectMatrixType _CoMatrixTy){
	switch (_CoMatrixTy)
	{
	case _RECTANGULARMAT:
		_WeMatrixTyp = _CoMatrixTy;
		break;
	case _HEXAGONALMAT:
		_WeMatrixTyp = _CoMatrixTy;
		break;
	default:
		_WeMatrixTyp = _RECTANGULARMAT;
		break;
	}
	controlConnect();
}
/*
void Connect::setConnectProcess(_ConnectProcessType _CoProcesTyp){
switch (_CoProcesTyp)
{
case _NOPROCESS:
_WeConvTyp = _NOCONV; _WePoolTyp = _NOPOOL; _WeActTyp = _NOACTV; _CoNormTyp = _NONORM; _WeLearningTyp = _STABIL;
break;
case _COPOAC:
_WeConvTyp = _CONV2D; _WePoolTyp = _AVRPOOL; _WeActTyp = _RELU; _CoNormTyp = _NONORM; _WeLearningTyp = _DYNAMIC;
break;
case _CONV:
_WeConvTyp = _CONV2D; _WePoolTyp = _NOPOOL; _WeActTyp = _NOACTV; _CoNormTyp = _NONORM; _WeLearningTyp = _DYNAMIC;
break;
case _POOL:
_WeConvTyp = _NOCONV; _WePoolTyp = _AVRPOOL; _WeActTyp = _NOACTV; _CoNormTyp = _NONORM; _WeLearningTyp = _STABIL;
break;
case _ACTIV:
_WeConvTyp = _NOCONV; _WePoolTyp = _NOPOOL; _WeActTyp = _RELU; _CoNormTyp = _NONORM; _WeLearningTyp = _STABIL;
break;
case _FULLYCON:
_WeConvTyp = _FULLYLAYERCON; _WePoolTyp = _NOPOOL; _WeActTyp = _RELU; _CoNormTyp = _NONORM; _WeLearningTyp = _DYNAMIC;
break;
case _ADDP:
_WeConvTyp = _NOCONV; _WePoolTyp = _NOPOOL; _WeActTyp = _NOACTV; _CoNormTyp = _NONORM; _WeLearningTyp = _STABIL;
break;
case _EXTRACT:
_WeConvTyp = _NOCONV; _WePoolTyp = _NOPOOL; _WeActTyp = _NOACTV; _CoNormTyp = _NONORM; _WeLearningTyp = _STABIL;
break;
default:
_WeConvTyp = _NOCONV; _WePoolTyp = _NOPOOL; _WeActTyp = _NOACTV; _CoNormTyp = _NONORM; _WeLearningTyp = _STABIL;
break;
}
setNewConv(_WeConvTyp);
setNewPool(_WePoolTyp);
setAct(_WeActTyp);
setLearn(_WeLearningTyp);
} */
void Connect::setNewPool(_ConnectPoolType _CoPoolTy, int p_Scale){
	_WeConvTyp = _NOCONV;
	switch (_CoPoolTy)
	{
	case _NOPOOL:
		_WePoolTyp = _NOPOOL;
		break;
	case _MAXPOOL:
		_WePoolTyp = _MAXPOOL;
		break;
	case _AVRPOOL:
		_WePoolTyp = _AVRPOOL;
		break;
	case _GAUSPOOL:
		_WePoolTyp = _GAUSPOOL;
		break;
	default:
		_WePoolTyp = _NOPOOL;
		break;
	}
	if (_CoPoolTy == _NOPOOL)
		poolSizeWe = 1;
	else
		poolSizeWe = p_Scale;
	biasConnect->changeHyperTensor(1, 1, 1, 1, 1, 1);
	changeHyperTensor(1, 1, 1, 1, 1, 1);
	if (totalOutLayNum != 0){
		buildTotalOutLayer();
	}

}
void Connect::setNewConv(_ConnectConvType _CoConvTy = _CONV2D, int k_Fea = 12, int k_Scale = 3, int k_Dep = 1
	, int strideConv = 1, int paddConv = 1, float  fac = 0.01){
	convStrideX = convStrideY = strideConv;
	convPadX = convPadY = paddConv;
	k_WeFeature = k_Fea;
	k_WeHeWiScale = k_Scale;
	k_WeDepth = k_Dep;
	randomizeFactorWe = fac;
	//delete Tensor;
	//delete paralelTensor;
	_WePoolTyp = _NOPOOL;
	_WeConvTyp = _CoConvTy;
	switch (_WeConvTyp)
	{
	case _NOCONV:
		break;
	case _CONV1D:
		setNewKernel(k_Fea, 1, 1, 1, fac);
		break;
	case _CONV2D:
		setNewKernel(k_Fea, k_Scale, k_Scale, 1, fac);
		break;
	case _CONV2DtoCH:
		setNewKernel(k_Fea, k_Scale, k_Scale, 1, fac);
		break;
	case _CONV3C:
		setNewKernel(k_Fea, k_Scale, k_Scale, inWeigthLayCh, fac);
		break;
	case _CONV3F:
		setNewKernel(k_Fea, k_Scale, k_Scale, inWeigthLayFea, fac);
		break;
	case _CONV3A:
		setNewKernel(k_Fea, k_Scale, k_Scale, inWeigthLayCh*inWeigthLayFea, fac);
		break;
	case _CONV_1C:
		setNewKernel(k_Fea, 1, 1, inWeigthLayCh, fac);
		break;
	case _CONV_1F:
		setNewKernel(k_Fea, 1, 1, inWeigthLayFea, fac);
		break;
	case _CONV_1A:
		setNewKernel(k_Fea, 1, 1, inWeigthLayCh*inWeigthLayFea, fac);
		break;
	case _FULLYLAYERCON:
		setFCConnect(fac);
		break;
	default:
		break;
	}
}
void Connect::setNewKernel(int k_Fea, int k_He, int k_Wi, int k_Ch, float fac){
	biasConnect->changeHyperTensor(1, 1, 1, 1, k_Fea, fac);
	biasTensor = biasConnect->Tensor;
	biasParalelTensor = biasConnect->paralelTensor;
	biasPastTensor = biasConnect->pastTensor;
	//printf("\n ConnectDe:newKernel fac: %.4f", fac);
	changeHyperTensor(1, k_Fea, k_He, k_Wi, k_Ch, fac);
	kernelWeigthTensor = Tensor;
	kernelWeigthParalelTensor = paralelTensor;
	kernelWeigthPastTensor = pastTensor;
	if (totalOutLayNum != 0){
		buildTotalOutLayer();
	}
}
void Connect::setFCConnect(float fac){
	biasConnect->changeHyperTensor(1, 1, 1, 1, totalOutLaySize / totalOutLaySamp, fac);
	biasConnect->TensorRandom(randomizeFactorWe);
	biasTensor = biasConnect->Tensor;
	biasParalelTensor = biasConnect->paralelTensor;
	biasPastTensor = biasConnect->pastTensor;
	changeHyperTensor(1, 1, 1, inWeigthLaySize / inWeigthLaySamp, totalOutLaySize / totalOutLaySamp, fac);
	//printf("\n ConnectDe:randtensor fac: %.4f", fac);
	TensorRandom(fac);
	kernelWeigthTensor = Tensor;
	kernelWeigthParalelTensor = paralelTensor;
	kernelWeigthPastTensor = pastTensor;
	if (totalOutLayNum != 0){
		buildTotalOutLayer();
	}
}
void Connect::buildTotalOutLayer(){
	if (totalOutLayerNoConnected){
		if (_WeConvTyp == _CONV2D || _WeConvTyp == _CONV2DtoFE){
			k_WeFeature = hTfeatureNum;  // hT ile baþlayan her þeyi silmek gerekebilir 
			k_WeHeWiScale = hTheigth;
			k_WeDepth = hTchanal;
			int newHeigth = (inWeigthLayHe - k_WeHeWiScale + 2 * convPadX) / convStrideX + 1;
			int newWidth = (inWeigthLayWi - k_WeHeWiScale + 2 * convPadY) / convStrideY + 1;
			int newOutLayFeature = inWeigthLayFea*k_WeFeature;
			int newOutLayCh = inWeigthLayCh;
			int newOutLaySamp = inWeigthLaySamp;
			totalOutLayPtr->changeHyperTensor(newOutLaySamp, newOutLayFeature, newHeigth / poolSizeWe, newWidth / poolSizeWe, newOutLayCh);
			totalOutLayPtr->isNoConnected = false;
			catchtotalOutLay(totalOutLayPtr);
		}
		if (_WeConvTyp == _CONV2DtoCH){
			k_WeFeature = hTfeatureNum;
			k_WeHeWiScale = hTheigth;
			k_WeDepth = hTchanal;
			int newHeigth = (inWeigthLayHe - k_WeHeWiScale + 2 * convPadX) / convStrideX + 1;
			int newWidth = (inWeigthLayWi - k_WeHeWiScale + 2 * convPadY) / convStrideY + 1;
			int newOutLayFeature = inWeigthLayFea;
			int newOutLayCh = inWeigthLayCh*k_WeFeature;
			int newOutLaySamp = inWeigthLaySamp;
			totalOutLayPtr->changeHyperTensor(newOutLaySamp, newOutLayFeature, newHeigth / poolSizeWe, newWidth / poolSizeWe, newOutLayCh);
			totalOutLayPtr->isNoConnected = false;
			catchtotalOutLay(totalOutLayPtr);
		}
		if (_WeConvTyp == _NOCONV && _WePoolTyp==_NOPOOL && poolSizeWe==1){
			int newHeigth = inWeigthLayHe;
			int newWidth = inWeigthLayWi;
			int newOutLayFeature = inWeigthLayFea;
			int newOutLayCh = inWeigthLayCh;
			int newOutLaySamp = inWeigthLaySamp;
			totalOutLayPtr->changeHyperTensor(newOutLaySamp, newOutLayFeature, newHeigth , newWidth , newOutLayCh);
			totalOutLayPtr->isNoConnected = false;
			catchtotalOutLay(totalOutLayPtr);
		}
		else if (_WeConvTyp == _NOCONV && _WePoolTyp != _NOPOOL && poolSizeWe > 1){
			int newHeigth = inWeigthLayHe / poolSizeWe;
			int newWidth = inWeigthLayWi / poolSizeWe;
			int newOutLayFeature = inWeigthLayFea;
			int newOutLayCh = inWeigthLayCh;
			int newOutLaySamp = inWeigthLaySamp;
			totalOutLayPtr->changeHyperTensor(newOutLaySamp, newOutLayFeature, newHeigth , newWidth, newOutLayCh);
			totalOutLayPtr->isNoConnected = false;
			catchtotalOutLay(totalOutLayPtr);
		}
		if (_WeConvTyp == _CONV_1C){
			k_WeFeature = hTfeatureNum;
			k_WeHeWiScale = hTheigth;
			k_WeDepth = hTchanal;
			int newHeigth = inWeigthLayHe;
			int newWidth = inWeigthLayWi;
			int newOutLayFeature = inWeigthLayFea;
			int newOutLayCh = k_WeFeature;
			int newOutLaySamp = inWeigthLaySamp;
			totalOutLayPtr->changeHyperTensor(newOutLaySamp, newOutLayFeature, newHeigth / poolSizeWe, newWidth / poolSizeWe, newOutLayCh);
			totalOutLayPtr->isNoConnected = false;
			catchtotalOutLay(totalOutLayPtr);
		}
		if (_WeConvTyp == _CONV_1F){
			k_WeFeature = hTfeatureNum;
			k_WeHeWiScale = hTheigth;
			k_WeDepth = hTchanal;
			int newHeigth = inWeigthLayHe;
			int newWidth = inWeigthLayWi;
			int newOutLayFeature = k_WeFeature;
			int newOutLayCh = inWeigthLayCh;
			int newOutLaySamp = inWeigthLaySamp;
			totalOutLayPtr->changeHyperTensor(newOutLaySamp, newOutLayFeature, newHeigth / poolSizeWe, newWidth / poolSizeWe, newOutLayCh);
			totalOutLayPtr->isNoConnected = false;
			catchtotalOutLay(totalOutLayPtr);
		}
		if (_WeConvTyp == _CONV_1A){
			k_WeFeature = hTfeatureNum;
			k_WeHeWiScale = hTheigth;
			k_WeDepth = hTchanal;
			int newHeigth = inWeigthLayHe;
			int newWidth = inWeigthLayWi;
			int newOutLayFeature = k_WeFeature;
			int newOutLayCh = 1;
			int newOutLaySamp = inWeigthLaySamp;
			totalOutLayPtr->changeHyperTensor(newOutLaySamp, newOutLayFeature, newHeigth / poolSizeWe, newWidth / poolSizeWe, newOutLayCh);
			totalOutLayPtr->isNoConnected = false;
			catchtotalOutLay(totalOutLayPtr);
		}
		if (_WeConvTyp == _CONV3C){
			k_WeFeature = hTfeatureNum;
			k_WeHeWiScale = hTheigth;
			k_WeDepth = hTchanal;
			int newHeigth = (inWeigthLayHe - k_WeHeWiScale + 2 * convPadX) / convStrideX + 1;
			int newWidth = (inWeigthLayWi - k_WeHeWiScale + 2 * convPadY) / convStrideY + 1;
			int newOutLayFeature = inWeigthLayFea;
			int newOutLayCh = k_WeFeature;
			int newOutLaySamp = inWeigthLaySamp;
			totalOutLayPtr->changeHyperTensor(newOutLaySamp, newOutLayFeature, newHeigth / poolSizeWe, newWidth / poolSizeWe, newOutLayCh);
			totalOutLayPtr->isNoConnected = false;
			catchtotalOutLay(totalOutLayPtr);
		}
		if (_WeConvTyp == _CONV3F){
			k_WeFeature = hTfeatureNum;
			k_WeHeWiScale = hTheigth;
			k_WeDepth = hTchanal;
			int newHeigth = (inWeigthLayHe - k_WeHeWiScale + 2 * convPadX) / convStrideX + 1;
			int newWidth = (inWeigthLayWi - k_WeHeWiScale + 2 * convPadY) / convStrideY + 1;
			int newOutLayFeature = k_WeFeature;
			int newOutLayCh = inWeigthLayCh;
			int newOutLaySamp = inWeigthLaySamp;
			totalOutLayPtr->changeHyperTensor(newOutLaySamp, newOutLayFeature, newHeigth / poolSizeWe, newWidth / poolSizeWe, newOutLayCh);
			totalOutLayPtr->isNoConnected = false;
			catchtotalOutLay(totalOutLayPtr);
		}
		if (_WeConvTyp == _CONV3A){
			k_WeFeature = hTfeatureNum;
			k_WeHeWiScale = hTheigth;
			k_WeDepth = hTchanal;
			int newHeigth = (inWeigthLayHe - k_WeHeWiScale + 2 * convPadX) / convStrideX + 1;
			int newWidth = (inWeigthLayWi - k_WeHeWiScale + 2 * convPadY) / convStrideY + 1;
			int newOutLayFeature = k_WeFeature;
			int newOutLayCh = 1;
			int newOutLaySamp = inWeigthLaySamp;
			totalOutLayPtr->changeHyperTensor(newOutLaySamp, newOutLayFeature, newHeigth / poolSizeWe, newWidth / poolSizeWe, newOutLayCh);
			totalOutLayPtr->isNoConnected = false;
			catchtotalOutLay(totalOutLayPtr);
		}
		if (_WeConvTyp == _FULLYLAYERCON){
			k_WeFeature = hTfeatureNum;
			k_WeHeWiScale = hTheigth;
			k_WeDepth = hTchanal;
			/*int newHeigth = (inWeigthLayHe - k_WeHeWiScale + 2 * convPadX) / convStrideX + 1;
			int newWidth = (inWeigthLayWi - k_WeHeWiScale + 2 * convPadY) / convStrideY + 1;
			int newOutLayFeature = k_WeFeature;
			int newOutLayCh = 1;
			int newOutLaySamp = inWeigthLaySamp;
			*/
			totalOutLayPtr->changeHyperTensor(inWeigthLaySamp, hTfeatureNum, hTheigth, hTheigth, hTchanal);
			totalOutLayPtr->isNoConnected = false;
			catchtotalOutLay(totalOutLayPtr);
		}
	}

}
int  Connect::controlConnect(){
	if (totalOutLayerNoConnected){
		buildTotalOutLayer();
	}
	/*
	if (inWeigthLayNum >= totalOutLayNum)
	return 1;
	else if (	inWeigthLaySamp != totalOutLaySamp)
	return 2;
	else if (_WeConvTyp == _CONV2D){
	return 3;
	}
	else
	return 0;//hiç yanlýþ yoksa   */
	return 0;
}

void Connect::ErrorConnect(){
	if (inWeigthLaySize == totalOutLaySize){
		//**********CrEn garip bir multiclass cross entropy hata belirliyor her label için
		errorSoftmaxCrEn(inputWeLayerTensor, totalOutLayerTensor, inputWeLayerparalelTensor,
			inWeigthLaySamp, inWeigthLayFea, inWeigthLayHe, inWeigthLayWi, inWeigthLayCh);
		//ErrorCalculateCu(inputWeLayerTensor, totalOutLayerTensor, inputWeLayerparalelTensor,
		//inWeigthLaySamp,inWeigthLayFea, inWeigthLayHe, inWeigthLayWi, inWeigthLayCh);
	}

	//iþte burasý 
	/*   perþembe günü burayý alacaðým ve son ttoal layer ile sýfýr numaralý layer ile olan connect ise
	doðrulama yaptýktan sonra klasik basit error tanýlama fonskiyonunu gpuda çalýþtýracaðým */
};
void Connect::FeedWorkConnect(){
	if (_WeConvTyp == _FULLYLAYERCON){
		if (isRegulatorConnect){
			FullyWeigthFullyRegulatorCu(
				inputWeLayerTensor, inWeigthLaySamp, inWeigthLayFea,
				inWeigthLayHe, inWeigthLayWi, inWeigthLayCh, 
				RegulatorLayerTensor, RegulatorLayFea, RegulatorLayHe,
				RegulatorLayWi, RegulatorLayCh,
				totalOutLayerTensor,totalOutLayFea, totalOutLayHe,
				totalOutLayWi, totalOutLayCh,
				kernelWeigthTensor,kernelWeigthPastTensor,
				kernelActivTensor,biasTensor);
		}
		else{
			if (cuBlass)
			{
			}
			else{
			FullyLayerFeedworkCu(inputWeLayerTensor, inWeigthLaySamp, inWeigthLayFea,
				inWeigthLayHe, inWeigthLayWi, inWeigthLayCh,
				totalOutLayerTensor, totalOutLayFea, totalOutLayHe, totalOutLayWi, totalOutLayCh,
				kernelWeigthTensor,biasTensor);
			}
		}	
	}
	if (_WeConvTyp == _CONV2D){
		if (false){ //isRegulatorConnect){
		/*	FullyWeigthFullyRegulatorCu(
				inputWeLayerTensor, inWeigthLaySamp, inWeigthLayFea,
				inWeigthLayHe, inWeigthLayWi, inWeigthLayCh,
				RegulatorLayerTensor, RegulatorLayFea, RegulatorLayHe,
				RegulatorLayWi, RegulatorLayCh,
				totalOutLayerTensor, totalOutLayFea, totalOutLayHe,
				totalOutLayWi, totalOutLayCh,
				kernelWeigthTensor, kernelWeigthPastTensor,
				kernelActivTensor, biasTensor);
				*/
		}
		else{//conv2d_Feed 
			conv2dV2Feed(inputWeLayerTensor, inWeigthLaySamp, inWeigthLayFea,
				inWeigthLayHe, inWeigthLayWi, inWeigthLayCh,
				totalOutLayerTensor, totalOutLayFea, totalOutLayHe, totalOutLayWi, totalOutLayCh,
				kernelWeigthTensor, k_WeFeature, k_WeHeWiScale, k_WeDepth, convStrideX, convStrideY,
				convPadX, convPadY, biasTensor);
		}
	}
	if (_WeConvTyp == _NOCONV){
		if (_WePoolTyp == _MAXPOOL){
			poolMax2d(inWeigthLayPtr, totalOutLayPtr, poolSizeWe);
		}
		if (_WePoolTyp == _AVRPOOL){
			poolAvg2d(inWeigthLayPtr, totalOutLayPtr, poolSizeWe);
		}


		if (_WePoolTyp == _GAUSPOOL){}
	}
};

void Connect::backPropagationConnect(){
	if (_WeConvTyp == _FULLYLAYERCON){
		if (isRegulatorConnect){
			FullyWeigthFullyRegBackProCu(inputWeLayerparalelTensor, inWeigthLaySamp, inWeigthLayFea, inWeigthLayHe,
				inWeigthLayWi, inWeigthLayCh, RegulatorLayerParalelTensor, RegulatorLayFea, RegulatorLayHe,
				RegulatorLayWi, RegulatorLayCh, totalOutLayerparalelTensor, totalOutLayFea, totalOutLayHe, totalOutLayWi,
				totalOutLayCh, kernelWeigthTensor,kernelWeigthPastTensor, kernelActivTensor);
		}
		else{
	
			//error en son layerda (2 num) error buconnect için total out layerda mevcut  mevcut 
			FullyLayerBackPropagationCu(inputWeLayerparalelTensor, inWeigthLaySamp, inWeigthLayFea,
				inWeigthLayHe, inWeigthLayWi, inWeigthLayCh,
				totalOutLayerparalelTensor, totalOutLayFea, totalOutLayHe, totalOutLayWi, totalOutLayCh,
				kernelWeigthTensor);
		// printf("\n fully connect backPro");
		}
	}
	if (_WeConvTyp == _CONV2D){
		if (false){ //isRegulatorConnect){
			/*	FullyWeigthFullyRegulatorCu(
			inputWeLayerTensor, inWeigthLaySamp, inWeigthLayFea,
			inWeigthLayHe, inWeigthLayWi, inWeigthLayCh,
			RegulatorLayerTensor, RegulatorLayFea, RegulatorLayHe,
			RegulatorLayWi, RegulatorLayCh,
			totalOutLayerTensor, totalOutLayFea, totalOutLayHe,
			totalOutLayWi, totalOutLayCh,
			kernelWeigthTensor, kernelWeigthPastTensor,
			kernelActivTensor, biasTensor);
			*/
		}
		else{//conv2d_Back
			conv2d_Back(inputWeLayerparalelTensor, inWeigthLaySamp, inWeigthLayFea,
				inWeigthLayHe, inWeigthLayWi, inWeigthLayCh,
				totalOutLayerparalelTensor, totalOutLayFea, totalOutLayHe, totalOutLayWi, totalOutLayCh,
				kernelWeigthTensor, k_WeFeature, k_WeHeWiScale, k_WeDepth, convStrideX, convStrideY,
				convPadX, convPadY, biasTensor);
		//	printf("\n  input %d   out %d ", inWeigthLayNum,totalOutLayNum);
		}
	}
	if (_WeConvTyp == _NOCONV){
		if (_WePoolTyp == _MAXPOOL){
			poolMax2dBack(inWeigthLayPtr, totalOutLayPtr, poolSizeWe);
		}
		if (_WePoolTyp == _AVRPOOL){
			poolAvg2dBack(inWeigthLayPtr, totalOutLayPtr, poolSizeWe);
		}


		if (_WePoolTyp == _GAUSPOOL){}
	}
};
void Connect::changeWeigthConnect(float learn,float momen,float nois,
	float learn2,float momen2,float nois2){
	noise = nois;
	momentum = momen;
	learning = learn;
	momentum2 =momen2;
	learning2=learn2;
	noise2=nois2;
	if (_WeConvTyp == _FULLYLAYERCON){
		if (isRegulatorConnect){
			ChangeFullyWeFullyRegCu(inputWeLayerTensor, inWeigthLaySamp,
				inWeigthLayFea, inWeigthLayHe,  inWeigthLayWi,  inWeigthLayCh,
				RegulatorLayerTensor, RegulatorLayFea, RegulatorLayHe,
				RegulatorLayWi, RegulatorLayCh,
				totalOutLayerparalelTensor, totalOutLayFea, totalOutLayHe, totalOutLayWi, totalOutLayCh,
				kernelWeigthTensor, kernelWeigthParalelTensor,kernelWeigthPastTensor,  kernelActivTensor, kernelActivParalelTensor, biasTensor,
				biasParalelTensor,learning, momentum,noise,learning2,momentum2,noise2);

		}
		else{
			
			//error en son layerda (2 num) error buconnect için total out layerda mevcut  mevcut 
			ChangeFullyWeigthCu( inputWeLayerTensor, inWeigthLaySamp,
				inWeigthLayFea,inWeigthLayHe, inWeigthLayWi, inWeigthLayCh,
				totalOutLayerparalelTensor, totalOutLayFea, totalOutLayHe, totalOutLayWi, totalOutLayCh,
				kernelWeigthTensor, kernelWeigthParalelTensor, biasTensor, biasParalelTensor,learning, momentum);

		}
	}
	if (_WeConvTyp == _CONV2D){
		if (false){
			ChangeFullyWeFullyRegCu(inputWeLayerTensor, inWeigthLaySamp,
				inWeigthLayFea, inWeigthLayHe, inWeigthLayWi, inWeigthLayCh,
				RegulatorLayerTensor, RegulatorLayFea, RegulatorLayHe,
				RegulatorLayWi, RegulatorLayCh,
				totalOutLayerparalelTensor, totalOutLayFea, totalOutLayHe, totalOutLayWi, totalOutLayCh,
				kernelWeigthTensor, kernelWeigthParalelTensor, kernelWeigthPastTensor, kernelActivTensor, kernelActivParalelTensor, biasTensor,
				biasParalelTensor, learning, momentum, noise, learning2, momentum2, noise2);

		}
		else{

			//error en son layerda (2 num) error buconnect için total out layerda mevcut  mevcut 
			conv2d_Update(inputWeLayerTensor, inWeigthLaySamp,
				inWeigthLayFea, inWeigthLayHe, inWeigthLayWi, inWeigthLayCh,
				totalOutLayerparalelTensor, totalOutLayFea, totalOutLayHe, totalOutLayWi, totalOutLayCh,
				kernelWeigthTensor, kernelWeigthParalelTensor, k_WeFeature, k_WeHeWiScale, k_WeDepth, convStrideX, convStrideY,
				convPadX, convPadY, biasTensor, biasParalelTensor, learning, momentum);

		}

	}

};

#endif