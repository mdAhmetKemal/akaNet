#ifndef NET_H
#define NET_H
#define MaxLayer  30
//#include "Layer.h"
#include "Connect.h"
#include "EnumType.h"
#include "NetData.h"
#include "NetGL.h"
#include "ShowNetCuda.h"
#include <algorithm>
#include <time.h>

//#include "HyperTensor.h"

class Net
{
public:
	void setParameter(int epoch, int batch,
		float learningRate1, float Momentum1, float noiseSubJoin1,
		float learningRate2, float Momentum2, float noiseSubJoin2,
		float learningRate3);
	Net(float * inputData, float * targetData, int totalSampleNum, int samplePiece,
		int inHeigth, int inWidth, int inChanal, int resHeigth,
		int resWidth, int resChanal);
	void setLayerNum(int totalLayerN);
	void setConnectPool(int inputLa, int outLa, _ConnectPoolType conPoolTy,
		 int p_Scale);
	void setConnectConv(int inputLa, int outLa, _ConnectConvType conConTy,
		int k_Fea, int k_Scale, int k_Dep
		, int strideConv, int paddConv, float  fac);
	void setConnectConv(int inputLa, int regLa,int outLa, _ConnectConvType conConTy,
		int k_Fea, int k_Scale, int k_Dep
		, int strideConv, int paddConv,float k_fac,
		_ConnectConvType a_InpTy,
		int a_Fea , int a_Scale , int a_Dep,
		 float  a_fac);
	void trainNet();
	void testNet();
	void showNet(int Heigt, int Width);
	~Net();
	bool isOpenGl;
private:
	void Data_WiHeSa2SaHeWi(float *DataIn,int width,int heigth,int sample);
	void manuelKernel(float * kernelData, int kernelFea, int heigthWeigth, int chanal);
	void shuffleNetData();
	void shuffleGPUData();
	void loadAllDataGPU();
	void loadNet(int LeaSamPiecNum);
	void loadNetGPU(int LeaSamPiecNum);
	void feedWorkNet();
	void errorCalculateNet();
	void backPropagationNet();
	void changeWeigth();
	void buildInputtargetLayer();
	void cleanData();
	void cleanInterLayer();
	float * inputDataPtr;
	float * targetDataPtr;
	float *allInputDataOnGpu;
	float *shuffledInputDataOnGpu;
	float *allTargetDataOnGpu;
	float *shuffledTargetDataOnGpu;
	int  inputSampleTotal;
	int  inputSampleHeigth;
	int  inputSampleWidth;
	int  inputSampleChanal;
	int  targetSampleTotal;
	int  targetSampleHeigth;
	int  targetSampleWidth;
	int  targetSampleChanal;
	int  pieceTotalNum;
	int  LearningSamplePiecesNum;
	int  pieceSampSize;
	int  Epoch;
	int  Batch;
	float learningRate1;
	float learningRate2;
	float learningRate3;
	float learningMomentum1;
	float learningMomentum2;
	float noiseSubJoinRate1;
	float noiseSubJoinRate2;
	int  totalLayerNumNet;
	int  totalConnectNum;
	Layer * layerMap[MaxLayer];
	Connect * connectMap[MaxLayer][MaxLayer];
	int  connectWhereLayerMap[MaxLayer][MaxLayer];
	int ShowHeigth;
	int ShowWidth;
	float * errorArray;
	float * accuracyTest;
	float * accuracyTrain;
	
};

#endif


