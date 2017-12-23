#ifndef NETCMD_H
#define NETCMD_H
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include "EnumType.h"
#include "Net.h"
#include "NetData.h"
using namespace std;
struct ConnectSpecs
{
	int inputLayNum,regulatorLayNum,outLayNum;
	_ConnectConvType inputConnect, regulatorConnect;
	int kernelFea, kernelScale, kernelDepth, kernelRanFac;
	int activatorFea, activatorScale, activatorDepth, activatorRanFac;
	int strideConv, PaddConv;
};
struct NetSpecs
{
	float *inputDataPtr;
	float *outDataPtr;
	int allSampleNum;
	int samplePiece;
	int inHeigth, inWeigth, inDepth;
	int outHeigth, outWeigth, outDepth;
	int totalLayerNum, totalConnectNum;
	int epoch;
	float learningRate1, learningRate2, learningRate3;
	float momentum1, momentum2;
	float noise1, noise2;
	bool netShow;
	int screenHeWe;
	ConnectSpecs * ConnectList;
};
class NetCmd
{
public:
	NetCmd();
	~NetCmd();
private:
	void loadCommand(string filename);
	string preProcesLine(string sampleStr);
	string dataFileName;
	string dataSpecsName;
	NetSpecs cmdNetSpecs;
	//Net cmdNet;
	NetData * file;
	//void standByLoop();
};
#endif

