#ifndef LAYER_H
#define LAYER_H
#include "HyperTensor.h"
#include "EnumType.h"


class Layer :public HyperTensor
{
public:
	
	Layer(int samp = 1, int fea = 1, int he = 5, int wi = 5, int ch = 2);
	~Layer();
	int layerStepNum=0;
	void changeLayer(int samp, int fea, int he, int wi, int ch);
	static	int totalLayerNum;
	_ConnectActType l_ActType;
	_ConnectMatrixType l_SpaceType;
	_LayerType l_CharacteType;
	bool isNoConnected;
private:
};
#endif


