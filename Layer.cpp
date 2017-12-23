#ifndef LAYER_CPP
#define LAYER_CPP
#include "Layer.h"

int Layer::totalLayerNum = 0;
Layer::Layer(int samp, int fea, int he, int wi, int ch) :
HyperTensor(samp, fea, he, wi, ch, 0)
{
	layerStepNum = totalLayerNum;
	totalLayerNum++;
	l_ActType = _NOACTV;
	l_SpaceType = _RECTANGULARMAT;
	l_CharacteType = _INTERLAYER;
	isNoConnected = true;
};

void Layer::changeLayer(int samp, int fea, int he, int wi, int ch){
	changeHyperTensor(samp, fea, he, wi, ch,0);
}
Layer::~Layer()
{
	HyperTensor::~HyperTensor();
};
#endif