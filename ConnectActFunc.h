#ifndef CONNECTACTFUNC_H
#define CONNECTACTFUNC_H
#include "ConnectCuda.h"


extern void ActivationFunction(_ConnectActType layerAct, float * layer, int layerSize,float parameter=0.1);
extern void DerivationFunction(_ConnectActType layerAct, float * layer, int layerSize, float parameter=0.1);
#endif