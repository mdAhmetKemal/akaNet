#ifndef ENUMTYPE_H
#define ENUMTYPE_H

typedef enum _LayerType{ _OUTPUT = 0, _INPUT = 1, _INTERLAYER, _target, _RESHAP };
typedef enum _ConnectProcessType{ _NOPROCESS = 0, _COPOAC, _CONV, _POOL, _ACTIV, _FULLYCON, _LOSS, _ADDP, _EXTRACT };
typedef enum _ConnectConvType{ _NOCONV = 0, _CONV1D, _CONV2D = 2, _CONV2DtoFE = 2, _CONV2DtoCH, _CONV3C, _CONV3F, _CONV3A, _CONV_1C, _CONV_1F, _CONV_1A, _FULLYLAYERCON };
typedef enum _ConnectPoolType{ _NOPOOL = 0, _MAXPOOL, _AVRPOOL, _GAUSPOOL };
typedef enum _ConnectActType{ _NOACTV = 0, _RELU, _ELU, _LRELU, _LINEER, _SIGM, _TANH, _STEP,_ARCTAN,_SOFTPLUS };
typedef enum _ConnectNormType{ _NONORM = 0, _L1, _L2 };
typedef enum _ConnectLearningType{ _STABIL = 0, _DYNAMIC, _OTHER };
typedef enum _ConnectMatrixType{ _RECTANGULARMAT = 0, _HEXAGONALMAT };

#endif