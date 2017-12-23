#ifndef SHOWNETCUDA_H
#define SHOWNETCUDA_H
#include <cuda_runtime.h>
#include "Connect.h"
extern void ConnectShower(int pikSize, uchar4 * other_out, int W, int H, Connect * showingConnnect);
#endif
