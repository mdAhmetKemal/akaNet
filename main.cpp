#include "Net.h"
#include <iostream>
#include "NetData.h"
#include "NetCmd.h"

using namespace std;

int main(int argc, char* argv[]){
	NetData file("Data/mnist_test3");
	Net akaNet = Net(file.InputDataPtr, file.OutDataPtr, 400, 1, 28, 28, 1, 10, 1, 1);
	akaNet.showNet(1000, 1000);
	akaNet.setParameter(999, 1, 0.00005, 0.9, 0.00, 0.0005, 0.9, 0., 1);
	akaNet.setLayerNum(4);
	akaNet.setConnectConv(1, 2, _CONV2D, 54, 5, 1, 3, 2, 0.01f);
	akaNet.setConnectPool(2, 3, _MAXPOOL, 2);
	akaNet.setConnectConv(3, 4, _FULLYLAYERCON, 1, 5, 5, 1, 1, 0.1f);
	akaNet.setConnectConv(4, 0, _FULLYLAYERCON, 1, 3, 1, 1, 1, 0.1f);

	
	akaNet.trainNet();
	
	system("pause");
//	cudaDeviceReset();
	return 0;
}