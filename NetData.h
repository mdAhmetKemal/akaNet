#ifndef NetData_H
#define NetData_H

#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;
class NetData
{
public:
	void loadFileSpecs(string filename);
	void loadFileData(string  filename);
	NetData(string filename);
	~NetData();
	float* InputDataPtr;
	float* OutDataPtr;
	int sampleNum;
	int inputNeuronNum;
	int outputNeuronNum;
private:
	float * DataModified;
	void shuffleArray(float* data);
	int totalNeuron;
	int inputOutputPoint;
	void dataModify();
	float* tempData;
	string dataCharacter;
	int inputStart;
	int inputFinish;
	int outStart;
	int outFinish;
	int SampleInpOutSize;
	int *ArrayShuffleTemplate;
	bool isDataStabil;
};
#endif
