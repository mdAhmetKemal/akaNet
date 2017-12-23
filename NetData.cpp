#include "NetData.h"
#include <iostream>
#include <algorithm>
#include <time.h>
NetData::NetData(string filename)
{
	loadFileSpecs(filename);
}


NetData::~NetData()
{
}
using namespace std;
void  NetData:: loadFileSpecs(string  filename){
	
	std::string specsData =   filename + ".specs";
	ifstream Specs(specsData);
	std::string okunan;
	std::string parca;
	int sayac = 0;
	
	while (!Specs.eof()){
		getline(Specs, okunan, '\n');
		stringstream satir(okunan);
		while (!satir.eof()){
			getline(satir, parca, '\,');
			if (sayac == 0)
				sampleNum = atoi(parca.c_str());
			else if (sayac==1)
				inputStart = atoi(parca.c_str());
			else if (sayac == 2)
				inputFinish = atoi(parca.c_str());
			else if (sayac == 3)
				outStart = atoi(parca.c_str());
			else if (sayac == 4)
				outFinish = atoi(parca.c_str());
			else if (sayac == 5){
				if ((outFinish - outStart + inputFinish - inputStart + 2) <= parca.length()){
					dataCharacter = parca.c_str();
					SampleInpOutSize = outFinish - outStart + inputFinish - inputStart + 2;
					isDataStabil = true;
					cout << "-This file loadableB: " + filename + ".data" ;
				}
				else{
					cout << "-Low Data Character definition in the File: " + specsData ;
					isDataStabil=false;
				}
			}
			sayac++;	
		}
	}
	tempData = new float[SampleInpOutSize*sampleNum]();
	Specs.close();
	specsData = filename + ".data";
	if (isDataStabil)
		loadFileData(specsData);
	
};

void NetData::loadFileData(string  filename){
	std::string fileData = filename ;
	ifstream Data(fileData);
	std::string okunan;
	std::string parca;
	int sayacSample = 0;
	int sayacParca = 0;
	int index;
	while (!Data.eof()){
		getline(Data, okunan, '\n');
		stringstream satir(okunan);
		sayacParca = 0;
		//cout << "\n-sayac: " << sayacSample << "\n";
			while (!satir.eof()){
				//cout << "\n-satr: " << sayacParca << "\n";
				getline(satir, parca, '\;');
				if (sayacParca >= inputStart&&sayacParca <= inputFinish){
					index = sayacParca - inputStart;
					tempData[SampleInpOutSize*sayacSample + index] = atof(parca.c_str());
				}
				else if (sayacParca >= outStart&&sayacParca <= outFinish){
					index = sayacParca - outStart + (inputFinish - inputStart + 1);
					tempData[SampleInpOutSize*sayacSample + index] = atof(parca.c_str());
				}
				sayacParca++;
			}
		sayacSample++;
	}

	if (sampleNum = sayacSample - 1){
		cout << "\n-Loading finish: " + fileData + "\n";
	}else
		cout << "\n-Loading problem : " + fileData + "\n";
	string temp = dataCharacter;
	for (int tempindex = 0; tempindex <= SampleInpOutSize; tempindex++){
		if (tempindex >= inputStart && tempindex <= inputFinish){
			index = tempindex - inputStart;
			dataCharacter[index] = temp[tempindex];
		}
		else if (tempindex >= outStart&& tempindex <= outFinish){
			index = tempindex - outStart + (inputFinish - inputStart +1);
			dataCharacter[index] = temp[tempindex];
		}	
	}
	//cout << temp << endl << dataCharacter << endl;
	Data.close();
	dataModify();
	SampleInpOutSize = outFinish - outStart + inputFinish - inputStart + 2;
/*	for (int Samp = 0; Samp < sampleNum; Samp++){
		for (int a = 0; a < SampleInpOutSize; a++){
			cout << tempData[SampleInpOutSize*Samp + a]<<" ";
		}
		cout << endl;
	}	*/
	
};
void NetData::dataModify(){
	float * maxList = new float[SampleInpOutSize]();
	float * medianList = new float[SampleInpOutSize]();
	int * supportList = new int[SampleInpOutSize]();
	float maksTemp ,medianTemp;
	for (int Sutun = 0; Sutun < SampleInpOutSize; Sutun++)
	{
		maksTemp = 0;
		medianTemp = 0;
		for (int samplerrr = 0; samplerrr < sampleNum; samplerrr++){
			medianTemp += tempData[SampleInpOutSize*samplerrr + Sutun];
			if (tempData[SampleInpOutSize*samplerrr + Sutun]>maksTemp){
				maksTemp = tempData[SampleInpOutSize*samplerrr + Sutun];
			}
		}
		maxList[Sutun] = maksTemp;
		medianList[Sutun] = medianTemp/(float)sampleNum;
	}
	totalNeuron = 0;
	inputOutputPoint = 0;
	for (int Sutun = 0; Sutun < SampleInpOutSize; Sutun++)
	{
		
		if (dataCharacter[Sutun] == 'N' || dataCharacter[Sutun] == 'n')
		{
			if (Sutun < inputFinish - inputStart+1 )
				inputOutputPoint += maxList[Sutun];
			supportList[Sutun] = totalNeuron-1;
			totalNeuron += maxList[Sutun];
		}
		else if (dataCharacter[Sutun] == 'V' || dataCharacter[Sutun] == 'v')
		{
			if (Sutun < inputFinish - inputStart+1 )
				inputOutputPoint++;
			supportList[Sutun] = totalNeuron-1;
			totalNeuron++;
		}
	}
	cout << totalNeuron << "totaal neuron " << SampleInpOutSize << "inoout \n";
	DataModified = new float[totalNeuron*sampleNum]();
	for (int satir = 0; satir < sampleNum; satir++){
		int goIndex;
		for (int inSamp = 0; inSamp < SampleInpOutSize; inSamp++)
		{
			
			if (dataCharacter[inSamp] == 'N' || dataCharacter[inSamp] == 'n')
			{
				if (tempData[SampleInpOutSize*satir + inSamp]!=0)
				{
					
					goIndex = supportList[inSamp] + (int)tempData[SampleInpOutSize*satir + inSamp];
					DataModified[totalNeuron*satir + goIndex] = 1;
					//cout << "goIndex " <<  goIndex << "\n";
				}
			}
			else if (dataCharacter[inSamp] == 'V' || dataCharacter[inSamp] == 'v')
			{

				if (tempData[SampleInpOutSize*satir + inSamp] !=0)
				{
					
					goIndex = supportList[inSamp] + 1;
					if (tempData[SampleInpOutSize*satir + inSamp] < 0){
						DataModified[totalNeuron*satir + goIndex] = medianList[inSamp] / maxList[inSamp];
					}
					else{
						DataModified[totalNeuron*satir + goIndex] = 
						(	tempData[SampleInpOutSize*satir + inSamp] / (maxList[inSamp]));
						//ekleme
					}
				}
			}
			//printf("%.1f ", tempData[SampleInpOutSize*satir + inSamp]);
		}
		//printf("\n");
		

	}
	/*
	for (int satir = 0; satir < sampleNum; satir++){
		for (int inSamp = 0; inSamp < totalNeuron; inSamp++)
		{
			printf("%.1f ", DataModified[totalNeuron*satir + inSamp] );
		}
		printf("\n");
	}
	*/
	shuffleArray(DataModified);	
	
	int outputNeuronNum = totalNeuron - inputOutputPoint;
	cout << "--input Neu:" << inputOutputPoint << "  out Neu:" << outputNeuronNum << "  total Neu:" << totalNeuron << endl;
	InputDataPtr = new float[sampleNum*inputOutputPoint]();
	OutDataPtr = new float[sampleNum*outputNeuronNum]();
	inputNeuronNum = inputOutputPoint;
	for (int samNum = 0; samNum < sampleNum; samNum++)
		for (int neu = 0; neu < totalNeuron; neu++){
			if (neu < inputOutputPoint){
				InputDataPtr[inputOutputPoint*samNum + neu] = DataModified[totalNeuron*samNum + neu];
			}
			else if(neu >= inputOutputPoint){ // 9 iken 
				OutDataPtr[outputNeuronNum*samNum + (neu - inputOutputPoint)] = DataModified[totalNeuron*samNum + neu];
				//OutDataPtr[outputNeuronNum*samNum + (neu - inputOutputPoint)] = (DataModified[totalNeuron*samNum + neu]-0.5f)*2.;
				//cout << (neu - inputOutputPoint) << ":outputNeuronNum     " << OutDataPtr[outputNeuronNum*samNum + (neu - inputOutputPoint)] << ":neu     ";
			}
			//cout << endl;
		}
	/*
	for (int Samp = 0; Samp < sampleNum; Samp++){
		for (int a = 0; a < inputOutputPoint; a++){
			cout << InputDataPtr[inputOutputPoint*Samp + a] << " ";

		}
		cout << "**   ";
		for (int a = 0; a < (outputNeuronNum); a++){
			cout << OutDataPtr[(outputNeuronNum)*Samp + (a)] << "-";
		}
		cout << endl;
	}
	*/
	/*
	for (int Samp = 0; Samp < sampleNum; Samp++){
		for (int a = 0; a < (totalNeuron - inputOutputPoint); a++){
				cout << OutDataPtr[(totalNeuron - inputOutputPoint)*Samp + (a - inputOutputPoint)] << " ";
		}
		cout << endl;
	}*/
};
void NetData::shuffleArray(float* data){
	int *arrayNum = new int[sampleNum]();
	for (int a = 0; a < sampleNum; a++){
		arrayNum[a] = a;
	}
	srand(time(NULL));
	random_shuffle(&arrayNum[0], &arrayNum[sampleNum]);
	random_shuffle(&arrayNum[0], &arrayNum[sampleNum]);
	float * temp = new float[totalNeuron*sampleNum]();

	for (int satir = 0; satir < sampleNum; satir++){
		int randomLine = arrayNum[satir];
		for (int n = 0; n < totalNeuron; n++){
			temp[totalNeuron*satir + n] = data[totalNeuron*randomLine + n];
		}
	}



	for (int satir = 0; satir < sampleNum; satir++){
		for (int n = 0; n < totalNeuron; n++){
			DataModified[totalNeuron*satir + n] = temp[totalNeuron*satir + n];
		}
	}
	/*
	for (int satir = 0; satir < sampleNum; satir++){
		for (int inSamp = 0; inSamp < totalNeuron; inSamp++)
		{
			printf("%.1f ", DataModified[totalNeuron*satir + inSamp]);
		}
		printf("\n");
	}
	*/
	delete []temp;

};