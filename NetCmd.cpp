#include "NetCmd.h"


NetCmd::NetCmd()
{
	loadCommand("Data/command.txt");
}
void NetCmd::loadCommand(string filename){
	ifstream CommandList(filename);
	std::string okunan;
	std::string tempStr;
	std::string parca;
	int sayac = 0;
	while (!CommandList.eof()){
		getline(CommandList, okunan, '\n');
		if ("null" != preProcesLine(okunan)){
			cout << okunan<<endl;
			if (okunan[0] == '>' && okunan[1] != '>'){
				//okunacak data file 
				okunan.erase(0, 1);
				tempStr = okunan;
				tempStr.erase (0, (tempStr.size() - 6 ));
				if (tempStr == ".specs"){
					okunan.erase((okunan.size() - 6), (okunan.size()));
					okunan = "Data/" + okunan;
					file = new NetData(okunan);
					cmdNetSpecs.inputDataPtr = file->InputDataPtr;
					cmdNetSpecs.outDataPtr = file->OutDataPtr;
					cmdNetSpecs.allSampleNum = file->sampleNum;
					cmdNetSpecs.inHeigth = file->inputNeuronNum;
					cmdNetSpecs.outHeigth = file->outputNeuronNum;
				}else
				{
					printf("\nProblem load .specs file!!");
					break;
				}
			// net kurulurken sample piece da lazým bize onu da iste specs file içinden 
			};

			if (okunan[0] == '#'){
				okunan.erase(0, 1);
				//#ShowNet,1000
				//#NetParameter,1800,1,0.0001,0.5,0.,0.001,0.1,0.,1
				//
			};
			if (okunan[0] == 'x' || okunan[0] == 'X' || okunan[0] == '+'){
				okunan.erase(0, 1);
				//+Connect,1, 1,2, _FULLYLAYERCON, 1, 1, 1, 1, 1, 0.01, _FULLYLAYERCON, 1, 1, 1,0.001
				//+Connect,1, 2, _FULLYLAYERCON, 5, 30, 20, 40, 4, 0.001
			};
			
		}


		//her birine göre  hareket etmeli sistem ona gçre belirlemeli komutlaýr 
		//okunan dosya okunmaz ise belrtmeli
		// komuttaki önemli kýsýmlarý belirtmeli iþte þu okundu böyle oldu falan diye 
		//daha sonra komutlarý çalýþtýrmaya baþlamalý 
		stringstream satir(okunan);
	/*	while (!satir.eof()){
			getline(satir, parca, '\,');
			if (sayac == 0)
				sampleNum = atoi(parca.c_str());
			else if (sayac == 1)
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
					cout << "\nThis file loadable: " + filename + ".data" + "\n";
				}
				else{
					cout << "\nLow Data Character definition in the File: " + specsData + "\n";
					isDataStabil = false;
				}
			}
			sayac++;
		} */
	}

}
string NetCmd::preProcesLine(string okunan){
	okunan.erase(std::remove(okunan.begin(), okunan.end(), ' '), okunan.end());
	replace(okunan.begin(), okunan.end(), '-', ',');
	replace(okunan.begin(), okunan.end(), '_', ',');
	replace(okunan.begin(), okunan.end(), ';', ',');
	if (okunan[0] == '\0'|| okunan[0] == '\%' || okunan[0] == '\/' || okunan[0] == '\*'){
		//printf("\nGereksiz:");
		//cout << okunan;
		/*okunan.erase(std::remove(okunan.begin(), okunan.end(), '\/'), okunan.end());
		okunan.erase(std::remove(okunan.begin(), okunan.end(), '\%'), okunan.end());
		okunan.erase(std::remove(okunan.begin(), okunan.end(), '\*'), okunan.end()); */
		okunan="null";
	};
	return okunan;
}
NetCmd::~NetCmd()
{
}
