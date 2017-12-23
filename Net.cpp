#ifndef NET_CPP
#define NET_CPP
#include "Net.h"

void Net::showNet(int Heigt, int Width){
	isOpenGl = true;
	ShowHeigth = Heigt;
	ShowWidth = Width;
	initGLUT(ShowWidth, ShowHeigth);
	cudaMalloc(&other_out, ShowHeigth*ShowWidth* sizeof(uchar4));
	cudaMemset(other_out, 0, ShowHeigth *ShowWidth*sizeof(uchar4));
	cudaMalloc(&accuracyTrain, ShowWidth* sizeof(float));
	cudaMemset(accuracyTrain, 0, ShowWidth* sizeof(float));
	cudaMalloc(&accuracyTest, ShowWidth* sizeof(float));
	cudaMemset(accuracyTest, 0, ShowWidth* sizeof(float));
	cudaMalloc(&errorArray, ShowWidth* sizeof(float));
	cudaMemset(errorArray, 0, ShowWidth* sizeof(float));
	testErrorArray = new float[ShowWidth]();
	trainErrorArray = new float[ShowWidth]();
	MSErrorArray = new float[ShowWidth]();
	d_out = new uchar4[ShowHeigth*ShowWidth]();

}
void Net::trainNet(){

		loadAllDataGPU();
		for (int trainEpoch = 0; trainEpoch <= Epoch; trainEpoch++){

			shuffleGPUData();

			// shuffleNetData();
			//
			for (int pieceNum = 0; pieceNum < pieceTotalNum / 2; pieceNum++){
				int load = pieceNum;

				loadNetGPU(load);

			//	loadNet(load);
				manuelKernel(connectMap[2][0]->Tensor, connectMap[2][0]->hTfeatureNum, 3, 3);
				feedWorkNet();
				errorCalculateNet();
				//showCompare(layerMap[0]->Tensor, layerMap[totalLayerNumNet]->Tensor, pieceSampSize);
				//errorPercent(layerMap[0]->Tensor, layerMap[totalLayerNumNet]->Tensor, accuracyTrain,inputSampleTotal, pieceSampSize, trainEpoch,ShowWidth);
				accuracyMultinominal(layerMap[0]->Tensor, layerMap[totalLayerNumNet]->Tensor, accuracyTrain, layerMap[0]->sizePiksel, inputSampleTotal, pieceSampSize, trainEpoch, ShowWidth);
				errorSummer(layerMap[totalLayerNumNet]->paralelTensor, errorArray, pieceSampSize, trainEpoch, ShowWidth);
				backPropagationNet(); ///////////////connect sýnýfý içinde 
				//changeWeigth(); ///////////////connect sýnýfý içinde 
				cudaMemcpy(testErrorArray, accuracyTrain, sizeof(float) * ShowWidth, cudaMemcpyDeviceToHost);
				cudaMemcpy(MSErrorArray, errorArray, sizeof(float) * ShowWidth, cudaMemcpyDeviceToHost);
			}

			if (true){
				//shuffleNetData();
				shuffleGPUData();
				for (int pieceNum = pieceTotalNum / 2; pieceNum < pieceTotalNum; pieceNum++){
					int load = pieceNum;
					//loadNet(load);
					loadNetGPU(load);
					//printf("\n testtttt");
					feedWorkNet();
					errorCalculateNet();
					//showCompare(layerMap[0]->Tensor, layerMap[totalLayerNumNet]->Tensor, pieceSampSize);
					//errorPercent(layerMap[0]->Tensor, layerMap[totalLayerNumNet]->Tensor, accuracyTest,inputSampleTotal, pieceSampSize, trainEpoch,ShowWidth);
					accuracyMultinominal(layerMap[0]->Tensor, layerMap[totalLayerNumNet]->Tensor, accuracyTest, layerMap[0]->sizePiksel, inputSampleTotal, pieceSampSize, trainEpoch, ShowWidth);
					cudaMemcpy(trainErrorArray, accuracyTest, sizeof(float) * ShowWidth, cudaMemcpyDeviceToHost);
				}
			}
			//

			//error1D2D(accuracyTrain,accuracyTest, errorArray, other_out, ShowWidth, ShowHeigth);
			//preComWeigthShower(25, other_out, ShowWidth, ShowHeigth, connectMap[2][0]->Tensor, connectMap[2][0]->sizePiksel,
			//connectMap[2][0]->RegulatorKernel->Tensor, connectMap[2][0]->RegulatorKernel->sizePiksel);
			LayerShower(6, other_out, ShowWidth, ShowHeigth, layerMap[2]);
			ConnectShower(15, other_out, ShowWidth, ShowHeigth, connectMap[2][0]);
			cudaMemcpy(d_out, other_out, sizeof(uchar4) *ShowHeigth* ShowWidth, cudaMemcpyDeviceToHost);
			updateNetGL();
		}

};
Net::Net(float * inputData, float * targetData, int totalSampleNum, int samplePiece,
	int inHeigth = 1, int inWidth = 1, int inChanal = 1, int resHeigth = 1,
	int resWidth = 1, int resChanal = 1)
{
	inputDataPtr = inputData;
	targetDataPtr = targetData;
	inputSampleTotal = totalSampleNum;
	pieceSampSize = samplePiece;
	inputSampleHeigth = inHeigth;
	inputSampleWidth = inWidth;
	inputSampleChanal = inChanal;
	targetSampleTotal = totalSampleNum;
	targetSampleHeigth = resHeigth;
	targetSampleWidth = resWidth;
	targetSampleChanal = resChanal;
	setParameter(700, 1, 0.00015, 0.9, 1, 1, 1, 1, 1);
	LearningSamplePiecesNum = 0;
	buildInputtargetLayer();
}
void Net::setParameter(
	int epoch = 30, int batch = 1,
	float learningR1 = 0.001, float learninMom1 = 0.5,
	float  noiseSubJoinR1 = 0.001, float learningR2 = 0.0001,
	float learninMom2 = 0.9, float  noiseSubJoinR2 = 0.001,
	float learningR3 = 0.01){
	pieceTotalNum = inputSampleTotal / pieceSampSize;
	Epoch = epoch;
	Batch = batch;
	learningRate1 = learningR1;
	learningRate2 = learningR2;
	learningRate3 = learningR3;
	learningMomentum1 = learninMom1;
	learningMomentum2 = learninMom2;
	noiseSubJoinRate1 = noiseSubJoinR1;
	noiseSubJoinRate2 = noiseSubJoinR2;

}

void Net::changeWeigth(){

	for (int backLayerNum = totalLayerNumNet; backLayerNum >= 2; backLayerNum--){
		//if (connectMap[feedLayerNum][0]->totalOutLayNum == connectWhereLayerMap[feedLayerNum][0]){
		connectMap[backLayerNum][0]->changeWeigthConnect(learningRate1, learningMomentum1, noiseSubJoinRate1,
			learningRate2,learningMomentum2,noiseSubJoinRate2);
		//printf("\nchangeWeigth ::Net");
		//}	
	}
};
void Net::backPropagationNet(){
	for (int backLayerNum = totalLayerNumNet; backLayerNum >= 3; backLayerNum--){
		//if (connectMap[feedLayerNum][0]->totalOutLayNum == connectWhereLayerMap[feedLayerNum][0]){
		connectMap[backLayerNum][0]->backPropagationConnect();
		
		//}	
	}
};
void Net::errorCalculateNet(){
	//out layer5tir ve 0 a baðlýdýr.
	//0 a baðlý iþlem gereði 
	connectMap[0][0]->ErrorConnect();
};
void Net::feedWorkNet(){
	for (int feedLayerNum = 2; feedLayerNum <= totalLayerNumNet; feedLayerNum++){
		//if (connectMap[feedLayerNum][0]->totalOutLayNum == connectWhereLayerMap[feedLayerNum][0]){
		connectMap[feedLayerNum][0]->FeedWorkConnect();
		//}	
	}
	///////////////connect sýnýfý içinde 
	// conv()
	//fullyCon()
	//fullyWithAct()
	//convWithAct()
}
void 	Net::setConnectPool(int inputLa, int outLa, _ConnectPoolType conPoolTy = _NOPOOL,int p_Scale = 1){
	if (outLa - inputLa == 1 && inputLa >= 1)
	{
		if (connectWhereLayerMap[inputLa][0] != -1 || inputLa == 1)
		{
			int connectStep = 0;
			while (connectWhereLayerMap[outLa][connectStep] != -1)
				connectStep++;
			connectWhereLayerMap[outLa][connectStep] = inputLa;
			totalConnectNum++;
			connectMap[outLa][connectStep] = new Connect(layerMap[inputLa], layerMap[outLa]);
			connectMap[outLa][connectStep]->setNewPool(
				conPoolTy, p_Scale);
		}
	}
	else if (outLa == 0 || inputLa == totalLayerNumNet)
	{	
		printf("\n No pooling Last to Out Layer! ");
	}
	isOpenGl = false;
}

void 	Net::setConnectConv(int inputLa, int outLa, _ConnectConvType conConvTy = _CONV2D,
	int k_Fea = 12, int k_Scale = 3, int k_Dep = 1,
	int strideConv = 1, int paddConv = 1, float  fac = 0.01){
	if (outLa - inputLa == 1 && inputLa >= 1)
	{
		if (connectWhereLayerMap[inputLa][0] != -1 || inputLa == 1)
		{
			int connectStep = 0;
			while (connectWhereLayerMap[outLa][connectStep] != -1)
				connectStep++;
			connectWhereLayerMap[outLa][connectStep] = inputLa;
			totalConnectNum++;
			connectMap[outLa][connectStep] = new Connect(layerMap[inputLa], layerMap[outLa]);
			connectMap[outLa][connectStep]->setNewConv(
				conConvTy, k_Fea, k_Scale, k_Dep, strideConv, paddConv, fac);
		}
	}
	else if (outLa == 0 && inputLa == totalLayerNumNet)
	{
		if (connectWhereLayerMap[inputLa][0] != -1)
		{
			int connectStep = 0;
			while (connectWhereLayerMap[outLa][connectStep] != -1)
				connectStep++;
			connectWhereLayerMap[outLa][connectStep] = inputLa;
			connectMap[outLa][connectStep] = new Connect(layerMap[inputLa], layerMap[outLa]);
			connectMap[outLa][connectStep]->setNewConv(
				conConvTy, k_Fea, k_Scale, k_Dep, strideConv, paddConv, fac);
				
		}
	}
	isOpenGl = false;
}
void Net::setConnectConv(int inputLa, int regLa, int outLa, _ConnectConvType k_InpTy = _CONV2D,
	int k_Fea = 12, int k_Scale = 3, int k_Dep = 1,
	int k_strideConv = 1, int k_paddConv = 1,
	float k_fac=0.01,
	_ConnectConvType a_RegTy = _CONV2D,
	int a_Fea = 12, int a_Scale = 3, int a_Dep = 1,
	float  a_fac = 0.01){
	if (outLa - inputLa == 1 && inputLa >= 1 && regLa <= inputLa)
	{
		if (connectWhereLayerMap[inputLa][0] != -1 || inputLa == 1)
		{
			int connectStep = 0;
			while (connectWhereLayerMap[outLa][connectStep] != -1)
				connectStep++;
			connectWhereLayerMap[outLa][connectStep] = inputLa;
			totalConnectNum++;
			//*****************************
			connectMap[outLa][connectStep] = new Connect(layerMap[inputLa], layerMap[regLa], layerMap[outLa]);
			connectMap[outLa][connectStep]->setNewConv(
				k_InpTy, k_Fea, k_Scale, k_Dep, k_strideConv, k_paddConv, k_fac);
			connectMap[outLa][connectStep]->setNewRegul(
				a_RegTy,a_Fea, a_Scale, a_Dep, a_fac);
			//****************************
		}
	}
	else if (outLa == 0 && inputLa == totalLayerNumNet && regLa <= inputLa)
	{
		if (connectWhereLayerMap[inputLa][0] != -1)
		{
			int connectStep = 0;
			while (connectWhereLayerMap[outLa][connectStep] != -1)
				connectStep++;
			connectWhereLayerMap[outLa][connectStep] = inputLa;
			/*******zaten son layer target layerý   baðlasan da baðlamasanda aaslýnda bir þey olmaz*/
			connectMap[outLa][connectStep] = new Connect(layerMap[inputLa], layerMap[regLa], layerMap[outLa]);
			/*connectMap[outLa][connectStep]->setNewConv(
			conConvTy, k_Fea, k_Scale, k_Dep, strideConv, paddConv, fac);
			*/
		}
	}
	
};

void Net::buildInputtargetLayer(){
	
	layerMap[0] = new Layer(pieceSampSize, 1, targetSampleHeigth,
		targetSampleWidth, targetSampleChanal);//target Layer
	layerMap[0]->l_CharacteType = _OUTPUT;
	layerMap[1] = new Layer(pieceSampSize, 1, inputSampleHeigth,
		inputSampleWidth, inputSampleChanal);//Input Layer
	layerMap[1]->isNoConnected = false;
	layerMap[1]->l_CharacteType = _INPUT;
	totalLayerNumNet = 2;
	totalConnectNum = 0;
	for (int l = 0; l < MaxLayer; l++)
		for (int co = 0; co < MaxLayer; co++)
			connectWhereLayerMap[l][co] = -1;
};
void Net::loadAllDataGPU(){

	if (pieceTotalNum*layerMap[1]->sizePiksel < 1000000000){
		
		float * tempIn = new float[pieceTotalNum*layerMap[1]->sizePiksel]();
		for (int b = 0; b < pieceTotalNum*layerMap[1]->sizePiksel; b++){
			tempIn[b] = inputDataPtr[b];
		} 
		
		if (pieceSampSize == layerMap[1]->hTsample &&
			1 == layerMap[1]->hTfeatureNum &&
			inputSampleHeigth == layerMap[1]->hTheigth &&
			inputSampleWidth == layerMap[1]->hTwidth &&
			inputSampleChanal == layerMap[1]->hTchanal)
		{

			Data_WiHeSa2SaHeWi(tempIn, inputSampleWidth, inputSampleHeigth, pieceSampSize*pieceTotalNum);
			int boyut = inputSampleWidth*inputSampleHeigth* pieceSampSize*pieceTotalNum;
			cudaError_t target = cudaMalloc(&allInputDataOnGpu, boyut*sizeof(float));
			cudaError_t target2 = cudaMemcpy(allInputDataOnGpu, tempIn, boyut*sizeof(float), cudaMemcpyHostToDevice);

			/*
			int sizeT = pieceTotalNum*layerMap[1]->sizePiksel;
			float * tempIn2 = new float[sizeT]();
			cudaMemcpy(tempIn2, allInputDataOnGpu, sizeT*sizeof(float), cudaMemcpyDeviceToHost);
			for (int Sa = 0; Sa < pieceTotalNum*layerMap[1]->hTsample; Sa++){
				for (int He = 0; He < inputSampleHeigth; He++){
					for (int Wi = 0; Wi < inputSampleWidth; Wi++){

						if (tempIn2[pieceTotalNum*layerMap[1]->hTsample*(inputSampleHeigth*(Wi)+He) + Sa] > 0){
							printf(" \x4B");
						}
						else{
							printf("  ");
						}

						if ((inputSampleWidth*(inputSampleHeigth*(Sa)+He) + Wi + 1) % 28 == 0){
							printf("\n- - - ");
						}

					}
				}
			}
			*/
			
			if (target != cudaSuccess){
				printf("\nSorun:Net::loadAllDataGPU1");
			}
			else{
					printf("\nBasari:Net::loadAllDataGPU1");
			}
			if (target2 != cudaSuccess){
				printf("\nSorun:Net::loadAllDataGPU2");
			}
			else{
						printf("\nBasari:Net::loadAllDataGPU2");
			}
			//showonGpuData(allInputDataOnGpu, inputSampleWidth*inputSampleHeigth, pieceSampSize*pieceTotalNum);
		}
		

		float * tempOut = new float[pieceTotalNum*layerMap[0]->sizePiksel]();
		for (int b = 0; b < pieceTotalNum*layerMap[0]->sizePiksel; b++){
			tempOut[b] = targetDataPtr[b];
		}

		

		if (pieceSampSize == layerMap[0]->hTsample &&
			1 == layerMap[0]->hTfeatureNum &&
			targetSampleHeigth == layerMap[0]->hTheigth &&
			targetSampleWidth == layerMap[0]->hTwidth &&
			targetSampleChanal == layerMap[0]->hTchanal)
		{
			Data_WiHeSa2SaHeWi(tempOut, targetSampleWidth, targetSampleHeigth, pieceSampSize*pieceTotalNum);
			cudaError_t target = cudaMalloc(&allTargetDataOnGpu, pieceTotalNum*layerMap[0]->sizePiksel*sizeof(float));
			cudaError_t target2 = cudaMemcpy(allTargetDataOnGpu, tempOut, pieceTotalNum*layerMap[0]->sizePiksel*sizeof(float), cudaMemcpyHostToDevice);

			//*******************************************************************************
			int sizeT = pieceTotalNum*layerMap[0]->sizePiksel;
			float * tempOut2 = new float[sizeT]();
			cudaMemcpy(tempOut2, allTargetDataOnGpu, sizeT*sizeof(float), cudaMemcpyDeviceToHost);
			/*
			for (int Sa = 0; Sa < pieceTotalNum*layerMap[0]->hTsample; Sa++){
				printf("\n> %d ",Sa);
				for (int He = 0; He < layerMap[0]->hTheigth; He++){
					for (int Wi = 0; Wi < layerMap[0] ->hTwidth; Wi++){

						
						if (tempOut2[pieceTotalNum*layerMap[0]->hTsample*(layerMap[0]->hTheigth*(Wi)+He) + Sa] > 0){

							printf("%d", layerMap[0]->hTheigth*(Wi)+He);
						}
						else{
							printf("-");
						}
					}
				}
			}
			*/
			//*********************************************************************************

			if (target != cudaSuccess){
				printf("\nSorun:Net::loadAllDataGPU3");
			}
			else{
				printf("\nBasari:Net::loadAllDataGPU3");
			}
			if (target2 != cudaSuccess){
				printf("\nSorun:Net::loadAllDataGPU4");
			}
			else{
					printf("\nBasari:Net::loadAllDataGPU4");
			}

		}
		delete[] tempIn;
		delete[] tempOut;
	}


	cudaError_t target = cudaMalloc(&shuffledInputDataOnGpu, pieceTotalNum*layerMap[1]->sizePiksel*sizeof(float));
	cudaError_t target2 = cudaMalloc(&shuffledTargetDataOnGpu, pieceTotalNum*layerMap[1]->sizePiksel*sizeof(float));
	if (target != cudaSuccess){
		printf("\nSorun:Net::shuffleGPUData1");
	}
	else{
		//	printf("\nBasari:Net::shuffleGPUData1");
	}
	if (target2 != cudaSuccess){
		printf("\nSorun:Net::shuffleGPUData2");
	}
	else{
		//	printf("\nBasari:Net::shuffleGPUData2");
	}
	
}
void Net::shuffleGPUData(){


	int  totalSampleNum = inputSampleTotal;

	int *arrayNum = new int[totalSampleNum]();
	for (int a = 0; a < totalSampleNum; a++){
		arrayNum[a] = a;
	}
	srand(time(NULL));
	random_shuffle(&arrayNum[0], &arrayNum[totalSampleNum]);
	random_shuffle(&arrayNum[0], &arrayNum[totalSampleNum]);


	shuffleGpu(allInputDataOnGpu,  shuffledInputDataOnGpu, totalSampleNum, inputSampleHeigth, inputSampleWidth, inputSampleChanal,arrayNum);
	shuffleGpu(allTargetDataOnGpu, shuffledTargetDataOnGpu, totalSampleNum, targetSampleHeigth, targetSampleWidth, targetSampleChanal,arrayNum);
	printf("\n shuffleGPUData");
};
void Net::loadNetGPU(int LearningProcesStepSample){
	if (LearningProcesStepSample*pieceSampSize <= inputSampleTotal)
	{


		float * tempIn;
		cudaMalloc(&tempIn, layerMap[1]->sizePiksel*sizeof(float));
		loadPieceOnGpu(shuffledInputDataOnGpu, tempIn, layerMap[1]->sizePiksel / layerMap[1]->hTsample, layerMap[1]->hTsample,
			LearningProcesStepSample, inputSampleTotal);
		layerMap[1]->loadTensorFromGPU(tempIn);
		
		/*
		int sizeT =  layerMap[1]->sizePiksel;
		float * tempIn2 = new float[sizeT]();
		cudaMemcpy(tempIn2, tempIn, sizeT*sizeof(float), cudaMemcpyDeviceToHost);
		for (int Sa = 0; Sa < layerMap[1]->hTsample; Sa++){
			for (int He = 0; He < inputSampleHeigth; He++){
				for (int Wi = 0; Wi < inputSampleWidth; Wi++){

					if (tempIn2[layerMap[1]->hTsample*(inputSampleHeigth*(Wi)+He) + Sa] > 0){
						printf(" \x4B");
					}
					else{
						printf("  ");
					}

					if ((inputSampleWidth*(inputSampleHeigth*(Sa)+He) + Wi + 1) % 28 == 0){
						printf("\n+++ ");
					}

				}
			}
		}
		delete[]  tempIn2;
		*/

		

		
		float * tempOut;
		cudaMalloc(&tempOut, layerMap[0]->sizePiksel*sizeof(float));
		loadPieceOnGpu(shuffledTargetDataOnGpu, tempOut, layerMap[0]->sizePiksel / layerMap[0]->hTsample, layerMap[0]->hTsample,
			LearningProcesStepSample, inputSampleTotal);
		layerMap[0]->loadTensorFromGPU(tempOut);
		
		/*
		 sizeT = layerMap[0]->sizePiksel;
		float * tempOut2 = new float[sizeT]();
		cudaMemcpy(tempOut2, tempOut, sizeT*sizeof(float), cudaMemcpyDeviceToHost);
		for (int Sa = 0; Sa < layerMap[0]->hTsample; Sa++){
			printf("\n> %d ", Sa);
			for (int size = 0; size < layerMap[0]->sizePiksel / layerMap[0]->hTsample; size++){

				if (tempOut2[layerMap[0]->hTsample*size + Sa] > 0){
					printf("%d", size);
				}
				else{
					printf("-");
				}

			}
		}
		printf("\n");
		*/
		cudaFree(tempIn);
		cudaFree(tempOut);
		//printf("\n loadNetGPU  %d  \n", LearningProcesStepSample);
	}
};



void Net::loadNet(int LearningProcesStepSample){
	if (LearningProcesStepSample*pieceSampSize <= inputSampleTotal)
	{


		float * tempIn = new float[layerMap[1]->sizePiksel]();
		int tempPiece = LearningProcesStepSample*layerMap[1]->sizePiksel;
		for (int b = 0; b < layerMap[1]->sizePiksel; b++)
			tempIn[b] = inputDataPtr[b + tempPiece];
		if (pieceSampSize == layerMap[1]->hTsample &&
			1 == layerMap[1]->hTfeatureNum &&
			inputSampleHeigth == layerMap[1]->hTheigth &&
			inputSampleWidth == layerMap[1]->hTwidth &&
			inputSampleChanal == layerMap[1]->hTchanal)
		{
			/****  tempIn bu kýsýmda dataSam1,dataSamp2,dataSamp3 diye kaydediliyor
			çevirici fonkisyon burdaki temp kýsmý deðiþtirmeli 
			*/
			Data_WiHeSa2SaHeWi(tempIn, inputSampleWidth, inputSampleHeigth, pieceSampSize);
			layerMap[1]->loadTensor(tempIn);
		}
		delete[] tempIn;

		float * tempOut = new float[layerMap[0]->sizePiksel]();
		tempPiece = LearningProcesStepSample*layerMap[0]->sizePiksel;
		for (int b = 0; b < layerMap[0]->sizePiksel; b++){
			tempOut[b] = targetDataPtr[b + tempPiece];
		}
		if (pieceSampSize == layerMap[0]->hTsample &&
			1 == layerMap[0]->hTfeatureNum &&
			targetSampleHeigth == layerMap[0]->hTheigth &&
			targetSampleWidth == layerMap[0]->hTwidth &&
			targetSampleChanal == layerMap[0]->hTchanal)
		{
			layerMap[0]->loadTensor(tempOut);

			int sizeT = layerMap[0]->sizePiksel;
		//	float * tempOut2 = new float[sizeT]();
			//cudaMemcpy(tempOut2, shuffledTargetDataOnGpu, sizeT*sizeof(float), cudaMemcpyDeviceToHost);
			/*
			for (int Sa = 0; Sa < layerMap[0]->hTsample; Sa++){
				for (int size = 0; size < layerMap[0]->sizePiksel / layerMap[0]->hTsample; size++){

					if (tempOut[layerMap[0]->hTsample*size + Sa] > 0){
						printf("%d", size);
					}
					else{
						printf("-");
					}
				}
			}
			printf("\n");
			*/
		}
		delete[] tempOut;
	}
};

void Net::setLayerNum(int totalLayerN){
	cleanInterLayer();
	totalLayerNumNet = totalLayerN;
	for (int layerNum = 2; layerNum < totalLayerNumNet; layerNum++)
	{
		layerMap[layerNum] = new Layer();
		layerMap[layerNum]->isNoConnected = true;
		layerMap[layerNum]->l_CharacteType = _INTERLAYER;
	}
	layerMap[totalLayerNumNet] = new Layer(pieceSampSize, 1, targetSampleHeigth,
		targetSampleWidth, targetSampleChanal);
	layerMap[totalLayerNumNet]->l_CharacteType = _target;
};



Net::~Net()
{
	cleanData();
	cudaDeviceReset();
}
void Net::cleanData(){
	if (pbo) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		   glDeleteBuffers(1, &pbo);
		   glDeleteTextures(1, &tex);
	}
	if (isOpenGl){
		cudaFree(accuracyTrain);
		cudaFree(accuracyTest);
		cudaFree(errorArray);
		cudaFree(other_out);
	}
	delete[] trainErrorArray;
	delete[] testErrorArray;
	delete[] layerMap;
	delete[] connectMap;
	cudaFree(allInputDataOnGpu);
	cudaFree(allTargetDataOnGpu);
	cudaFree(shuffledInputDataOnGpu);
	cudaFree(shuffledTargetDataOnGpu);
	//delete[] connectMap;
};
void Net::cleanInterLayer(){
	for (int tl = 0; tl < totalLayerNumNet; tl++){
		if (tl > 1){
			delete layerMap[tl];
			delete connectMap[tl][0];
		}
	}
	totalLayerNumNet = 2;
	totalConnectNum = 0;
	for (int l = 0; l < MaxLayer; l++)
		for (int co = 0; co < MaxLayer; co++)
			connectWhereLayerMap[l][co] = -1;
};
void Net::shuffleNetData(){

	//	inputDataPtr
	//	targetDataPtr
	int inputSize = inputSampleHeigth *inputSampleWidth *inputSampleChanal;
	int targetSize = targetSampleHeigth*targetSampleWidth *targetSampleChanal;
	int  totalSampleNum = inputSampleTotal;

	int *arrayNum = new int[totalSampleNum]();
	for (int a = 0; a < totalSampleNum; a++){
		arrayNum[a] = a;
	}
	srand(time(NULL));
	random_shuffle(&arrayNum[0], &arrayNum[totalSampleNum]);
	random_shuffle(&arrayNum[0], &arrayNum[totalSampleNum]);
	float * tempInput = new float[inputSize*totalSampleNum]();
	float * tempTarget = new float[targetSize*totalSampleNum]();

	for (int satir = 0; satir < totalSampleNum; satir++){
		int randomLine = arrayNum[satir];
		for (int n = 0; n < inputSize; n++){
			tempInput[inputSize*satir + n] = inputDataPtr[inputSize*randomLine + n];
		}
		for (int n = 0; n < targetSize; n++){
			tempTarget[targetSize*satir + n] = targetDataPtr[targetSize*randomLine + n];
		}
	}

	for (int satir = 0; satir < totalSampleNum; satir++){
		for (int n = 0; n < inputSize; n++){
			inputDataPtr[inputSize*satir + n] = tempInput[inputSize*satir + n];
		}
		for (int n = 0; n < targetSize; n++){
			targetDataPtr[targetSize*satir + n] = tempTarget[targetSize*satir + n];
		}
	}
	delete[]tempInput;
	delete[]tempTarget;
};
void Net::Data_WiHeSa2SaHeWi(float *DataIn, int width, int heigth, int sample){

	float * temp = new float[width*heigth*sample]();
	for ( int Wi = 0; Wi < width; Wi++){
		for (int He = 0; He < heigth; He++){
			for (int Sa = 0; Sa < sample; Sa++){
				temp[sample*(heigth*(Wi)+He)+Sa] = DataIn[width*(heigth*(Sa)+He) + Wi];

			}
		}
	}
	for (int k = 0; k < width*heigth*sample; k++){
		DataIn[k] = temp[k];
	}
	delete[] temp;
	/*
	for (int Sa = 0; Sa < sample; Sa++){
		for (int He = 0; He < heigth; He++){
			for (int Wi = 0; Wi < width; Wi++){

				if (DataIn[sample*(heigth*(Wi)+He) + Sa] > 0){
					printf(" \x6A");
				}
				else{
					printf("  ");
				}

				if ((width*(heigth*(Sa)+He) + Wi + 1) % 28 == 0){
					printf("\n");
				}

			}
		}
	} */

};

void Net::manuelKernel(float * kernelData, int kernelFea, int heigthWeigth, int chanal){
	//float  tempData[] = (1., 2., 1., 0., 0., 0., -1., -2., -1.);
	float  tempData2[9] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
	float  tempData3[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	//float  tempData4[9] = -1, -2, -1, 0, 0, 0, 1, 2, 1);

	float  tempData5[36] = { 1, 1, -1, -1, 2, 0, 0, -2, 1, -1, 1, -1, 0, 2, -2, 0, 0, 0, 0, 0, 0, -2, 2, 0, -1, 1, -1, 1, -2, 0, 0, 2, -1, -1, 1, 1 };
	for (int size = 0; size < 36; size++){
		tempData5[size] = tempData5[size];//   /400 +0.005;
		//tempData5[size] = 0.0005*size;
	}
	if (kernelFea == 1){
		if (heigthWeigth == 3){
			cudaMemcpy(kernelData, tempData2, 9 * sizeof(float), cudaMemcpyHostToDevice);
		}
	}
	else if (kernelFea == 4){
		if (heigthWeigth == 3){
			cudaMemcpy(kernelData, tempData5, 36 * sizeof(float), cudaMemcpyHostToDevice);
		}
	}
}

#endif