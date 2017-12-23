#include <cuda_runtime.h>
#include "ConnectCuda.h"

void FullyWeigthFullyRegulatorCu(
	float* inputWeLayerTensor, int inpLaySamp, int inpLayFea,
	int inpLayHe, int inpLayWi,int inpLayCh, 
	float * regulatorLayerTensor,int regLayFea, int regLayHe,
	int regLayWi,int regLayCh,
	float * outLayerTensor,int outLayFe, int outLayHe,
	int outLayWi,int outLayCh,////*************************************************kernelWeigthpast ekle 
	float* kernelWeigthTensor,//////////******bu pasta  reg ve act çarpýmýsonucu toplamý kaydet ,changeWe yaparken kolaylýk olsun 
	float* kernelPastTensor,
	float* kernelActivatorTensor,float * biasData){

	
};


/*buna benzer fully connect witt activitation için  4-4-3 boyurlu input-regulator-output  layerlarýndan
oluþan 3 baðlantýlý konnect alan ve input-inputweigth-regulator-regulatorActivite-output datalarý olan bir 
connect sýnýfý lazým bunlarý iþlem yaparken özellikle input -regulator kýsmý için shared meymory kullanma 
þansýný kullanmak gerekli olacak gibi durutyor */