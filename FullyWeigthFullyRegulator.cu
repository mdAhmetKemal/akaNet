#include <cuda_runtime.h>
#include "ConnectCuda.h"

void FullyWeigthFullyRegulatorCu(
	float* inputWeLayerTensor, int inpLaySamp, int inpLayFea,
	int inpLayHe, int inpLayWi,int inpLayCh, 
	float * regulatorLayerTensor,int regLayFea, int regLayHe,
	int regLayWi,int regLayCh,
	float * outLayerTensor,int outLayFe, int outLayHe,
	int outLayWi,int outLayCh,////*************************************************kernelWeigthpast ekle 
	float* kernelWeigthTensor,//////////******bu pasta  reg ve act �arp�m�sonucu toplam� kaydet ,changeWe yaparken kolayl�k olsun 
	float* kernelPastTensor,
	float* kernelActivatorTensor,float * biasData){

	
};


/*buna benzer fully connect witt activitation i�in  4-4-3 boyurlu input-regulator-output  layerlar�ndan
olu�an 3 ba�lant�l� konnect alan ve input-inputweigth-regulator-regulatorActivite-output datalar� olan bir 
connect s�n�f� laz�m bunlar� i�lem yaparken �zellikle input -regulator k�sm� i�in shared meymory kullanma 
�ans�n� kullanmak gerekli olacak gibi durutyor */