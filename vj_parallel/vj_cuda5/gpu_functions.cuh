
#include <stdio.h>

__global__  void integralImageRows(int *sumdata_cuda, int *sqsumdata_cuda, unsigned char *data_cuda, int width, int height){

	int id = blockIdx.x*256+threadIdx.x;

	int i;
	int temp;

	int sum = 0, sq_sum = 0;

	if (id < height){
		for (i = 0; i<width; i++){
			temp = data_cuda[id*width + i];
			sum += temp;
			sq_sum += temp*temp;
			sumdata_cuda[id*width + i] += temp;
			sqsumdata_cuda[id*width + i] += temp * temp;
		}
	}
}

__global__  void integralImageCols(int *sumdata_cuda, int *sqsumdata_cuda, int width, int height){	

	int id = blockIdx.x*256+threadIdx.x;

	int i;

	if (id < width){
		for (i = 1; i<height; i++){
			sumdata_cuda[width*i + id] += sumdata_cuda[width*(i-1) + id];
			sqsumdata_cuda[width*i + id] += sqsumdata_cuda[width*(i-1) + id];
		}
	}

}

