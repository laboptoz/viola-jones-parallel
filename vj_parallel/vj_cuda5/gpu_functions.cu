#include "gpu_functions.cuh"
#include <stdio.h>

__global__  void integralImageRows(int *sumdata_cu, int *sqsumdata_cu, unsigned char *data_cu, int width, int height){

	int id = blockIdx.x*THREADS+threadIdx.x;
	unsigned char temp;

	int sum = 0, sq_sum = 0;

	if (id < height){
		for (int i = 0; i<width; i++){
			temp = data_cu[id*width + i];
			sum += temp;
			sq_sum += temp*temp;
			sumdata_cu[id*width + i] = sum;
			sqsumdata_cu[id*width + i] = sq_sum;
		}
	}
}

__global__  void integralImageCols(int *sumdata_cu, int *sqsumdata_cu, int width, int height){	

	int id = blockIdx.x*THREADS+threadIdx.x;

	if (id < width){
		for (int i = 1; i<height; i++){
			sumdata_cu[width*i + id] += sumdata_cu[width*(i-1) + id];
			sqsumdata_cu[width*i + id] += sqsumdata_cu[width*(i-1) + id];
		}
	}

}

