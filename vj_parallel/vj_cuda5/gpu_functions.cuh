#define THREADS 256

__global__  void integralImageRows(int *sumdata_cuda, int *sqsumdata_cuda, unsigned char *data_cuda, int width, int height);

__global__  void integralImageCols(int *sumdata_cuda, int *sqsumdata_cuda, int width, int height);

