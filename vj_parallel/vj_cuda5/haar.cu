/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   haar.cpp
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Haar features evaluation for face detection
 *
 *  History         :
 *      12-11-12    :   Initial version.
 *
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation.
 *x
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program;  If not, see <http://www.gnu.org/licenses/>
 *
 * In other words, you are welcome to use, share and improve this program.
 * You are forbidden to forbid anyone else to use, share and improve
 * what you give them.   Happy coding!
 */

#include "haar.h"
#include "image.h"
#include <stdio.h>
#include <stdlib.h>
#include "stdio-wrapper.h"

#include <chrono>
#include <iostream>





/* include the gpu functions */
#include "gpu_functions.cuh"

/* TODO: use matrices */
/* classifier parameters */
/************************************
 * Notes:
 * To paralleism the filter,
 * these monolithic arrays may
 * need to be splitted or duplicated
 ***********************************/
static int *stages_array;
static int *rectangles_array;
static int *weights_array;
static int *alpha1_array;
static int *alpha2_array;
static int *tree_thresh_array;
static int *stages_thresh_array;
static int *scaled_rectangles_array;

int* result_cu;
int *scaled_rectangles_array_cu;
int *weights_array_cu;
int *alpha1_array_cu;
int *alpha2_array_cu;
int *tree_thresh_array_cu;
int *stages_thresh_array_cu;
int *stages_array_cu;
int *sum_data_cu;
int *sqsum_data_cu;



int clock_counter = 0;
int n_features = 0;
int haar_nodes;


int iter_counter = 0;

/* compute integral images */
void integralImages( MyImage *src, MyIntImage *sum, MyIntImage *sqsum );

/* scale down the image */
void ScaleImage_Invoker( myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec);

/* compute scaled image */
void nearestNeighbor (MyImage *src, MyImage *dst);

/* rounding function */
inline  int  myRound( float value )
{
	return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

/*******************************************************
 * Function: detectObjects
 * Description: It calls all the major steps
 ******************************************************/

std::vector<MyRect> detectObjects( MyImage* _img, MySize minSize, MySize maxSize, myCascade* cascade,
					 float scaleFactor, int minNeighbors)
{

	// cudaSetDevice(0);
	/* group overlaping windows */
	const float GROUP_EPS = 0.4f;
	/* pointer to input image */
	MyImage *img = _img;
	/***********************************
	 * create structs for images
	 * see haar.h for details 
	 * img1: normal image (unsigned char)
	 * sum1: integral image (int)
	 * sqsum1: square integral image (int)
	 **********************************/
	MyImage image1Obj;
	MyIntImage sum1Obj;
	MyIntImage sqsum1Obj;
	/* pointers for the created structs */
	MyImage *img1 = &image1Obj;
	MyIntImage *sum1 = &sum1Obj;
	MyIntImage *sqsum1 = &sqsum1Obj;

	/********************************************************
	 * allCandidates is the preliminaray face candidate,
	 * which will be refined later.
	 *
	 * std::vector is a sequential container 
	 * http://en.wikipedia.org/wiki/Sequence_container_(C++) 
	 *
	 * Each element of the std::vector is a "MyRect" struct 
	 * MyRect struct keeps the info of a rectangle (see haar.h)
	 * The rectangle contains one face candidate 
	 *****************************************************/
	std::vector<MyRect> allCandidates;

	/* scaling factor */
	float factor;

	/* maxSize */
	if( maxSize.height == 0 || maxSize.width == 0 )
		{
			maxSize.height = img->height;
			maxSize.width = img->width;
		}

	/* window size of the training set */
	MySize winSize0 = cascade->orig_window_size;

	/* malloc for img1: unsigned char */
	createImage(img->width, img->height, img1);
	/* malloc for sum1: unsigned char */
	createSumImage(img->width, img->height, sum1);
	/* malloc for sqsum1: unsigned char */
	createSumImage(img->width, img->height, sqsum1);


	int imageSize = cascade->sum.width* cascade->sum.height; 
  
	cudaMalloc((void**) &tree_thresh_array_cu, sizeof(int)*haar_nodes*12);
	cudaMalloc((void**) &scaled_rectangles_array_cu, sizeof(int)*haar_nodes*12);
	cudaMalloc((void**) &weights_array_cu, sizeof(int)*haar_nodes*3);	
	cudaMalloc((void**) &alpha1_array_cu, sizeof(int)*haar_nodes);
	cudaMalloc((void**) &alpha2_array_cu, sizeof(int)*haar_nodes);
	cudaMalloc((void**) &stages_thresh_array_cu, sizeof(int)*cascade->n_stages);
	cudaMalloc((void**) &stages_array_cu, sizeof(int)*cascade->n_stages);


	/* initial scaling factor */
	factor = 1;


	/* iterate over the image pyramid */
	for( factor = 1; ; factor *= scaleFactor )
		{
			/* iteration counter */
			iter_counter++;

			/* size of the image scaled up */
			MySize winSize = { myRound(winSize0.width*factor), myRound(winSize0.height*factor) };

			/* size of the image scaled down (from bigger to smaller) */
			MySize sz = { ( img->width/factor ), ( img->height/factor ) };

			/* difference between sizes of the scaled image and the original detection window */
			MySize sz1 = { sz.width - winSize0.width, sz.height - winSize0.height };

			/* if the actual scaled image is smaller than the original detection window, break */
			if( sz1.width < 0 || sz1.height < 0 )
				break;

			/* if a minSize different from the original detection window is specified, continue to the next scaling */
			if( winSize.width < minSize.width || winSize.height < minSize.height )
				continue;

			/*************************************
			 * Set the width and height of 
			 * img1: normal image (unsigned char)
			 * sum1: integral image (int)
			 * sqsum1: squared integral image (int)
			 * see image.c for details
			 ************************************/
			setImage(sz.width, sz.height, img1);
			setSumImage(sz.width, sz.height, sum1);
			setSumImage(sz.width, sz.height, sqsum1);

			/***************************************
			 * Compute-intensive step:
			 * building image pyramid by downsampling
			 * downsampling using nearest neighbor
			 **************************************/
			nearestNeighbor(img, img1);

			/***************************************************
			 * Compute-intensive step:
			 * At each scale of the image pyramid,
			 * compute a new integral and squared integral image
			 ***************************************************/
			integralImages(img1, sum1, sqsum1);

			/* sets images for haar classifier cascade */
			/**************************************************
			 * Note:
			 * Summing pixels within a haar window is done by
			 * using four corners of the integral image:
			 * http://en.wikipedia.org/wiki/Summed_area_table
			 * 
			 * This function loads the four corners,
			 * but does not do compuation based on four coners.
			 * The computation is done next in ScaleImage_Invoker
			 *************************************************/
			setImageForCascadeClassifier( cascade, sum1, sqsum1);

			/* print out for each scale of the image pyramid */
			printf("detecting faces, iter := %d\n", iter_counter);

			/****************************************************
			 * Process the current scale with the cascaded fitler.
			 * The main computations are invoked by this function.
			 * Optimization oppurtunity:
			 * the same cascade filter is invoked each time
			 ***************************************************/
			ScaleImage_Invoker(cascade, factor, sum1->height, sum1->width,
			 allCandidates);

		} /* end of the factor loop, finish all scales in pyramid*/

	if( minNeighbors != 0)
		{
			groupRectangles(allCandidates, minNeighbors, GROUP_EPS);
		}



	cudaFree(scaled_rectangles_array_cu); 
	cudaFree(weights_array_cu); 
	cudaFree(alpha1_array_cu);
	cudaFree(alpha2_array_cu); 
  	cudaFree(tree_thresh_array_cu);

	cudaFree(stages_thresh_array_cu); 

	freeImage(img1);
	freeSumImage(sum1);
	freeSumImage(sqsum1);
	return allCandidates;

}

void setImageForCascadeClassifier( myCascade* _cascade, MyIntImage* _sum, MyIntImage* _sqsum)
{
	MyIntImage *sum = _sum;
	MyIntImage *sqsum = _sqsum;
	myCascade* cascade = _cascade;
	int i, j, k;
	MyRect equRect;
	int r_index = 0;
	int w_index = 0;
	MyRect tr;

	cascade->sum = *sum;
	cascade->sqsum = *sqsum;

	equRect.x = equRect.y = 0;
	equRect.width = cascade->orig_window_size.width;
	equRect.height = cascade->orig_window_size.height;

	cascade->inv_window_area = equRect.width*equRect.height;

	cascade->p0 = (sum->data) ;
	cascade->p1 = (sum->data +  equRect.width - 1) ;
	cascade->p2 = (sum->data + sum->width*(equRect.height - 1));
	cascade->p3 = (sum->data + sum->width*(equRect.height - 1) + equRect.width - 1);
	cascade->pq0 = (sqsum->data);
	cascade->pq1 = (sqsum->data +  equRect.width - 1) ;
	cascade->pq2 = (sqsum->data + sqsum->width*(equRect.height - 1));
	cascade->pq3 = (sqsum->data + sqsum->width*(equRect.height - 1) + equRect.width - 1);

	/****************************************
	 * Load the index of the four corners 
	 * of the filter rectangle
	 **************************************/

	/* loop over the number of stages */
	for( i = 0; i < cascade->n_stages; i++ )
	{
		/* loop over the number of haar features */
		for( j = 0; j < stages_array[i]; j++ )
		{
			int nr = 3;
			/* loop over the number of rectangles */
			for( k = 0; k < nr; k++ )
			{
				tr.x = rectangles_array[r_index + k*4];
				tr.width = rectangles_array[r_index + 2 + k*4];
				tr.y = rectangles_array[r_index + 1 + k*4];
				tr.height = rectangles_array[r_index + 3 + k*4];
				if (k < 2)
				{
					scaled_rectangles_array[r_index + k*4] = (sum->width*(tr.y ) + (tr.x )) ;
					scaled_rectangles_array[r_index + k*4 + 1] = (sum->width*(tr.y ) + (tr.x  + tr.width)) ;
					scaled_rectangles_array[r_index + k*4 + 2] = (sum->width*(tr.y  + tr.height) + (tr.x ));
					scaled_rectangles_array[r_index + k*4 + 3] = (sum->width*(tr.y  + tr.height) + (tr.x  + tr.width));
				}
				else
				{
					if ((tr.x == 0)&& (tr.y == 0) &&(tr.width == 0) &&(tr.height == 0))
					{
						scaled_rectangles_array[r_index + k*4] = 0 ;
						scaled_rectangles_array[r_index + k*4 + 1] = 0 ;
						scaled_rectangles_array[r_index + k*4 + 2] = 0;
						scaled_rectangles_array[r_index + k*4 + 3] = 0;
					}
					else
					{
						scaled_rectangles_array[r_index + k*4] = (sum->width*(tr.y ) + (tr.x )) ;
						scaled_rectangles_array[r_index + k*4 + 1] = (sum->width*(tr.y ) + (tr.x  + tr.width)) ;
						scaled_rectangles_array[r_index + k*4 + 2] = (sum->width*(tr.y  + tr.height) + (tr.x ));
						scaled_rectangles_array[r_index + k*4 + 3] = (sum->width*(tr.y  + tr.height) + (tr.x  + tr.width));
					}
				} /* end of branch if(k<2) */
			} /* end of k loop*/
			r_index+=12;
			w_index+=3;
		} /* end of j loop */
	} /* end i loop */
}

using namespace std::chrono;
void ScaleImage_Invoker( myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec)
{


	myCascade* cascade = _cascade;

	float factor = _factor;
	int y2, x2, step;
  int blockx, blocky;
	std::vector<MyRect> *vec = &_vec;

	MySize winSize0 = cascade->orig_window_size;
	MySize winSize;

	winSize.width =  myRound(winSize0.width*factor);
	winSize.height =  myRound(winSize0.height*factor);

	y2 = sum_row - winSize0.height;
	x2 = sum_col - winSize0.width;

	step = 1;

	if (x2>THREADS){
		blockx = ceil((float)x2/(float)THREADS);
	} else {
		blockx = 1;
	}

	if (y2>THREADS){
		blocky = ceil((float)y2/(float)THREADS);
	} else {
		blocky = 1;
	}

  int* result = (int*)malloc(x2*y2*sizeof(int));

	int* sqsum_data = cascade->pq0;
	int* sum_data = cascade->p0;

  int imageSize = cascade->sum.width* cascade->sum.height; 
  

  MyRect equRect;
  equRect.x = equRect.y = 0;
  equRect.width = cascade->orig_window_size.width;
	equRect.height = cascade->orig_window_size.height;
  

	int cascade_pq1_offset = equRect.width - 1;
	int cascade_pq2_offset = (cascade->sqsum.width)*(equRect.height-1);
	int cascade_pq3_offset = cascade->sqsum.width*(equRect.height-1)+equRect.width-1;

	// auto start = high_resolution_clock::now();	

	// auto stop = high_resolution_clock::now();
	// auto duration = duration_cast<microseconds>(stop - start);
	// std::cout << "Kernel Execution Time: "
	// 		 << duration.count() << " microseconds\n";
	



	cudaMalloc((void**) &result_cu, sizeof(int)*x2*y2); 
	cudaMalloc((void**) &sqsum_data_cu, sizeof(int)*imageSize);
	cudaMalloc((void**) &sum_data_cu, sizeof(int)*imageSize);


	cudaMemcpy(tree_thresh_array_cu, tree_thresh_array, sizeof(int)*haar_nodes*12, cudaMemcpyHostToDevice);
	cudaMemcpy(scaled_rectangles_array_cu, scaled_rectangles_array, sizeof(int)*haar_nodes*12, cudaMemcpyHostToDevice);
	cudaMemcpy(weights_array_cu, weights_array, sizeof(int)*haar_nodes*3, cudaMemcpyHostToDevice);
	cudaMemcpy(alpha1_array_cu, alpha1_array, sizeof(int)*haar_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(alpha2_array_cu, alpha2_array, sizeof(int)*haar_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(stages_thresh_array_cu, stages_thresh_array, sizeof(int)*cascade->n_stages, cudaMemcpyHostToDevice);
	cudaMemcpy(stages_array_cu, stages_array, sizeof(int)*cascade->n_stages, cudaMemcpyHostToDevice);

	cudaMemcpy(sqsum_data_cu, sqsum_data, sizeof(int)*imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(sum_data_cu, sum_data, sizeof(int)*imageSize, cudaMemcpyHostToDevice);

	
				 
	dim3 blocks(blockx, blocky);
	dim3 threadsPerBlock (THREADS,THREADS);


  int sum_width = cascade->sum.width;
  int sqsum_width = cascade->sqsum.width;
  int inv_window_area = cascade->inv_window_area;
  int n_stages = cascade->n_stages;



  runCascadeClassifier<<<blocks, threadsPerBlock>>>(result_cu, x2, y2,
									sqsum_data_cu, cascade_pq1_offset, cascade_pq2_offset, cascade_pq3_offset, 
									sum_data_cu, tree_thresh_array_cu, scaled_rectangles_array_cu, 
                  weights_array_cu, alpha1_array_cu, alpha2_array_cu, stages_thresh_array_cu, stages_array_cu,
                  n_stages, inv_window_area, sum_width, sqsum_width);
				
  
  


	cudaMemcpy(result, result_cu, sizeof(int)*x2*y2, cudaMemcpyDeviceToHost);



	cudaFree(result_cu);




	for(int x = 0; x < x2; x += step ) {
    	for(int y = 0; y < y2; y += step ) {

			if (result[y*x2+x] > 0)
			{
				MyRect r = {myRound(x*factor), myRound(y*factor), winSize.width, winSize.height};
				vec->push_back(r);
			}
		}
	}

}


/*****************************************************
 * Compute the integral image (and squared integral)
 * Integral image helps quickly sum up an area.
 * More info:
 * http://en.wikipedia.org/wiki/Summed_area_table
 ****************************************************/

void integralImages( MyImage *src, MyIntImage *sum, MyIntImage *sqsum )
{
  int x, y, s, sq, t, tq;
  unsigned char it;
  int height = src->height;
  int width = src->width;
  unsigned char *data = src->data;
  int * sumData = sum->data;
  int * sqsumData = sqsum->data;

  // CUDA Point
  for( y = 0; y < height; y++)
    {
      s = 0;
      sq = 0;
      //* loop over the number of columns 
      for( x = 0; x < width; x ++)
      {
        it = data[y*width+x];
        //* sum of the current row (integer)
        s += it; 
        sq += it*it;

        t = s;
        tq = sq;
        if (y != 0)
        {
          t += sumData[(y-1)*width+x];
          tq += sqsumData[(y-1)*width+x];
        }
        sumData[y*width+x]=t;
        sqsumData[y*width+x]=tq;
      }
    }
}



/***********************************************************
 * This function downsample an image using nearest neighbor
 * It is used to build the image pyramid
 **********************************************************/
void nearestNeighbor (MyImage *src, MyImage *dst)
{

	int y;
	int j;
	int x;
	int i;
	unsigned char* t;
	unsigned char* p;
	int w1 = src->width;
	int h1 = src->height;
	int w2 = dst->width;
	int h2 = dst->height;

	int rat = 0;

	unsigned char* src_data = src->data;
	unsigned char* dst_data = dst->data;


	int x_ratio = (int)((w1<<16)/w2) +1;
	int y_ratio = (int)((h1<<16)/h2) +1;


	for (i=0;i<h2;i++)
	{
		t = dst_data + i*w2;
		y = ((i*y_ratio)>>16);
		p = src_data + y*w1;
		rat = 0;
		for (j=0;j<w2;j++)
		{
			x = (rat>>16);
			*t++ = p[x];
			rat += x_ratio;
		}
	}
}

void readTextClassifier()//(myCascade * cascade)
{
	/*number of stages of the cascade classifier*/
	int stages;
	/*total number of weak classifiers (one node each)*/
	int total_nodes = 0;
	int i, j, k, l;
	char mystring [12];
	int r_index = 0;
	int w_index = 0;
	int tree_index = 0;
	FILE *finfo = fopen("info.txt", "r");

	/**************************************************
	/* how many stages are in the cascaded filter? 
	/* the first line of info.txt is the number of stages 
	/* (in the 5kk73 example, there are 25 stages)
	**************************************************/
	if ( fgets (mystring , 12 , finfo) != NULL )
		{
			stages = atoi(mystring);
		}
	i = 0;

	stages_array = (int *)malloc(sizeof(int)*stages);

	/**************************************************
	 * how many filters in each stage? 
	 * They are specified in info.txt,
	 * starting from second line.
	 * (in the 5kk73 example, from line 2 to line 26)
	 *************************************************/
	while ( fgets (mystring , 12 , finfo) != NULL )
		{
			stages_array[i] = atoi(mystring);
			total_nodes += stages_array[i];
			i++;
		}

	fclose(finfo);


  haar_nodes = total_nodes;


	/* TODO: use matrices where appropriate */
	/***********************************************
	 * Allocate a lot of array structures
	 * Note that, to increase parallelism,
	 * some arrays need to be splitted or duplicated
	 **********************************************/
	rectangles_array = (int *)malloc(sizeof(int)*total_nodes*12);
	scaled_rectangles_array = (int *)malloc(sizeof(int)*total_nodes*12);
	weights_array = (int *)malloc(sizeof(int)*total_nodes*3);
	alpha1_array = (int*)malloc(sizeof(int)*total_nodes);
	alpha2_array = (int*)malloc(sizeof(int)*total_nodes);
	tree_thresh_array = (int*)malloc(sizeof(int)*total_nodes);
	stages_thresh_array = (int*)malloc(sizeof(int)*stages);
	FILE *fp = fopen("class.txt", "r");

	/******************************************
	 * Read the filter parameters in class.txt
	 *
	 * Each stage of the cascaded filter has:
	 * 18 parameter per filter x tilter per stage
	 * + 1 threshold per stage
	 *
	 * For example, in 5kk73, 
	 * the first stage has 9 filters,
	 * the first stage is specified using
	 * 18 * 9 + 1 = 163 parameters
	 * They are line 1 to 163 of class.txt
	 *
	 * The 18 parameters for each filter are:
	 * 1 to 4: coordinates of rectangle 1
	 * 5: weight of rectangle 1
	 * 6 to 9: coordinates of rectangle 2
	 * 10: weight of rectangle 2
	 * 11 to 14: coordinates of rectangle 3
	 * 15: weight of rectangle 3
	 * 16: threshold of the filter
	 * 17: alpha 1 of the filter
	 * 18: alpha 2 of the filter
	 ******************************************/

	/* loop over n of stages */
	for (i = 0; i < stages; i++)
		{    /* loop over n of trees */
			for (j = 0; j < stages_array[i]; j++)
				 {  /* loop over n of rectangular features */
					 for(k = 0; k < 3; k++)
						 {  /* loop over the n of vertices */
							 for (l = 0; l <4; l++)
								{
									if (fgets (mystring , 12 , fp) != NULL)
										rectangles_array[r_index] = atoi(mystring);
									else
										break;
									r_index++;
								} /* end of l loop */
								 if (fgets (mystring , 12 , fp) != NULL)
									{
										weights_array[w_index] = atoi(mystring);
										/* Shift value to avoid overflow in the haar evaluation */
										/*TODO: make more general */
										/*weights_array[w_index]>>=8; */
									}
								 else
									break;
							 w_index++;
						 } /* end of k loop */
						if (fgets (mystring , 12 , fp) != NULL)
							tree_thresh_array[tree_index]= atoi(mystring);
						else
							break;
						if (fgets (mystring , 12 , fp) != NULL)
							alpha1_array[tree_index]= atoi(mystring);
						else
							break;
						if (fgets (mystring , 12 , fp) != NULL)
							alpha2_array[tree_index]= atoi(mystring);
						else
							break;
						tree_index++;
						if (j == stages_array[i]-1)
							{
							 if (fgets (mystring , 12 , fp) != NULL)
								stages_thresh_array[i] = atoi(mystring);
							 else
								break;
							 }  
				 } /* end of j loop */
		} /* end of i loop */
	fclose(fp);
}


void releaseTextClassifier()
{
	free(stages_array);
	free(rectangles_array);
	free(scaled_rectangles_array);
	free(weights_array);
	free(tree_thresh_array);
	free(alpha1_array);
	free(alpha2_array);
	free(stages_thresh_array);
}