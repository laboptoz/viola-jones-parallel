#include "gpu_functions.cuh"
#include <stdio.h>

__global__ void runCascadeClassifier(int* result_cu,int x2, int y2,
									int* sqsum_data_cu, int cascade_pq1_offset, int cascade_pq2_offset, int cascade_pq3_offset, 
									int* sum_data_cu, int* tree_thresh_array_cu, int* scaled_rectangles_array_cu, int* weights_array_cu, int* alpha1_array_cu, int* alpha2_array_cu, int* stages_thresh_array_cu, int* stages_array_cu,
                                    int n_stages, int inv_window_area, int sum_width, int sqsum_width)
{

	int p_offset, pq_offset;
	int i, j;
	unsigned int mean;
	unsigned int variance_norm_factor;
	int haar_counter = 0;
	int w_index = 0;
	int r_index = 0;
	int stage_sum;
	int finished = 0;

    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;


    int* cascade_p1_cu = sum_data_cu + cascade_pq1_offset;
	int* cascade_p2_cu = sum_data_cu + cascade_pq2_offset;
	int* cascade_p3_cu = sum_data_cu + cascade_pq3_offset;

	int* cascade_pq1_cu = sqsum_data_cu + cascade_pq1_offset;
	int* cascade_pq2_cu = sqsum_data_cu + cascade_pq2_offset;
	int* cascade_pq3_cu = sqsum_data_cu + cascade_pq3_offset;

	if (x <= x2 && y <= y2){

		int idx = y*x2 + x;
		
		p_offset = y * (sum_width) + x;
		pq_offset = y * (sqsum_width) + x;

		variance_norm_factor =  (sqsum_data_cu[pq_offset] - cascade_pq1_cu[pq_offset] - cascade_pq2_cu[pq_offset] + cascade_pq3_cu[pq_offset]);
		mean = (sum_data_cu[p_offset] - cascade_p1_cu[p_offset] - cascade_p2_cu[p_offset] + cascade_p3_cu[p_offset]);

		variance_norm_factor = (variance_norm_factor*inv_window_area);
		variance_norm_factor =  variance_norm_factor - mean*mean;

		if( variance_norm_factor > 0 )
			variance_norm_factor = (int)sqrtf((float)variance_norm_factor);	
		else
			variance_norm_factor = 1;

		for( i = 0; i < n_stages; i++ )
			{

				stage_sum = 0;

				for( j = 0; j < stages_array_cu[i]; j++ )
				{

					
					int t = tree_thresh_array_cu[haar_counter] * variance_norm_factor;
					int sum;


					sum = (*(sum_data_cu + scaled_rectangles_array_cu[r_index] + p_offset)
							 - *(sum_data_cu + scaled_rectangles_array_cu[r_index + 1] + p_offset)
							 - *(sum_data_cu + scaled_rectangles_array_cu[r_index + 2] + p_offset)
							 + *(sum_data_cu + scaled_rectangles_array_cu[r_index + 3] + p_offset))
						* weights_array_cu[w_index];


					sum += (*(sum_data_cu + scaled_rectangles_array_cu[r_index+4] + p_offset)
						- *(sum_data_cu + scaled_rectangles_array_cu[r_index + 5] + p_offset)
						- *(sum_data_cu + scaled_rectangles_array_cu[r_index + 6] + p_offset)
						+ *(sum_data_cu + scaled_rectangles_array_cu[r_index + 7] + p_offset))
						* weights_array_cu[w_index + 1];

					if ((scaled_rectangles_array_cu[r_index+8] != 0))
						sum += (*(sum_data_cu + scaled_rectangles_array_cu[r_index+8] + p_offset)
							- *(sum_data_cu + scaled_rectangles_array_cu[r_index + 9] + p_offset)
							- *(sum_data_cu + scaled_rectangles_array_cu[r_index + 10] + p_offset)
							+ *(sum_data_cu + scaled_rectangles_array_cu[r_index + 11] + p_offset))
							* weights_array_cu[w_index + 2];

					if(sum >= t)
						stage_sum += alpha2_array_cu[haar_counter];
					else
						stage_sum += alpha1_array_cu[haar_counter];

				
					haar_counter++;
					w_index+=3;
					r_index+=12;
				}  


				if( stage_sum < 0.4*stages_thresh_array_cu[i] ){
				 	result_cu[idx] =  -i;
				 	finished = 1;

				 	break;
				}
			} 

		if (finished == 0)
			result_cu[idx] = 1;

	}

    __syncthreads();

    

}
