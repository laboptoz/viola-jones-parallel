#include "gpu_functions.cuh"
#include <stdio.h>

__global__  void evalWeakClassifier(int variance_norm_factor, int p_offset, int tree_index, int w_index, int r_index, int* stage_sum_array_cu, int* stages_array_cu,
									int* tree_thresh_array_cu, int* scaled_rectangles_array_cu, int* weights_array_cu, int* alpha1_array_cu,
                                    int* alpha2_array_cu, int* sum_data_cu){
	for(int j = 0; j < stages_array_cu[threadIdx.x]; j++ ){
		int t = tree_thresh_array_cu[tree_index] * variance_norm_factor;

		int sum = *(sum_data_cu + scaled_rectangles_array_cu[r_index] + p_offset)
				 - *(sum_data_cu + scaled_rectangles_array_cu[r_index + 1] + p_offset)
		 		 - *(sum_data_cu + scaled_rectangles_array_cu[r_index + 2] + p_offset)
		 		 + *(sum_data_cu + scaled_rectangles_array_cu[r_index + 3] + p_offset)
    			 * weights_array_cu[w_index];

    	sum += (*(sum_data_cu + scaled_rectangles_array_cu[r_index+4] + p_offset)
            - *(sum_data_cu + scaled_rectangles_array_cu[r_index+5] + p_offset)
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
			stage_sum_array_cu[threadIdx.x]+=alpha2_array_cu[tree_index];
 		else
			stage_sum_array_cu[threadIdx.x]+= alpha1_array_cu[tree_index];
	}

	tree_index++;
	w_index+=3;
	r_index+=12;

}

