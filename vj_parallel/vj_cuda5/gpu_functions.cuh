
__global__  void evalWeakClassifier(unsigned int variance_norm_factor, int p_offset, int tree_index, int w_index, int r_index, int* stage_sum_array_cu, int* stages_array, 
                                    int* tree_thresh_array_cu, int* scaled_rectangles_array_cu, int* weights_array_cu, int* alpha1_array_cu,
                                    int* alpha2_array_cu, int* sum_data_cu, int* stages_thresh_array_cu);
