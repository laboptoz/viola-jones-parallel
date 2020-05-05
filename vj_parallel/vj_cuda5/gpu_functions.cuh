#define THREADS 16



__global__ void runCascadeClassifier(int* result_cu, int x2, int y2,
									int* sqsum_data_cu, int cascade_pq1_offset, int cascade_pq2_offset, int cascade_pq3_offset, 
									int* sum_data_cu, int* tree_thresh_array_cu, int* scaled_rectangles_array_cu, int* weights_array_cu, int* alpha1_array_cu, int* alpha2_array_cu, int* stages_thresh_array_cu, int* stages_array_cu,
									int n_stages, int inv_window_area, int sum_width, int sqsum_width);
