#ifndef LAYER_UTILS_H
#define LAYER_UTILS_H

#include <stddef.h>
#include <math.h>
#include <stdlib.h>

size_t calculate_output_dim(size_t input_dim, int kernel_size, int stride, int padding);
void initialize_biases(float *biases, size_t size, int fan_in);
void im2col(float* data_im, int channels, int height, int width,
    int kernel_size, int padding, int stride,
    float* data_col, int output_h, int output_w);
void col2im(const float *data_col,
    int channels, int height, int width,
    int kernel_size, int padding, int stride,
    float *data_im, int output_h, int output_w);
    
void im2col_pruned(float* input, int in_channels, int input_dim,
        int kernel_size, int padding, int stride,
        float* col, int output_h, int output_w,
        const bool* active_channels);

#endif // LAYER_UTILS_H
