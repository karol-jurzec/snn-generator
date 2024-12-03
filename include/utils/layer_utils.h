#ifndef LAYER_UTILS_H
#define LAYER_UTILS_H

#include <stddef.h>

// Function to calculate output dimensions after convolution/pooling
size_t calculate_output_dim(size_t input_dim, int kernel_size, int stride, int padding);

#endif // LAYER_UTILS_H
