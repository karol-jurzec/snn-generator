#ifndef LAYER_UTILS_H
#define LAYER_UTILS_H

#include <stddef.h>
#include <math.h>
#include <stdlib.h>

size_t calculate_output_dim(size_t input_dim, int kernel_size, int stride, int padding);
void initialize_biases(float *biases, size_t size, int fan_in);

#endif // LAYER_UTILS_H
