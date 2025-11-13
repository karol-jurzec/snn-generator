#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <stddef.h>
#include "layer_base.h"
#include <stdbool.h>


// Structure for the Linear layer
typedef struct {
    LayerBase base;          // Inherits LayerBase for polymorphic forward
    size_t in_features;      // Number of input features
    size_t out_features;     // Number of output features
    float *weights;          // Weight matrix
    float *biases;           // Bias vector
    float *output;           // Output vector

    int *in_active_channels_idx;
    int *out_active_channels_idx;

    float *weight_mat;
    int cache_IC;
    int cache_OC;
    bool weight_mat_valid;

    float *input;            // Input buffer
} LinearLayer;


void linear_initialize(LinearLayer *layer, size_t in_features, size_t out_features);
void linear_forward(void *self, float *input, size_t input_size, size_t time_step);
void linear_free(LinearLayer *layer);

#endif // LINEAR_LAYER_H
