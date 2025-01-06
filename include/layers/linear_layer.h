#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <stddef.h>
#include "layer_base.h"

// Structure for the Linear layer
typedef struct {
    LayerBase base;          // Inherits LayerBase for polymorphic forward
    size_t in_features;      // Number of input features
    size_t out_features;     // Number of output features
    float *weights;          // Weight matrix
    float *biases;           // Bias vector
    float *output;           // Output vector

    // Training-specific
    float *weight_gradients; // Gradient of weights
    float *bias_gradients;   // Gradient of biases
    float *input_gradients;  // Gradient of input to propagate backward
    float *input;            // Input buffer
} LinearLayer;

// Function declarations
void linear_initialize(LinearLayer *layer, size_t in_features, size_t out_features);
void linear_forward(void *self, float *input, size_t input_size);
void linear_backward(void *self, float *gradients);
void linear_update_weights(void *self, float learning_rate);
void linear_free(LinearLayer *layer);

#endif // LINEAR_LAYER_H
