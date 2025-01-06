#ifndef CONV2D_LAYER_H
#define CONV2D_LAYER_H

#include <stddef.h>
#include "layer_base.h"

typedef struct {
    LayerBase base;          // Inherits LayerBase for polymorphic forward
    int in_channels;         // Number of input channels
    int out_channels;        // Number of output channels
    int kernel_size;         // Kernel dimensions (assuming square kernels)
    int stride;              // Stride size
    int padding;             // Padding size
    float *weights;          // Pointer to weight matrix
    float *biases;           // Pointer to biases
    float *output;           // Output feature map (after convolution)
    size_t output_size;      // Size of the output feature map

    float *weight_gradients; // Gradient of weights
    float *bias_gradients;   // Gradient of biases
    float *input_gradients;  // Gradient of input to propagate backward

    float *input;
} Conv2DLayer;

void conv2d_initialize(Conv2DLayer *layer, int in_channels, int out_channels, int kernel_size, int stride, int padding);
void conv2d_forward(void *self, float *input, size_t input_size);
void conv2d_backward(void *self, float *gradients);
void conv2d_update_weights(void *self, float learning_rate);
void conv2d_free(Conv2DLayer *layer);

#endif // CONV2D_LAYER_H
