#ifndef MAXPOOL2D_LAYER_H
#define MAXPOOL2D_LAYER_H

#include <stddef.h>
#include "layer_base.h"

// Structure for the MaxPool2D Layer
typedef struct {
    LayerBase base;         // Inherits LayerBase for polymorphic updates
    int kernel_size;        // Kernel dimensions (assuming square kernels)
    int stride;             // Stride size
    int padding;            // Padding size
    float *output;          // Output feature map (after pooling)
    size_t output_size;     // Size of the output feature map

      // Training-specific
    size_t *max_indices;    // Stores the indices of max values for backpropagation
    float *input_gradients; // Gradient of input to propagate backward
} MaxPool2DLayer;

// Function declarations
void maxpool2d_initialize(MaxPool2DLayer *layer, int kernel_size, int stride, int padding);
void maxpool2d_forward(void *self, float *input, size_t input_size);
void maxpool2d_backward(void *self, float *gradients);
void maxpool2d_free(MaxPool2DLayer *layer);

#endif // MAXPOOL2D_LAYER_H
