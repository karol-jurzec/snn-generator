#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include <stddef.h>

#include "layer_base.h"

// Structure for the Flatten layer
typedef struct {
    LayerBase base;          // Inherits LayerBase for polymorphic forward
    float *output;           // Output buffer (1D flattened output)
    float *input_gradients;            // Input gradients for backpropagation
    size_t output_size;      // Size of the flattened output
} FlattenLayer;

// Function declarations
void flatten_initialize(FlattenLayer *layer, size_t input_size);
void flatten_forward(void *self, float *input, size_t input_size);
float* flatten_backward(void *self, float *gradients); // Simply reshapes gradients back
void flatten_free(FlattenLayer *layer);
void flatten_zero_grad(void *self);

#endif // FLATTEN_LAYER_H
