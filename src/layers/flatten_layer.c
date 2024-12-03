#include <stdlib.h>
#include <stdio.h>

#include "../../include/layers/flatten_layer.h"

// Initialize Flatten layer with the expected input size
void flatten_initialize(FlattenLayer *layer, size_t input_size) {
    layer->base.forward = flatten_forward;
    layer->output_size = input_size;  // Output size matches input size (flattening does not change total number of elements)
    layer->output = (float *)malloc(input_size * sizeof(float));
}

// Forward pass: simply copy input to output in a linear (flattened) fashion
void flatten_forward(void *self, float *input, size_t input_size) {
    FlattenLayer *layer = (FlattenLayer *)self;

    printf("Performing Flatten forward pass...\n");

    // Copy input to output (flattened)
    for (size_t i = 0; i < input_size; i++) {
        layer->output[i] = input[i];
    }
}

// Free allocated memory
void flatten_free(FlattenLayer *layer) {
    free(layer->output);
}
