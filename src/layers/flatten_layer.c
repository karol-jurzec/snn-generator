#include <stdlib.h>
#include <stdio.h>

#include "../../include/layers/flatten_layer.h"

void flatten_initialize(FlattenLayer *layer, size_t input_size) {
    layer->base.forward = flatten_forward;
    layer->base.output_size = input_size;  // Output size matches input size (flattening does not change total number of elements)
    layer->base.output = (float *)malloc(layer->base.output_size * sizeof(float));
}

void flatten_forward(void *self, float *input, size_t input_size) {
    FlattenLayer *layer = (FlattenLayer *)self;

    printf("Performing Flatten forward pass...\n");

    for (size_t i = 0; i < input_size; i++) {
        layer->base.output[i] = input[i];
    }
}

void flatten_free(FlattenLayer *layer) {
    free(layer->output);
}
