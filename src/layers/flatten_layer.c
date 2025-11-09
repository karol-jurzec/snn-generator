#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../../include/layers/flatten_layer.h"

void flatten_initialize(FlattenLayer *layer, size_t input_size) {
    layer->base.layer_type = LAYER_FLATTEN;
    layer->base.num_inputs = input_size;
    layer->base.is_spiking = false;
    layer->base.inputs = (float*)malloc(input_size * sizeof(float));
    layer->base.forward = flatten_forward;
    layer->base.output_size = input_size; 
    layer->base.output = (float *)malloc(layer->base.output_size * sizeof(float));

}

void flatten_forward(void *self, float *input, size_t input_size, size_t time_step) {
    FlattenLayer *layer = (FlattenLayer *)self;

    memcpy(layer->base.inputs, input, input_size * sizeof(float));

    for (size_t i = 0; i < input_size; i++) {
        layer->base.output[i] = input[i];
    }
}

void flatten_free(FlattenLayer *layer) {
    free(layer->output);
}
