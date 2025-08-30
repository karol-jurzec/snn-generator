#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../../include/layers/linear_layer.h"
#include "../../include/utils/layer_utils.h" 


void he_kaiming_uniform_init_linear(float *weights, size_t size, int fan_in) {
    float limit = sqrtf(6.0f / fan_in);
    for (size_t i = 0; i < size; i++) {
        float rand_val = (float)rand() / RAND_MAX; // [0, 1]
        weights[i] = (2.0f * rand_val - 1.0f) * limit; // Scale to [-limit, limit]
    }
}

void linear_initialize(LinearLayer *layer, size_t in_features, size_t out_features) {
    layer->base.layer_type = LAYER_LINEAR;
    layer->base.forward = linear_forward;
    layer->base.is_spiking = false;
    layer->base.num_inputs = in_features;
    layer->base.inputs = (float*)malloc(in_features * sizeof(float));
    layer->in_features = in_features;
    layer->out_features = out_features;

    layer->base.weights = (float *)malloc(in_features * out_features * sizeof(float));
    layer->base.num_weights = in_features * out_features;
    layer->biases = (float *)malloc(out_features * sizeof(float));
    layer->base.output = (float *)malloc(out_features * sizeof(float));
    layer->base.output_size = out_features;
}

void linear_forward(void *self, float *input, size_t input_size, size_t time_step) {
    LinearLayer *L = (LinearLayer*)self;
    memcpy(L->base.inputs, input, input_size * sizeof(float));

    for (size_t o = 0; o < L->out_features; o++) {
        float acc = L->biases[o];
        size_t row_offset = o * L->in_features;
        for (size_t i = 0; i < L->in_features; i++) {
            acc += L->base.weights[row_offset + i] * input[i];
        }
        L->base.output[o] = acc;
    }
}

void linear_free(LinearLayer *layer) {
    free(layer->base.weights);
    free(layer->biases);
    free(layer->base.output);
}

