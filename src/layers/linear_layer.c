#include <stdlib.h>
#include <stdio.h>

#include "../../include/layers/linear_layer.h"

// Initialize Linear layer with random weights and biases
void linear_initialize(LinearLayer *layer, size_t in_features, size_t out_features) {
    layer->base.forward = linear_forward;
    layer->in_features = in_features;
    layer->out_features = out_features;

    layer->weights = (float *)malloc(in_features * out_features * sizeof(float));
    layer->biases = (float *)malloc(out_features * sizeof(float));
    layer->output = (float *)malloc(out_features * sizeof(float));

    // Initialize weights and biases (random small values for weights, zeros for biases)
    for (size_t i = 0; i < in_features * out_features; i++) {
        layer->weights[i] = ((float)rand() / RAND_MAX) * 0.1f;
    }
    for (size_t i = 0; i < out_features; i++) {
        layer->biases[i] = 0.0f;
    }
}

// Forward pass: perform matrix multiplication and add biases
void linear_forward(void *self, float *input, size_t input_size) {
    LinearLayer *layer = (LinearLayer *)self;

    printf("Performing Linear forward pass...\n");

    for (size_t o = 0; o < layer->out_features; o++) {
        float sum = 0.0f;
        for (size_t i = 0; i < layer->in_features; i++) {
            sum += input[i] * layer->weights[o * layer->in_features + i];
        }
        layer->output[o] = sum + layer->biases[o];
    }
}

// Free allocated memory
void linear_free(LinearLayer *layer) {
    free(layer->weights);
    free(layer->biases);
    free(layer->output);
}
