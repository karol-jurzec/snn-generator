#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../../include/layers/linear_layer.h"

void linear_initialize(LinearLayer *layer, size_t in_features, size_t out_features) {
    layer->base.forward = linear_forward;
    layer->in_features = in_features;
    layer->out_features = out_features;

    layer->weights = (float *)malloc(in_features * out_features * sizeof(float));
    layer->biases = (float *)malloc(out_features * sizeof(float));
    layer->base.output = (float *)malloc(out_features * sizeof(float));
    layer->base.output_size = out_features;

    // Initialize weights and biases (random small values for weights, zeros for biases)
    for (size_t i = 0; i < in_features * out_features; i++) {
        layer->weights[i] = ((float)rand() / RAND_MAX) * 0.1f;
    }
    for (size_t i = 0; i < out_features; i++) {
        layer->biases[i] = 0.0f;
    }

    layer->weight_gradients = (float *)malloc(in_features * out_features * sizeof(float));
    layer->bias_gradients = (float *)malloc(out_features * sizeof(float));
    layer->base.input_gradients = NULL;
}

void linear_forward(void *self, float *input, size_t input_size) {
    LinearLayer *layer = (LinearLayer *)self;
    layer->input = input;

    //printf("Performing Linear forward pass...\n");

    for (size_t o = 0; o < layer->out_features; o++) {
        float sum = 0.0f;
        for (size_t i = 0; i < layer->in_features; i++) {
            sum += input[i] * layer->weights[o * layer->in_features + i];
        }
        layer->base.output[o] = sum + layer->biases[o];
    }
}

// Backward pass for Linear Layer
void linear_backward(void *self, float *gradients) {
    LinearLayer *layer = (LinearLayer *)self;

    // Gradients for weights and biases
    memset(layer->weight_gradients, 0, sizeof(float) * layer->in_features * layer->out_features);
    memset(layer->bias_gradients, 0, sizeof(float) * layer->out_features);

    for (size_t i = 0; i < layer->out_features; i++) {
        for (size_t j = 0; j < layer->in_features; j++) {
            layer->weight_gradients[i * layer->in_features + j] += gradients[i] * layer->input[j];
        }
        layer->bias_gradients[i] += gradients[i];
    }

    // Input gradients
    layer->base.input_gradients = (float *)realloc(layer->base.input_gradients, layer->in_features * sizeof(float));
    memset(layer->base.input_gradients, 0, layer->in_features * sizeof(float));

    for (size_t j = 0; j < layer->in_features; j++) {
        for (size_t i = 0; i < layer->out_features; i++) {
            layer->base.input_gradients[j] += layer->weights[i * layer->in_features + j] * gradients[i];
        }
    }
}

// Update weights for Linear Layer
void linear_update_weights(void *self, float learning_rate) {
    LinearLayer *layer = (LinearLayer *)self;

    for (size_t i = 0; i < layer->in_features * layer->out_features; i++) {
        layer->weights[i] -= learning_rate * layer->weight_gradients[i];
    }
    for (size_t i = 0; i < layer->out_features; i++) {
        layer->biases[i] -= learning_rate * layer->bias_gradients[i];
    }
}

// Free allocated memory
void linear_free(LinearLayer *layer) {
    free(layer->weights);
    free(layer->biases);
    free(layer->weight_gradients);
    free(layer->bias_gradients);
    free(layer->base.input_gradients);
    free(layer->output);
}

