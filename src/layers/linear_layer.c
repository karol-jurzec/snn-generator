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
    layer->base.backward = linear_backward;
    layer->base.zero_grad = linear_zero_grad;  // Assign function pointer
    layer->base.update_weights = linear_update_weights;
    layer->base.num_inputs = in_features;
    layer->base.inputs = (float*)malloc(in_features * sizeof(float));
    layer->in_features = in_features;
    layer->out_features = out_features;

    layer->base.weights = (float *)malloc(in_features * out_features * sizeof(float));
    layer->base.num_weights = in_features * out_features;
    layer->biases = (float *)malloc(out_features * sizeof(float));
    layer->base.output = (float *)malloc(out_features * sizeof(float));
    layer->base.output_size = out_features;

    // weights and bias initalization

    he_kaiming_uniform_init_linear(layer->base.weights, in_features * out_features, in_features);
    initialize_biases(layer->biases, out_features, in_features);

    layer->base.weight_gradients = (float *)malloc(layer->base.num_weights * sizeof(float));
    layer->base.bias_gradients = (float *)malloc(out_features * sizeof(float));
    layer->base.input_gradients = (float *)malloc(in_features * sizeof(float));
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

float* linear_backward(void *self, float *gradients, size_t time_step) {
    LinearLayer *L = (LinearLayer*)self;

    // dW += dY * x^T
    for (size_t o = 0; o < L->out_features; o++) {
        size_t row_offset = o * L->in_features;
        float g = gradients[o];
        for (size_t i = 0; i < L->in_features; i++) {
            L->base.weight_gradients[row_offset + i] += g * L->base.inputs[i];
        }
    }

    // db += dY
    for (size_t o = 0; o < L->out_features; o++) {
        L->base.bias_gradients[o] += gradients[o];
    }

    // dX = W^T * dY
    for (size_t i = 0; i < L->in_features; i++) {
        float acc = 0.0f;
        for (size_t o = 0; o < L->out_features; o++) {
            acc += L->base.weights[o * L->in_features + i] * gradients[o];
        }
        L->base.input_gradients[i] = acc;
    }

    return L->base.input_gradients;
}

// Update weights for Linear Layer
void linear_update_weights(void *self, float learning_rate) {
    LinearLayer *layer = (LinearLayer *)self;
    
    // Update weights
    for (size_t i = 0; i < layer->in_features * layer->out_features; i++) {
        layer->base.weights[i] -= learning_rate * layer->base.weight_gradients[i];
    }
    
    // Update biases
    for (size_t i = 0; i < layer->out_features; i++) {
        layer->biases[i] -= learning_rate * layer->base.bias_gradients[i];
    }
}

void linear_zero_grad(void *self) {
    LinearLayer *layer = (LinearLayer *)self;
    memset(layer->base.weight_gradients, 0, sizeof(float) * layer->in_features * layer->out_features);
    memset(layer->base.bias_gradients, 0, sizeof(float) * layer->out_features);
    memset(layer->base.input_gradients, 0, sizeof(float) * layer->in_features);
}

// Free allocated memory
void linear_free(LinearLayer *layer) {
    free(layer->base.weights);
    free(layer->biases);
    free(layer->base.weight_gradients);
    free(layer->base.bias_gradients);
    free(layer->base.input_gradients);
    free(layer->base.output);
}

