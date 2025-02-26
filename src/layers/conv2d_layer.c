#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../../include/utils/layer_utils.h" 
#include "../../include/layers/conv2d_layer.h"

void conv2d_initialize(Conv2DLayer *layer, int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    size_t input_dim = 28;

    layer->base.forward = conv2d_forward;
    layer->base.backward = conv2d_backward;
    layer->base.update_weights = conv2d_update_weights;
    layer->base.output_size = calculate_output_dim(input_dim, kernel_size, stride, padding) *
                              calculate_output_dim(input_dim, kernel_size, stride, padding) *
                              out_channels;
    layer->base.output = (float *)malloc(layer->base.output_size * sizeof(float));

    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;

    size_t weight_size = in_channels * out_channels * kernel_size * kernel_size;
    layer->base.weights = (float *)malloc(weight_size * sizeof(float));
    layer->base.num_weights = weight_size;
    layer->biases = (float *)malloc(out_channels * sizeof(float));

    for (size_t i = 0; i < weight_size; i++) {
        layer->base.weights[i] = ((float)rand() / RAND_MAX) * 0.1f;
    }
    for (size_t i = 0; i < out_channels; i++) {
        layer->biases[i] = 0.0f;
    }

    layer->base.weight_gradients = (float *)malloc(weight_size * sizeof(float));
    layer->base.bias_gradients = (float *)malloc(out_channels * sizeof(float));
    layer->base.input_gradients = NULL;
    layer->input = NULL;
}

void conv2d_forward(void *self, float *input, size_t input_size) {
    Conv2DLayer *layer = (Conv2DLayer *)self;
    size_t output_dim = calculate_output_dim(28, layer->kernel_size, layer->stride, layer->padding);

    layer->input = input;

    for (int oc = 0; oc < layer->out_channels; oc++) {
        for (size_t oy = 0; oy < output_dim; oy++) {
            for (size_t ox = 0; ox < output_dim; ox++) {
                float sum = 0.0f;
                for (int ic = 0; ic < layer->in_channels; ic++) {
                    for (int ky = 0; ky < layer->kernel_size; ky++) {
                        for (int kx = 0; kx < layer->kernel_size; kx++) {
                            size_t ix = ox * layer->stride + kx - layer->padding;
                            size_t iy = oy * layer->stride + ky - layer->padding;
                            if (ix < 34 && iy < 34) {
                                size_t input_idx = (ic * input_size) + (iy * 34 + ix);
                                size_t weight_idx = (((oc * layer->in_channels + ic) * layer->kernel_size + ky) * layer->kernel_size + kx);
                                sum += input[input_idx] * layer->base.weights[weight_idx];
                            }
                        }
                    }
                }
                size_t output_idx = (oc * output_dim * output_dim) + (oy * output_dim + ox);
                layer->base.output[output_idx] = sum + layer->biases[oc];
            }
        }
    }
}

void conv2d_backward(void *self, float *gradients) {
    Conv2DLayer *layer = (Conv2DLayer *)self;

    memset(layer->base.weight_gradients, 0, sizeof(float) * layer->in_channels * layer->out_channels * layer->kernel_size * layer->kernel_size);
    memset(layer->base.bias_gradients, 0, sizeof(float) * layer->out_channels);

    size_t input_dim = 34;  // Assuming square input
    size_t output_dim = calculate_output_dim(input_dim, layer->kernel_size, layer->stride, layer->padding);

    // Allocate memory for input gradients
    if (!layer->base.input_gradients) {
        layer->base.input_gradients = (float *)malloc(input_dim * input_dim * layer->in_channels * sizeof(float));
    }
    memset(layer->base.input_gradients, 0, input_dim * input_dim * layer->in_channels * sizeof(float));

    // Compute gradients for weights, biases, and inputs
    for (int oc = 0; oc < layer->out_channels; oc++) {
        for (size_t oy = 0; oy < output_dim; oy++) {
            for (size_t ox = 0; ox < output_dim; ox++) {
                float grad = gradients[(oc * output_dim * output_dim) + (oy * output_dim + ox)];

                // Bias gradients
                layer->base.bias_gradients[oc] += grad;

                for (int ic = 0; ic < layer->in_channels; ic++) {
                    for (int ky = 0; ky < layer->kernel_size; ky++) {
                        for (int kx = 0; kx < layer->kernel_size; kx++) {
                            size_t iy = oy * layer->stride + ky - layer->padding;
                            size_t ix = ox * layer->stride + kx - layer->padding;

                            if (iy < input_dim && ix < input_dim) {
                                size_t weight_idx = (((oc * layer->in_channels + ic) * layer->kernel_size + ky) * layer->kernel_size + kx);
                                size_t input_idx = (ic * input_dim * input_dim) + (iy * input_dim + ix);

                                // Weight gradients
                                layer->base.weight_gradients[weight_idx] += grad * layer->input[input_idx];

                                // Input gradients
                                layer->base.input_gradients[input_idx] += grad * layer->base.weights[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

void conv2d_update_weights(void *self, float learning_rate) {
    Conv2DLayer *layer = (Conv2DLayer *)self;

    printf("Updating weights for Conv2D Layer:\n");

    for (size_t i = 0; i < layer->out_channels; i++) {
        //printf("Bias[%zu]: %f -> ", i, layer->biases[i]);
        layer->biases[i] -= learning_rate * layer->base.bias_gradients[i];
        //printf("%f\n", layer->biases[i]);
    }
}


void conv2d_free(Conv2DLayer *layer) {
    free(layer->base.weights);
    free(layer->biases);
    free(layer->base.weight_gradients);
    free(layer->base.bias_gradients);
    free(layer->base.input_gradients);
    free(layer->base.output);
}
