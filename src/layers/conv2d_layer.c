#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../../include/utils/layer_utils.h" 
#include "../../include/layers/conv2d_layer.h"

void he_kaiming_uniform_init(float *weights, size_t size, int in_channels, int kernel_size) {
    float limit = sqrtf(6.0f / (in_channels * kernel_size * kernel_size));
    for (size_t i = 0; i < size; i++) {
        float rand_val = (float)rand() / RAND_MAX; // [0, 1]
        weights[i] = (2.0f * rand_val - 1.0f) * limit; // Scale to [-limit, limit]
    }
}

void conv2d_initialize(Conv2DLayer *layer, int in_channels, int out_channels, int kernel_size, int stride, int padding, int input_dim) {
    //size_t input_dim = 28;
    layer->base.layer_type = LAYER_CONV2D;
    layer->base.forward = conv2d_forward;
    layer->base.is_spiking = false;
    layer->base.num_inputs = input_dim * input_dim * in_channels; 
    layer->base.backward = conv2d_backward;
    layer->base.zero_grad = conv2d_zero_grad;  // Assign function pointer
    layer->base.update_weights = conv2d_update_weights;
    layer->base.output_size = calculate_output_dim(input_dim, kernel_size, stride, padding) *
                              calculate_output_dim(input_dim, kernel_size, stride, padding) *
                              out_channels;
    layer->base.output = (float *)malloc(layer->base.output_size * sizeof(float));

    layer->input_dim = input_dim;
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;

    size_t weight_size = in_channels * out_channels * kernel_size * kernel_size;
    layer->base.weights = (float *)malloc(weight_size * sizeof(float));
    layer->base.num_weights = weight_size;
    layer->biases = (float *)malloc(out_channels * sizeof(float));

    // weights and bias initalization
    he_kaiming_uniform_init(layer->base.weights, weight_size, in_channels, kernel_size);
    initialize_biases(layer->biases, out_channels, in_channels * kernel_size * kernel_size);

    layer->base.weight_gradients = (float *)malloc(weight_size * sizeof(float));
    layer->base.bias_gradients = (float *)malloc(out_channels * sizeof(float));
    layer->base.input_gradients = (float *)malloc(input_dim * input_dim * layer->in_channels * sizeof(float));
    layer->base.inputs = (float *)malloc(input_dim * input_dim * layer->in_channels  * sizeof(float));
}

void conv2d_forward(void *self, float *input, size_t input_size, size_t time_step) {
    Conv2DLayer *layer = (Conv2DLayer *)self;
    size_t output_dim = calculate_output_dim(layer->input_dim, layer->kernel_size, layer->stride, layer->padding);
    size_t output_size = layer->out_channels * output_dim * output_dim;


    memcpy(layer->base.inputs, input, input_size * sizeof(float));

    // Perform convolution
    for (int oc = 0; oc < layer->out_channels; oc++) {
        for (size_t oy = 0; oy < output_dim; oy++) {
            for (size_t ox = 0; ox < output_dim; ox++) {
                float sum = 0.0f;
                for (int ic = 0; ic < layer->in_channels; ic++) {
                    for (int ky = 0; ky < layer->kernel_size; ky++) {
                        for (int kx = 0; kx < layer->kernel_size; kx++) {
                            size_t ix = ox * layer->stride + kx - layer->padding;
                            size_t iy = oy * layer->stride + ky - layer->padding;
                            if (ix < layer->input_dim && iy < layer->input_dim) {
                                size_t input_idx = ic * layer->input_dim * layer->input_dim + iy * layer->input_dim + ix;
                                size_t weight_idx = ((oc * layer->in_channels + ic) * layer->kernel_size + ky) * layer->kernel_size + kx;
                                sum += input[input_idx] * layer->base.weights[weight_idx];
                            }
                        }
                    }
                }
                size_t output_idx = oc * output_dim * output_dim + oy * output_dim + ox;
                layer->base.output[output_idx] = sum + layer->biases[oc];
            }
        }
    }

    // Store output for this time step
    // if (layer->base.output_history) {
    //     memcpy(&layer->base.output_history[time_step * output_size], 
    //            layer->base.output, 
    //            output_size * sizeof(float));
    // }
}


float* conv2d_backward(void *self, float *gradients, size_t time_step) {
    Conv2DLayer *layer = (Conv2DLayer *)self;
    size_t input_dim = layer->input_dim;
    size_t output_dim = calculate_output_dim(input_dim, layer->kernel_size, layer->stride, layer->padding);

    // Load input for this time step
    float* input = layer->base.inputs;
    // if (layer->base.output_history) {
    //     input = &layer->base.output_history[time_step * layer->base.output_size];
    // }

    // Zero input gradients for this time step
    // memset(layer->base.input_gradients, 0, layer->input_dim * layer->input_dim * layer->in_channels * sizeof(float));

    for (int oc = 0; oc < layer->out_channels; oc++) {
        for (size_t oy = 0; oy < output_dim; oy++) {
            for (size_t ox = 0; ox < output_dim; ox++) {
                float grad = gradients[(oc * output_dim * output_dim) + (oy * output_dim + ox)];
                
                // Bias gradients (accumulate across time)
                layer->base.bias_gradients[oc] += grad;

                for (int ic = 0; ic < layer->in_channels; ic++) {
                    for (int ky = 0; ky < layer->kernel_size; ky++) {
                        for (int kx = 0; kx < layer->kernel_size; kx++) {
                            size_t iy = oy * layer->stride + ky - layer->padding;
                            size_t ix = ox * layer->stride + kx - layer->padding;
                            
                            if (iy < input_dim && ix < input_dim) {
                                size_t weight_idx = (((oc * layer->in_channels + ic) * layer->kernel_size + ky) * layer->kernel_size + kx);
                                size_t input_idx = (ic * input_dim * input_dim) + (iy * input_dim + ix);
                                
                                // Weight gradients (accumulate across time)
                                layer->base.weight_gradients[weight_idx] += grad * input[input_idx];
                                
                                // Input gradients (time-step specific)
                                layer->base.input_gradients[input_idx] += grad * layer->base.weights[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    return layer->base.input_gradients;
}

void conv2d_update_weights(void *self, float learning_rate) {
    Conv2DLayer *layer = (Conv2DLayer *)self;

    for (size_t i = 0; i < layer->out_channels; i++) {
        layer->biases[i] -= learning_rate * layer->base.bias_gradients[i];
    }

    size_t num_weights = layer->in_channels * layer->out_channels * layer->kernel_size * layer->kernel_size;
    for (size_t i = 0; i < num_weights; i++) {
        layer->base.weights[i] -= learning_rate * layer->base.weight_gradients[i];
    }
}

void conv2d_zero_grad(void *self) {
    Conv2DLayer *layer = (Conv2DLayer *)self;
    memset(layer->base.weight_gradients, 0, sizeof(float) * layer->in_channels * layer->out_channels * layer->kernel_size * layer->kernel_size);
    memset(layer->base.bias_gradients, 0, sizeof(float) * layer->out_channels);
    memset(layer->base.input_gradients, 0, sizeof(float) * layer->input_dim * layer->input_dim * layer->in_channels);
}


void conv2d_free(Conv2DLayer *layer) {
    free(layer->base.weights);
    free(layer->biases);
    free(layer->base.weight_gradients);
    free(layer->base.bias_gradients);
    free(layer->base.input_gradients);
    free(layer->base.output);
}
