#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../../include/utils/layer_utils.h" 
#include "../../include/layers/maxpool2d_layer.h"

void maxpool2d_initialize(MaxPool2DLayer *layer, int kernel_size, int stride, int padding, int input_dim, int num_of_channels) {
    layer->base.layer_type = LAYER_MAXPOOL2D;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->base.is_spiking = false;
    layer->padding = padding;
    layer->input_dim = input_dim;
    layer->num_of_channels = num_of_channels;

    size_t C = num_of_channels;
    size_t H = input_dim;
    size_t W = input_dim;

    size_t input_size = C * H * W;
    size_t output_H = (H - kernel_size + 2 * padding) / stride + 1;
    size_t output_W = (W - kernel_size + 2 * padding) / stride + 1;
    size_t output_size = C * output_H * output_W;

    // Save sizes
    layer->input_size = input_size;
    layer->base.num_inputs = input_size;
    layer->base.output_size = output_size;

    // Allocate memory
    layer->base.inputs = (float *)malloc(input_size * sizeof(float));
    layer->base.output = (float *)malloc(output_size * sizeof(float));
    layer->max_indices = (size_t *)malloc(output_size * sizeof(size_t));

    // Optional: NULL for now
    layer->base.input_gradients = NULL;

    // Function pointers
    layer->base.forward = maxpool2d_forward;
    layer->base.backward = maxpool2d_backward;
    layer->base.zero_grad = maxpool2d_zero_grad;  // Assign function pointer

    layer->base.update_weights = NULL;
}



void maxpool2d_forward(void *self, float *input, size_t input_size, size_t time_step) {
    MaxPool2DLayer *layer = (MaxPool2DLayer *)self;

    memcpy(layer->base.inputs, input, input_size * sizeof(float));

    size_t C = layer->num_of_channels;
    size_t H = layer->input_dim;
    size_t W = layer->input_dim;
    size_t output_H = (H - layer->kernel_size + 2 * layer->padding) / layer->stride + 1;
    size_t output_W = (W - layer->kernel_size + 2 * layer->padding) / layer->stride + 1;

    for (size_t c = 0; c < C; c++) {
        for (size_t oh = 0; oh < output_H; oh++) {
            for (size_t ow = 0; ow < output_W; ow++) {
                float max_val = -INFINITY;
                size_t max_idx = 0;

                for (size_t kh = 0; kh < layer->kernel_size; kh++) {
                    for (size_t kw = 0; kw < layer->kernel_size; kw++) {
                        size_t ih = oh * layer->stride + kh - layer->padding;
                        size_t iw = ow * layer->stride + kw - layer->padding;

                        if (ih < H && iw < W) {
                            size_t input_idx = c * H * W + ih * W + iw;
                            if (input[input_idx] > max_val) {
                                max_val = input[input_idx];
                                max_idx = input_idx;
                            }
                        }
                    }
                }

                size_t output_idx = c * output_H * output_W + oh * output_W + ow;
                layer->base.output[output_idx] = max_val;
                layer->max_indices[output_idx] = max_idx;
            }
        }
    }

    // if (layer->max_indices_history) {
    //     memcpy(&layer->max_indices_history[time_step * layer->base.output_size],
    //            layer->max_indices,
    //            layer->base.output_size * sizeof(size_t));
    // }
}

float* maxpool2d_backward(void *self, float *gradients, size_t time_step) {
    MaxPool2DLayer *layer = (MaxPool2DLayer *)self;
    size_t C = layer->num_of_channels;
    size_t H = layer->input_dim;
    size_t W = layer->input_dim;
    size_t input_size = C * H * W;

    float *new_gradients = (float *)calloc(input_size, sizeof(float));
    if (!new_gradients) {
        fprintf(stderr, "Error: Memory allocation failed during backward pass.\n");
        return NULL;
    }

    // Load max_indices for this time step
    size_t *max_indices = layer->max_indices;
    // if (layer->max_indices_history) {
    //     max_indices = &layer->max_indices_history[time_step * layer->base.output_size];
    // }

    // Route gradients using max_indices
    for (size_t i = 0; i < layer->base.output_size; i++) {
        size_t max_idx = max_indices[i];
        if (max_idx < input_size) {
            new_gradients[max_idx] += gradients[i];
        }
    }

    
    layer->base.input_gradients = new_gradients;

    return layer->base.input_gradients;
}

void maxpool2d_zero_grad(void *self) {
    MaxPool2DLayer *layer = (MaxPool2DLayer *)self;
    
    // Zero out input gradients if they exist
    if (layer->base.input_gradients) {
        memset(layer->base.input_gradients, 0, layer->input_size * sizeof(float));
    }
}


void maxpool2d_free(MaxPool2DLayer *layer) {
    free(layer->base.output);
    free(layer->max_indices);
    free(layer->base.input_gradients);
}
