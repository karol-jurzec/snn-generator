#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../../include/utils/layer_utils.h" 
#include "../../include/layers/maxpool2d_layer.h"

void maxpool2d_initialize(MaxPool2DLayer *layer, int kernel_size, int stride, int padding, int input_dim, int num_of_channels) {
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->input_dim = input_dim;
    layer->num_of_channels = num_of_channels;
    layer->base.num_inputs = 0;


    layer->base.num_inputs = input_dim * input_dim * num_of_channels;

    //size_t input_dim = 28;
    //size_t output_dim = calculate_output_dim(input_dim, kernel_size, stride, padding);

    layer->max_indices = NULL;
    layer->base.input_gradients = NULL;

    //layer->base.output_size = output_dim * output_dim;
    //layer->base.output = (float *)malloc(layer->base.output_size * sizeof(float));

    layer->input_size = 0;
    layer->base.forward = maxpool2d_forward;
    layer->base.backward = maxpool2d_backward;
    layer->base.update_weights = NULL; 
}

/*
void maxpool2d_forward(void *self, float *input, size_t input_size) {
    MaxPool2DLayer *layer = (MaxPool2DLayer *)self;

    size_t input_dim = (size_t)sqrt(input_size); 
    size_t output_dim = (input_dim - layer->kernel_size) / layer->stride + 1;

    layer->base.output = (float *)realloc(layer->base.output, output_dim * output_dim * sizeof(float));
    layer->max_indices = (size_t *)realloc(layer->max_indices, output_dim * output_dim * sizeof(size_t));
    layer->input_size = input_size;

    if (!layer->base.output || !layer->max_indices) {
        fprintf(stderr, "Error: Memory allocation failed during forward pass.\n");
        return;
    }

    for (size_t oy = 0; oy < output_dim; oy++) {
        for (size_t ox = 0; ox < output_dim; ox++) {
            float max_val = -INFINITY; // smallest possible value
            size_t max_idx = 0;

            for (size_t ky = 0; ky < layer->kernel_size; ky++) {
                for (size_t kx = 0; kx < layer->kernel_size; kx++) {
                    size_t iy = oy * layer->stride + ky - layer->padding;
                    size_t ix = ox * layer->stride + kx - layer->padding;

                    if (iy < input_dim && ix < input_dim) {
                        size_t input_idx = iy * input_dim + ix;
                        if (input[input_idx] > max_val) {
                            max_val = input[input_idx];
                            max_idx = input_idx;
                        }
                    }
                }
            }

            size_t output_idx = oy * output_dim + ox;
            layer->base.output[output_idx] = max_val;
            layer->max_indices[output_idx] = max_idx;
        }
    }
}
*/

void maxpool2d_forward(void *self, float *input, size_t input_size) {
    MaxPool2DLayer *layer = (MaxPool2DLayer *)self;

    layer->base.inputs = input;
    
    size_t C = layer->num_of_channels; // Number of channels
    size_t H = layer->input_dim; // Input height
    size_t W = layer->input_dim; // Input width

    size_t output_H = (H - layer->kernel_size + 2 * layer->padding) / layer->stride + 1;
    size_t output_W = (W - layer->kernel_size + 2 * layer->padding) / layer->stride + 1;

    layer->base.output = (float *)realloc(layer->base.output, C * output_H * output_W * sizeof(float));
    layer->base.output_size = C * output_H * output_W;
    layer->max_indices = (size_t *)realloc(layer->max_indices, C * output_H * output_W * sizeof(size_t));

    if (layer->base.output == NULL || layer->max_indices == NULL) {
        fprintf(stderr, "Error: Memory allocation failed during forward pass.\n");
        return;
    }

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
}

/*
float* maxpool2d_backward(void *self, float *gradients) {
    MaxPool2DLayer *layer = (MaxPool2DLayer *)self;

    size_t input_size = layer->input_size; // Must match 6912

    // Allocate memory safely
    float *new_gradients = (float *)realloc(input_size, sizeof(float));
    if (!new_gradients) {
        fprintf(stderr, "Error: Memory allocation failed during backward pass.\n");
        return NULL;
    }

    memset(new_gradients, 0, input_size * sizeof(float)); // Explicitly zero-out memory

    // Route gradients using max_indices
    for (size_t i = 0; i < layer->base.output_size; i++) {
        size_t max_idx = layer->max_indices[i];
        if (max_idx < input_size) {
            new_gradients[max_idx] += gradients[i];
        }
    }

    // Free old memory only if allocated
    if (layer->base.input_gradients) {
        free(layer->base.input_gradients);
    }
    layer->base.input_gradients = new_gradients;

    return layer->base.input_gradients;
}*/

float* maxpool2d_backward(void *self, float *gradients) {
    MaxPool2DLayer *layer = (MaxPool2DLayer *)self;

    size_t C = layer->num_of_channels;  // Number of channels
    size_t H = layer->input_dim;        // Input height
    size_t W = layer->input_dim;        // Input width
    size_t input_size = C * H * W; // Total input size (6912)

    float *new_gradients = (float *)calloc(input_size, sizeof(float)); // Zero-initialized
    if (!new_gradients) {
        fprintf(stderr, "Error: Memory allocation failed during backward pass.\n");
        return NULL;
    }

    size_t output_H = (H - layer->kernel_size + 2 * layer->padding) / layer->stride + 1;
    size_t output_W = (W - layer->kernel_size + 2 * layer->padding) / layer->stride + 1;
    size_t output_size = C * output_H * output_W; // Total output size

    // Route gradients using max_indices
    for (size_t i = 0; i < output_size; i++) {
        size_t max_idx = layer->max_indices[i];
        if (max_idx < input_size) {
            new_gradients[max_idx] += gradients[i];
        }
    }

    // Free old memory if allocated
    if (layer->base.input_gradients) {
        free(layer->base.input_gradients);
    }
    layer->base.input_gradients = new_gradients;

    return layer->base.input_gradients;
}




void maxpool2d_free(MaxPool2DLayer *layer) {
    free(layer->base.output);
    free(layer->max_indices);
    free(layer->base.input_gradients);
}
