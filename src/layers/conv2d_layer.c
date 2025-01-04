#include <stdlib.h>
#include <stdio.h>

#include "../../include/utils/layer_utils.h" 
#include "../../include/layers/conv2d_layer.h"

void conv2d_initialize(Conv2DLayer *layer, int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    layer->base.forward = conv2d_forward;
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;

    size_t weight_size = in_channels * out_channels * kernel_size * kernel_size;
    layer->weights = (float *)malloc(weight_size * sizeof(float));
    layer->biases = (float *)malloc(out_channels * sizeof(float));

    for (size_t i = 0; i < weight_size; i++) {
        layer->weights[i] = ((float)rand() / RAND_MAX) * 0.1f;
    }
    for (size_t i = 0; i < out_channels; i++) {
        layer->biases[i] = 0.0f;
    }

    size_t input_dim = 28;
    layer->base.output_size = calculate_output_dim(input_dim, kernel_size, stride, padding) *
                         calculate_output_dim(input_dim, kernel_size, stride, padding) *
                         out_channels;
    layer->base.output = (float *)malloc(layer->base.output_size * sizeof(float));
}

void conv2d_forward(void *self, float *input, size_t input_size) {
    Conv2DLayer *layer = (Conv2DLayer *)self;
    size_t output_dim = calculate_output_dim(28, layer->kernel_size, layer->stride, layer->padding);

    printf("Performing Conv2D forward pass...\n");

    for (int oc = 0; oc < layer->out_channels; oc++) {
        for (size_t oy = 0; oy < output_dim; oy++) {
            for (size_t ox = 0; ox < output_dim; ox++) {
                float sum = 0.0f;
                for (int ic = 0; ic < layer->in_channels; ic++) {
                    for (int ky = 0; ky < layer->kernel_size; ky++) {
                        for (int kx = 0; kx < layer->kernel_size; kx++) {
                            size_t ix = ox * layer->stride + kx - layer->padding;
                            size_t iy = oy * layer->stride + ky - layer->padding;
                            if (ix < 28 && iy < 28) {
                                size_t input_idx = (ic * input_size) + (iy * 28 + ix);
                                size_t weight_idx = (((oc * layer->in_channels + ic) * layer->kernel_size + ky) * layer->kernel_size + kx);
                                sum += input[input_idx] * layer->weights[weight_idx];
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

void conv2d_free(Conv2DLayer *layer) {
    free(layer->weights);
    free(layer->biases);
    free(layer->base.output);
}
