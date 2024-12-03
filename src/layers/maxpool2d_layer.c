#include <stdlib.h>
#include <stdio.h>

#include "../../include/utils/layer_utils.h" 
#include "../../include/layers/maxpool2d_layer.h"

// Initialize MaxPool2D layer
void maxpool2d_initialize(MaxPool2DLayer *layer, int kernel_size, int stride, int padding) {
    layer->base.forward = maxpool2d_forward;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;

    size_t input_dim = 28;
    size_t output_dim = calculate_output_dim(input_dim, kernel_size, stride, padding);

    layer->output_size = output_dim * output_dim;
    layer->output = (float *)malloc(layer->output_size * sizeof(float));
}

// Max pooling operation (adjusted to match forward function signature)
void maxpool2d_forward(void *self, float *input, size_t input_size) {
    MaxPool2DLayer *layer = (MaxPool2DLayer *)self;
    size_t input_dim = 28; // Assume fixed input size for now
    size_t output_dim = calculate_output_dim(input_dim, layer->kernel_size, layer->stride, layer->padding);

    printf("Performing MaxPool2D forward pass...\n");

    for (size_t oy = 0; oy < output_dim; oy++) {
        for (size_t ox = 0; ox < output_dim; ox++) {
            float max_value = -__FLT_MAX__;
            for (int ky = 0; ky < layer->kernel_size; ky++) {
                for (int kx = 0; kx < layer->kernel_size; kx++) {
                    size_t ix = ox * layer->stride + kx - layer->padding;
                    size_t iy = oy * layer->stride + ky - layer->padding;
                    if (ix < input_dim && iy < input_dim) {
                        size_t input_idx = (iy * input_dim + ix);
                        if (input[input_idx] > max_value) {
                            max_value = input[input_idx];
                        }
                    }
                }
            }
            size_t output_idx = (oy * output_dim + ox);
            layer->output[output_idx] = max_value;
        }
    }
}

// Free allocated memory
void maxpool2d_free(MaxPool2DLayer *layer) {
    free(layer->output);
}
