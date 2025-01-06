#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../../include/utils/layer_utils.h" 
#include "../../include/layers/maxpool2d_layer.h"

void maxpool2d_initialize(MaxPool2DLayer *layer, int kernel_size, int stride, int padding) {
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;

    size_t input_dim = 28;
    size_t output_dim = calculate_output_dim(input_dim, kernel_size, stride, padding);

    layer->max_indices = NULL;
    layer->base.input_gradients = NULL;

    layer->base.output_size = output_dim * output_dim;
    layer->base.output = (float *)malloc(layer->base.output_size * sizeof(float));
    layer->base.forward = maxpool2d_forward;
    layer->base.backward = maxpool2d_backward;
    layer->base.update_weights = NULL; 
}

void maxpool2d_forward(void *self, float *input, size_t input_size) {
    MaxPool2DLayer *layer = (MaxPool2DLayer *)self;

    printf("Maxpool2d input: \n");

    for(int i = 0; i < 10; i++) {
        if(input[i] > 0) {
            printf("input[%d] = %f\n", i, input[i]);
        }
    }

    size_t input_dim = (size_t)sqrt(input_size); // Assuming square input
    size_t output_dim = (input_dim - layer->kernel_size) / layer->stride + 1;

    // Reallocate memory for output and max_indices
    layer->base.output = (float *)realloc(layer->base.output, output_dim * output_dim * sizeof(float));
    layer->max_indices = (size_t *)realloc(layer->max_indices, output_dim * output_dim * sizeof(size_t));

    if (!layer->base.output || !layer->max_indices) {
        fprintf(stderr, "Error: Memory allocation failed during forward pass.\n");
        return;
    }

    for (size_t oy = 0; oy < output_dim; oy++) {
        for (size_t ox = 0; ox < output_dim; ox++) {
            float max_val = -INFINITY; // Smallest possible value
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

void maxpool2d_backward(void *self, float *gradients) {
    MaxPool2DLayer *layer = (MaxPool2DLayer *)self;

    // Calculate the size of the input gradients array
    size_t input_dim = layer->stride * sqrt(layer->base.output_size); // Approximate input size
    size_t input_size = input_dim * input_dim;

    // Safely allocate or reallocate memory for input gradients
    float *new_gradients = (float *)malloc(input_size * sizeof(float));
    if (!new_gradients) {
        fprintf(stderr, "Error: Memory allocation failed for input_gradients.\n");
        return;
    }
    memset(new_gradients, 0, input_size * sizeof(float)); // Initialize to zero

    // Route gradients to the correct input indices using max_indices
    for (size_t i = 0; i < layer->base.output_size; i++) {
        size_t max_idx = layer->max_indices[i];

        // Ensure max_idx is within bounds
        if (max_idx >= input_size) {
            fprintf(stderr, "Error: max_idx (%zu) out of bounds (input_size: %zu).\n", max_idx, input_size);
            free(new_gradients); // Free allocated memory
            return;
        }

        new_gradients[max_idx] += gradients[i];
    }

    // Free the old input_gradients memory and assign the new array
    free(layer->base.input_gradients);
    layer->base.input_gradients = new_gradients;
}

void maxpool2d_free(MaxPool2DLayer *layer) {
    free(layer->base.output);
    free(layer->max_indices);
    free(layer->base.input_gradients);
}
