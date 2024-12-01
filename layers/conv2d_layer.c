#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int kernel_size;
    int stride;
    int padding;
    float **kernel;
} Conv2DLayer;

float **allocate_2d_array(int rows, int cols) {
    float **array = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++) {
        array[i] = (float *)malloc(cols * sizeof(float));
    }
    return array;
}

void initialize_conv2d_layer(Conv2DLayer *layer, int kernel_size, int stride, int padding) {
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;

    layer->kernel = allocate_2d_array(kernel_size, kernel_size);
    // random values for testing 
    // todo: implement initialization algorithm 
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            layer->kernel[i][j] = (float)rand() / RAND_MAX * 2 + 5; 
        }
    }
}

void free_conv2d_layer(Conv2DLayer *layer) {
    for (int i = 0; i < layer->kernel_size; i++) {
        free(layer->kernel[i]);
    }
    free(layer->kernel);
}

float **conv2d_forward(Conv2DLayer *layer, float **input, int input_size) {
    int output_size = (input_size - layer->kernel_size + 2 * layer->padding) / layer->stride + 1;
    float **output = allocate_2d_array(output_size, output_size);

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float sum = 0.0;

            for (int ki = 0; ki < layer->kernel_size; ++ki) {
                for (int kj = 0; kj < layer->kernel_size; ++kj) {
                    int input_i = i * layer->stride + ki - layer->padding;
                    int input_j = j * layer->stride + kj - layer->padding;

                    if (input_i >= 0 && input_i < input_size && input_j >= 0 && input_j < input_size) {
                        sum += input[input_i][input_j] * layer->kernel[ki][kj];
                    }
                }
            }

            output[i][j] = sum;
        }
    }
    return output;
}