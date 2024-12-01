#include <stdlib.h>
#include <stdio.h>

typedef struct {
    int kernel_size;
    int stride;
} MaxPool2DLayer;

void initialize_maxpool2d_layer(MaxPool2DLayer *layer, int kernel_size, int stride) {
    layer->kernel_size = kernel_size;
    layer->stride = stride;
}

float **maxpool2d_forward(MaxPool2DLayer *layer, float **input, int input_size) {
    int output_size = (input_size - layer->kernel_size) / layer->stride + 1;
    float **output = allocate_2d_array(output_size, output_size);

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            float max_val = -__FLT_MAX__;
            for (int ki = 0; ki < layer->kernel_size; ki++) {
                for (int kj = 0; kj < layer->kernel_size; kj++) {
                    int input_i = i * layer->stride + ki;
                    int input_j = j * layer->stride + kj;

                    if (input_i < input_size && input_j < input_size) {
                        if (input[input_i][input_j] > max_val) {
                            max_val = input[input_i][input_j];
                        }
                    }
                }
            }
            output[i][j] = max_val;
        }
    }
    return output;
}