#include <stdlib.h>
#include <stdio.h>

typedef struct {
    int in_features;
    int out_features;
    float **weights;
    float *biases;
} LinearLayer;

void initialize_linear_layer(LinearLayer *layer, int in_features, int out_features) {
    layer->in_features = in_features;
    layer->out_features = out_features;

    layer->weights = allocate_2d_array(out_features, in_features);
    layer->biases = (float *)malloc(out_features * sizeof(float));


    // rand biases and weight for testing TODO; 
    for (int i = 0; i < out_features; i++) {
        for (int j = 0; j < in_features; j++) {
            layer->weights[i][j] = (float)rand() / RAND_MAX * 2 - 1; 
        }
        layer->biases[i] = (float)rand() / RAND_MAX * 2 - 1;
    }
}

float *linear_forward(LinearLayer *layer, float *input) {
    float *output = (float *)malloc(layer->out_features * sizeof(float));

    for (int i = 0; i < layer->out_features; i++) {
        output[i] = layer->biases[i];
        for (int j = 0; j < layer->in_features; j++) {
            output[i] += layer->weights[i][j] * input[j];
        }
    }
    return output;
}

void free_linear_layer(LinearLayer *layer) {
    for (int i = 0; i < layer->out_features; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->biases);
}
