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
    int output_dim = calculate_output_dim(layer->input_dim, layer->kernel_size, layer->stride, layer->padding);
    int kernel_dim = layer->in_channels * layer->kernel_size * layer->kernel_size;
    int num_outputs = output_dim * output_dim;

    memcpy(layer->base.inputs, input, input_size * sizeof(float));

    // im2col: convert input into column matrix
    float* col = (float *)malloc(kernel_dim * num_outputs * sizeof(float));
    im2col(input, layer->in_channels, layer->input_dim, layer->input_dim,
           layer->kernel_size, layer->padding, layer->stride,
           col, output_dim, output_dim);

    // GEMM: output = W * col
    // W: [out_channels x kernel_dim], col: [kernel_dim x num_outputs] => output: [out_channels x num_outputs]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                layer->out_channels, num_outputs, kernel_dim,
                1.0f,
                layer->base.weights, kernel_dim,
                col, num_outputs,
                0.0f,
                layer->base.output, num_outputs);

    // Add biases
    for (int oc = 0; oc < layer->out_channels; ++oc) {
        for (int i = 0; i < num_outputs; ++i) {
            layer->base.output[oc * num_outputs + i] += layer->biases[oc];
        }
    }

    free(col);
}



float* conv2d_backward(void *self, float *gradients, size_t time_step) {
    Conv2DLayer *L = (Conv2DLayer*)self;
    int H  = L->input_dim;
    int K  = L->kernel_size;
    int P  = L->padding;
    int S  = L->stride;
    int IC = L->in_channels;
    int OC = L->out_channels;

    int OUT_H = calculate_output_dim(H, K, S, P);
    int OUT_W = OUT_H;
    int N     = OUT_H * OUT_W;         // num outputs per channel
    int KD    = IC * K * K;            // flattened kernel dim

    // 1) unfold input and gradients into matrices
    float *X_col = malloc(sizeof(float) * KD * N);
    im2col(L->base.inputs, IC, H, H, K, P, S, X_col, OUT_H, OUT_W);

    float *dY = gradients;             // already shaped [OC * OUT_H * OUT_W]
    // we can treat dY as [OC x N] in row-major

    // 2) compute weight gradients: dW = dY * X_col^T
    // L->base.weight_gradients is [OC x KD]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                OC, KD, N,
                1.0f,
                dY,   N,        // A: [OC x N]
                X_col, N,       // B: [KD x N] (we transpose B)
                1.0f,           // accumulate
                L->base.weight_gradients, KD);

    // 3) accumulate bias gradients: sum each row of dY
    for(int oc=0; oc<OC; ++oc){
      float acc = 0.0f;
      float *row = dY + oc*N;
      for(int i=0; i<N; ++i) acc += row[i];
      L->base.bias_gradients[oc] += acc;
    }

    // 4) compute input‑gradients in column form: dX_col = W^T * dY
    float *dX_col = malloc(sizeof(float) * KD * N);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                KD, N, OC,
                1.0f,
                L->base.weights, KD,  // W: [OC x KD]
                dY,               N,  // dY: [OC x N]
                0.0f,
                dX_col,           N); // dX_col: [KD x N]

    // 5) fold columns back to spatial gradients
    col2im(dX_col, IC, H, H, K, P, S,
           L->base.input_gradients, OUT_H, OUT_W);

    free(X_col);
    free(dX_col);

    return L->base.input_gradients;
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
