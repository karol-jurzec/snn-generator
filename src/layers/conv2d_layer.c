#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <malloc.h>  // dla _aligned_malloc i _aligned_free
#else
#include <stdlib.h>  // dla aligned_alloc
#endif

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

    layer->deactive_out_channels = (bool*)calloc(out_channels, sizeof(bool));

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
    Conv2DLayer *L = (Conv2DLayer *)self;
    int H  = L->input_dim;
    int K  = L->kernel_size;
    int P  = L->padding;
    int S  = L->stride;
    int IC = L->in_channels;
    int OC = L->out_channels;

    int OUT_H = calculate_output_dim(H, K, S, P);
    int OUT_W = OUT_H;
    int N     = OUT_H * OUT_W;

    memcpy(L->base.inputs, input, input_size * sizeof(float));

    for (int oc = 0; oc < OC; ++oc) {
        for (int oy = 0; oy < OUT_H; ++oy) {
            for (int ox = 0; ox < OUT_W; ++ox) {
                float acc = L->biases[oc];
                for (int ic = 0; ic < IC; ++ic) {
                    for (int ky = 0; ky < K; ++ky) {
                        int iy = oy * S + ky - P;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < K; ++kx) {
                            int ix = ox * S + kx - P;
                            if (ix < 0 || ix >= H) continue;

                            int widx = oc * (IC*K*K) + ic * (K*K) + ky * K + kx;
                            int iidx = ic * (H*H) + iy * H + ix;
                            acc += L->base.weights[widx] * input[iidx];
                        }
                    }
                }
                L->base.output[oc * N + oy * OUT_W + ox] = acc;
            }
        }
    }
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
    int N     = OUT_H * OUT_W;

    // db += sum(dY)
    for (int oc = 0; oc < OC; ++oc) {
        float accb = 0.0f;
        for (int oy = 0; oy < OUT_H; ++oy) {
            for (int ox = 0; ox < OUT_W; ++ox) {
                accb += gradients[oc * N + oy * OUT_W + ox];
            }
        }
        L->base.bias_gradients[oc] += accb;
    }

    // dW += dY * X (konwolucyjne)
    for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    float acc = 0.0f;
                    for (int oy = 0; oy < OUT_H; ++oy) {
                        int iy = oy * S + ky - P;
                        if (iy < 0 || iy >= H) continue;
                        for (int ox = 0; ox < OUT_W; ++ox) {
                            int ix = ox * S + kx - P;
                            if (ix < 0 || ix >= H) continue;
                            float dy = gradients[oc * N + oy * OUT_W + ox];
                            float x  = L->base.inputs[ic * (H*H) + iy * H + ix];
                            acc += dy * x;
                        }
                    }
                    int widx = oc * (IC*K*K) + ic * (K*K) + ky * K + kx;
                    L->base.weight_gradients[widx] += acc;
                }
            }
        }
    }

    // dX = sum over oc,ky,kx of W * dY (transposed conv)
    // Wyzeruj bufor
    memset(L->base.input_gradients, 0, sizeof(float) * (H*H*IC));

    for (int oc = 0; oc < OC; ++oc) {
        for (int oy = 0; oy < OUT_H; ++oy) {
            for (int ox = 0; ox < OUT_W; ++ox) {
                float dy = gradients[oc * N + oy * OUT_W + ox];
                for (int ic = 0; ic < IC; ++ic) {
                    for (int ky = 0; ky < K; ++ky) {
                        int iy = oy * S + ky - P;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < K; ++kx) {
                            int ix = ox * S + kx - P;
                            if (ix < 0 || ix >= H) continue;
                            int widx = oc * (IC*K*K) + ic * (K*K) + ky * K + kx;
                            int iidx = ic * (H*H) + iy * H + ix;
                            L->base.input_gradients[iidx] += L->base.weights[widx] * dy;
                        }
                    }
                }
            }
        }
    }

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
