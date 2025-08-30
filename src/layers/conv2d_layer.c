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

    //layer->deactive_out_channels = (bool*)calloc(out_channels, sizeof(bool));
    //layer->deactive_in_channels = (bool*)calloc(in_channels, sizeof(bool));  

    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    
    // 🆕 DODAJ: Store original dimensions for pruning
    layer->original_in_channels = in_channels;
    layer->original_out_channels = out_channels;
    
    // Initialize mapping pointers to NULL
    layer->in_active_channels_idx = NULL;
    layer->out_active_channels_idx = NULL;

    size_t weight_size = in_channels * out_channels * kernel_size * kernel_size;
    layer->base.weights = (float *)malloc(weight_size * sizeof(float));
    layer->base.num_weights = weight_size;
    layer->biases = (float *)malloc(out_channels * sizeof(float));

    // weights and bias initalization
    initialize_biases(layer->biases, out_channels, in_channels * kernel_size * kernel_size);

    layer->base.inputs = (float *)malloc(input_dim * input_dim * layer->in_channels  * sizeof(float));
}

void conv2d_forward(void *self, float *input, size_t input_size, size_t time_step) {
    Conv2DLayer *L = (Conv2DLayer *)self;
    int H  = L->input_dim;
    int K  = L->kernel_size;
    int P  = L->padding;
    int S  = L->stride;
    int IC = L->in_channels;        // EFFECTIVE channels (po pruning)
    int OC = L->out_channels;       // EFFECTIVE channels (po pruning)
    int ORIG_IC = L->original_in_channels;

    int OUT_H = calculate_output_dim(H, K, S, P);
    int OUT_W = OUT_H;
    int N     = OUT_H * OUT_W;

    memcpy(L->base.inputs, input, input_size * sizeof(float));

    // 🚀 UNIVERSAL LOOP: Works with and without pruning
    for (int oc_idx = 0; oc_idx < OC; ++oc_idx) {
        // Get actual output channel index
        int oc = L->out_active_channels_idx ? L->out_active_channels_idx[oc_idx] : oc_idx;
        
        for (int oy = 0; oy < OUT_H; ++oy) {
            for (int ox = 0; ox < OUT_W; ++ox) {
                float acc = L->biases[oc];
                
                for (int ic_idx = 0; ic_idx < IC; ++ic_idx) {
                    // Get actual input channel index
                    int ic = L->in_active_channels_idx ? L->in_active_channels_idx[ic_idx] : ic_idx;
                    
                    for (int ky = 0; ky < K; ++ky) {
                        int iy = oy * S + ky - P;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < K; ++kx) {
                            int ix = ox * S + kx - P;
                            if (ix < 0 || ix >= H) continue;

                            // Use ORIGINAL dimensions for weight indexing
                            int widx = oc * (ORIG_IC*K*K) + ic * (K*K) + ky * K + kx;
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

void conv2d_free(Conv2DLayer *layer) {
    free(layer->base.weights);
    free(layer->biases);
    free(layer->base.output);
    
    // 🆕 DODAJ: Free pruning arrays
    //if (layer->deactive_out_channels) free(layer->deactive_out_channels);
   // if (layer->deactive_in_channels) free(layer->deactive_in_channels);
    if (layer->out_active_channels_idx) free(layer->out_active_channels_idx);
    if (layer->in_active_channels_idx) free(layer->in_active_channels_idx);
}