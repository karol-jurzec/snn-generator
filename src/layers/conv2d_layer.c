#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <malloc.h> 
#else
#include <stdlib.h>
#endif

#include "../../include/utils/layer_utils.h" 
#include "../../include/layers/conv2d_layer.h"

void he_kaiming_uniform_init(float *weights, size_t size, int in_channels, int kernel_size) {
    float limit = sqrtf(6.0f / (in_channels * kernel_size * kernel_size));
    for (size_t i = 0; i < size; i++) {
        float rand_val = (float)rand() / RAND_MAX; 
        weights[i] = (2.0f * rand_val - 1.0f) * limit;
    }
}

void conv2d_initialize(Conv2DLayer *layer, int in_channels, int out_channels, int kernel_size, int stride, int padding, int input_dim) {
    layer->base.layer_type = LAYER_CONV2D;
    layer->base.forward = conv2d_forward;
    layer->base.is_spiking = false;
    layer->input_dim = input_dim;

    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;

    layer->original_in_channels = in_channels;
    layer->original_out_channels = out_channels;

    layer->in_channels = in_channels;
    layer->out_channels = out_channels;

    int OUT_DIM = calculate_output_dim(input_dim, kernel_size, stride, padding);
    layer->base.num_inputs = input_dim * input_dim * in_channels;
    layer->base.output_size = OUT_DIM * OUT_DIM * out_channels;
    layer->base.output = (float *)malloc(layer->base.output_size * sizeof(float));
    if (!layer->base.output) { fprintf(stderr, "conv2d_initialize: alloc output failed\n"); exit(1); }

    size_t weight_size = (size_t)out_channels * in_channels * kernel_size * kernel_size;
    layer->base.weights = (float *)malloc(weight_size * sizeof(float));
    if (!layer->base.weights) { fprintf(stderr, "conv2d_initialize: alloc weights failed\n"); exit(1); }
    layer->base.num_weights = weight_size;

    layer->biases = (float *)malloc(out_channels * sizeof(float));
    if (!layer->biases) { fprintf(stderr, "conv2d_initialize: alloc biases failed\n"); exit(1); }

    layer->base.inputs = (float *)malloc((size_t)input_dim * input_dim * layer->original_in_channels * sizeof(float));
    if (!layer->base.inputs) { fprintf(stderr, "conv2d_initialize: alloc inputs failed\n"); exit(1); }

    // init weights/biases
    he_kaiming_uniform_init(layer->base.weights, weight_size, in_channels, kernel_size);
    for (int i = 0; i < out_channels; ++i) layer->biases[i] = 0.0f;

    // pruning arrays default to NULL
    layer->in_active_channels_idx = NULL;
    layer->out_active_channels_idx = NULL;

    // --- persistent buffers (allocated with max possible sizes based on original channels)
    layer->tmp_im_pruned = NULL;
    layer->data_col = NULL;
    layer->weight_mat = NULL;
    layer->out_mat = NULL;

    layer->cache_IC = -1;
    layer->cache_OC = -1;
    layer->cache_K2 = -1;
    layer->cache_N = -1;
    layer->weight_mat_valid = false;
}

/* internal helper: ensure buffers sized for current (effective) configuration.
   Allocates or (re)allocates persistent buffers used across forwards.
*/
void ensure_forward_buffers(Conv2DLayer *L, int IC, int OC, int K2, int N) {
    int H = L->input_dim;
    // allocate tmp_im_pruned : IC * H * H
    size_t im_pruned_elems = (size_t)IC * H * H;
    if (!L->tmp_im_pruned || L->cache_IC != IC) {
        if (L->tmp_im_pruned) free(L->tmp_im_pruned);
        L->tmp_im_pruned = (float *)malloc(im_pruned_elems * sizeof(float));
        if (!L->tmp_im_pruned) { fprintf(stderr, "ensure_forward_buffers: alloc tmp_im_pruned failed\n"); exit(1); }
    }

    // data_col: (IC * K2) x N
    size_t data_col_elems = (size_t)IC * K2 * N;
    if (!L->data_col || L->cache_IC != IC || L->cache_K2 != K2 || L->cache_N != N) {
        if (L->data_col) free(L->data_col);
        L->data_col = (float *)malloc(data_col_elems * sizeof(float));
        if (!L->data_col) { fprintf(stderr, "ensure_forward_buffers: alloc data_col failed\n"); exit(1); }
    }

    // weight_mat: OC x (IC*K2)
    size_t weight_mat_elems = (size_t)OC * IC * K2;
    if (!L->weight_mat || L->cache_IC != IC || L->cache_OC != OC || L->cache_K2 != K2) {
        if (L->weight_mat) free(L->weight_mat);
        L->weight_mat = (float *)malloc(weight_mat_elems * sizeof(float));
        if (!L->weight_mat) { fprintf(stderr, "ensure_forward_buffers: alloc weight_mat failed\n"); exit(1); }
        L->weight_mat_valid = false; // must rebuild when reallocated
    }

    // out_mat: OC x N
    size_t out_mat_elems = (size_t)OC * N;
    if (!L->out_mat || L->cache_OC != OC || L->cache_N != N) {
        if (L->out_mat) free(L->out_mat);
        L->out_mat = (float *)malloc(out_mat_elems * sizeof(float));
        if (!L->out_mat) { fprintf(stderr, "ensure_forward_buffers: alloc out_mat failed\n"); exit(1); }
    }

    // update cache metadata
    L->cache_IC = IC;
    L->cache_OC = OC;
    L->cache_K2 = K2;
    L->cache_N = N;
}

/* internal helper: build compact weight matrix for effective channels
   weight_mat row-major shape: OC x (IC*K2)
*/
void build_weight_mat_if_needed(Conv2DLayer *L, int IC, int OC, int K2) {
    if (L->weight_mat_valid) return;
    int ORIG_IC = L->original_in_channels;

    size_t cols = (size_t)IC * K2;
    for (int oc_idx = 0; oc_idx < OC; ++oc_idx) {
        int oc_actual = L->out_active_channels_idx ? L->out_active_channels_idx[oc_idx] : oc_idx;
        float *wrow = L->weight_mat + (size_t)oc_idx * cols;
        for (int ic_idx = 0; ic_idx < IC; ++ic_idx) {
            int ic_actual = L->in_active_channels_idx ? L->in_active_channels_idx[ic_idx] : ic_idx;
            size_t base_widx = (size_t)oc_actual * (ORIG_IC * K2) + (size_t)ic_actual * K2;
            memcpy(wrow + (size_t)ic_idx * K2, L->base.weights + base_widx, (size_t)K2 * sizeof(float));
        }
    }
    L->weight_mat_valid = true;
}

void conv2d_forward(void *self, float *input, size_t input_size, size_t time_step) {
    Conv2DLayer *L = (Conv2DLayer *)self;
    int H  = L->input_dim;
    int K  = L->kernel_size;
    int P  = L->padding;
    int S  = L->stride;
    int IC  = L->in_channels;
    int OC  = L->out_channels;
    int ORIG_IC = L->original_in_channels;
    int ORIG_OC = L->original_out_channels;

    int OUT_H = calculate_output_dim(H, K, S, P);
    int OUT_W = OUT_H;
    int N     = OUT_H * OUT_W;
    int K2    = K * K;

    // copy raw input into base.inputs
    memcpy(L->base.inputs, input, input_size * sizeof(float));

    // ensure buffers exist / sized
    ensure_forward_buffers(L, IC, OC, K2, N);

    // build pruned contiguous input buffer (IC, H, H)
    if (L->in_active_channels_idx) {
        for (int ic_idx = 0; ic_idx < IC; ++ic_idx) {
            int ic_actual = L->in_active_channels_idx[ic_idx];
            float *src = L->base.inputs + (size_t)ic_actual * H * H;
            float *dst = L->tmp_im_pruned + (size_t)ic_idx * H * H;
            memcpy(dst, src, (size_t)H * H * sizeof(float));
        }
    } else {
        // no input pruning -> copy contiguous
        for (int ic = 0; ic < IC; ++ic) {
            float *src = L->base.inputs + (size_t)ic * H * H;
            float *dst = L->tmp_im_pruned + (size_t)ic * H * H;
            memcpy(dst, src, (size_t)H * H * sizeof(float));
        }
    }

    // im2col into data_col: shape (IC*K2, N)
    im2col(L->tmp_im_pruned, IC, H, H, K, P, S, L->data_col, OUT_H, OUT_W);

    // build weight_mat if not built (or invalidated)
    build_weight_mat_if_needed(L, IC, OC, K2);

    // GEMM using cblas_sgemm
    // weight_mat: (OC x (IC*K2)) in row-major
    // data_col: (IC*K2 x N) in row-major
    // out_mat = weight_mat * data_col
    // cblas uses row-major parameters: CblasRowMajor
    // M = OC, N = N, K = IC*K2
    int GEMM_M = OC;
    int GEMM_N = N;
    int GEMM_K = IC * K2;

    // cblas_sgemm: CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc
    // For row-major, lda = K (number of columns in A), ldb = N (number of columns in B), ldc = N (columns in C)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                GEMM_M, GEMM_N, GEMM_K,
                1.0f,
                L->weight_mat, GEMM_K,
                L->data_col, GEMM_N,
                0.0f,
                L->out_mat, GEMM_N
            );

    // add biases and scatter into L->base.output using original channel slots
    // Optionally zero whole output buffer first
    // If you want inactive outputs to be zero, uncomment the next line:
    // memset(L->base.output, 0, (size_t)ORIG_OC * N * sizeof(float));

    for (int oc_idx = 0; oc_idx < OC; ++oc_idx) {
        int oc_actual = L->out_active_channels_idx ? L->out_active_channels_idx[oc_idx] : oc_idx;
        float bias = L->biases[oc_actual];
        float *src_row = L->out_mat + (size_t)oc_idx * N;
        float *dst = L->base.output + (size_t)oc_actual * N;
        for (int n = 0; n < N; ++n) {
            dst[n] = src_row[n] + bias;
        }
    }
}

/* Invalidate weight_mat whenever pruning changes.
   Call this from whatever code performs pruning (or directly set weight_mat_valid=false).
*/
void conv2d_invalidate_weight_mat(Conv2DLayer *L) {
    L->weight_mat_valid = false;
}

void conv2d_free(Conv2DLayer *layer) {
    if (!layer) return;
    if (layer->base.weights) free(layer->base.weights);
    if (layer->biases) free(layer->biases);
    if (layer->base.output) free(layer->base.output);
    if (layer->base.inputs) free(layer->base.inputs);

    if (layer->out_active_channels_idx) free(layer->out_active_channels_idx);
    if (layer->in_active_channels_idx) free(layer->in_active_channels_idx);

    // persistent buffers
    if (layer->tmp_im_pruned) free(layer->tmp_im_pruned);
    if (layer->data_col) free(layer->data_col);
    if (layer->weight_mat) free(layer->weight_mat);
    if (layer->out_mat) free(layer->out_mat);
}